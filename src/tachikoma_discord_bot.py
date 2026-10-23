"""
==============================================================================
TACHIKOMA Discord Bridge ＆ システム自律防衛エージェント (src/tachikoma_discord_bot.py)
==============================================================================
【概要】
Discord サーバー経由で TACHIKOMA との自然言語対話・実機マニピュレーションを行うと同時に、
常時稼働サーバー（HeroBox: Celeron J4125 / Xubuntu 24.04 ヘッドレス環境）の
リソース・セキュリティ状態を常時監視し、異常時の自動防衛およびスマホからの遠隔制御を
一元化して提供する複合型統合エージェントです。

【主要機能と動作メカニズム】

1. サーバー (Guild) 単位の厳格なアクセス制御:
   - .env の DISCORD_GUILD_ID で指定された個人専用サーバー内のみで応答。
   - サーバー内の全チャンネルおよび動的スレッドに対応しつつ、外部サーバーや
     DM（ダイレクトメッセージ）からの誤爆・不正アクセスを即座に破棄。

2. 実機ハードウェア ＆ 物理空間コンテキストの動的注入 (Grounding):
   - メッセージ受信ごとに以下の実測ステータスを非同期収集し、LLMプロンプトへ動的注入：
     ① 実機アームシリアルポート（FOLLOWER_PORT）の接続・オープン状態
     ② USB Webカメラ（/dev/video0）の物理接続・認識状況
     ③ 机上物体認識結果（config/current_world_state.json の 3D 座標・色・ID）
   - 物理環境と同期したプロンプトを構築することで、LLMのハルシネーションを防止。

3. 4変数ステップ表示と非同期実機マニピュレーション:
   - LLMが立案した Pick / Place タスクを [r (cm), theta (deg), z (mm), Roll(ID5) (deg)]
     の 4 変数形式で Embed 表示。
   - 承認ボタン（TaskConfirmView: 実行 / キャンセル）によるヒューマン・イン・ザ・ループ。
   - discord.py のイベントループ阻害（ブロッキング）を防ぐため、軌道生成および
     シリアル通信は asyncio.to_thread 経由でバックグラウンド実行。
   - is_busy フラグによる排他制御と、Discord API の 3 秒応答制限を回避する defer 先行処理。

4. 段階的防衛 (Tiered Defense) に基づくセキュリティ監視ループ:
   - discord.ext.tasks により 2 分間隔で HeroBox のカーネルおよびプロセス状態を自律巡回。
   - 【レベル1: 警告通知 (15分クールタイム付き)】
     ・ディスク使用率 90% 超過の検知 (shutil.disk_usage)
     ・CPU 異常高負荷の検知 (os.getloadavg による 1分負荷 3.8 超過)
     ・SSH 不正ログイン試行の検知 (journalctl からの直近 Failed password 集計)
   - 【レベル2: 自動防衛・即時フェイルセーフ (クールタイム無視)】
     ・/tmp, /var/tmp, /dev/shm 等の一時ディレクトリ直下で実行されている身元不明プロセスを
       psutil により常時探索。
     ・ホワイトリスト外の異常バイナリを検知した瞬間、人間の判断を待たずに当該プロセスを
       即時 kill し、Web公開アプリ (kakeibo.service) を自動停止して被害拡大・踏み台化を防止。
     ・実行結果および遮断状況を事後報告として Discord へ緊急 Embed 送信。

5. スマホ完結型・緊急制御パネル (SystemControlView):
   - アラート通知時、または !control / !status コマンド送信時に操作パネルを展開。
   - client.add_view() による永続化（Persistent View）により、Bot再起動後も
     過去のメッセージに配置されたボタンから即座にアクションが可能。
   - sudoers の NOPASSWD 設定と連携し、スマホからワンタップで以下を実行：
     ① 🛑 アーム緊急停止: abort_arm_flag による軌道生成ループ割り込み ＆ 全サーボ脱力
     ② 🛑 家計簿を停止: systemctl stop による Web 公開（ポート8501）の緊急閉鎖
     ③ ▶️ 家計簿を再開: systemctl start による安全復旧
     ④ 🔌 外部隔離: 3秒の待機猶予（通知送信完了バッファ）後に tailscale down を実行し、
        物理的なモニター・キーボード接続以外からのアクセスを遮断する最終防壁
==============================================================================
"""

"""
==============================================================================
TACHIKOMA Discord Bridge エージェント (src/tachikoma_discord_bot.py)
[段階的防衛・セキュリティ監視 ＆ 緊急キルスイッチ統合版]
==============================================================================
"""

import os
import sys
import time
import json
import shutil
import subprocess
import asyncio
import argparse
import psutil
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
from discord.ui import View, Button

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from config.joint_config import FOLLOWER_PORT, BAUDRATE, SERVO_IDS
from core.llm_planner import LLMTaskPlanner
from core.sts3215 import STS3215Driver
from core.motion_generator import ParametricMotionGenerator
from core.kinematics import raw_to_radian, radian_to_raw, get_home_radians

# ==============================================================================
# 環境変数・設定読み込み (Public公開対応: 固有値はすべて .env から注入)
# ==============================================================================
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

TARGET_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
if TARGET_GUILD_ID:
    TARGET_GUILD_ID = int(TARGET_GUILD_ID)

ALERT_CHANNEL_ID = os.getenv("DISCORD_ALERT_CHANNEL_ID")
if ALERT_CHANNEL_ID:
    ALERT_CHANNEL_ID = int(ALERT_CHANNEL_ID)

# 監視・保護対象のsystemdサービス名 (デフォルトはプレースホルダー)
PROTECTED_SERVICE = os.getenv("PROTECTED_SERVICE_NAME", "my_web_app")

# プロセスホワイトリスト (.env のカンマ区切り文字列から動的生成、未設定時は最小限の標準構成)
_env_safe_procs = os.getenv("SAFE_PROCESS_NAMES", "")
if _env_safe_procs.strip():
    SAFE_PROCESS_NAMES = {p.strip() for p in _env_safe_procs.split(",") if p.strip()}
else:
    SAFE_PROCESS_NAMES = {
        "python", "python3", "bash", "sh", "git", "pip", "sshd",
        "tailscaled", "systemd", "apt", "dpkg"
    }

# 監視閾値およびクールタイム設定
DISK_USAGE_THRESHOLD = float(os.getenv("DISK_USAGE_THRESHOLD", "90.0"))
CPU_LOAD_THRESHOLD = float(os.getenv("CPU_LOAD_THRESHOLD", "3.8"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "900"))
CAMERA_DEVICE_PATH = os.getenv("CAMERA_DEVICE_PATH", "/dev/video0")

# ロボット・システム状態管理
is_busy = False
abort_arm_flag = False
last_alert_time = 0.0

planner = LLMTaskPlanner()
generator = ParametricMotionGenerator()
follower = None

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


def get_hardware_and_world_context():
    """現在の実機接続状態、カメラ状態、机上物体情報を取得してプロンプト用テキストを生成"""
    global follower

    arm_status = (
        "接続中 (動作可能)"
        if (follower is not None and getattr(follower, "is_open", False))
        else "未接続 (実機動作不可・シミュレーションのみ)"
    )

    camera_status = "未接続 / 不明"
    if os.path.exists(CAMERA_DEVICE_PATH):
        camera_status = f"接続中 ({CAMERA_DEVICE_PATH} 検出済み)"

    world_state_path = os.path.join(BASE_DIR, "config", "current_world_state.json")
    objects_summary = "現在、机の上に検知された物体データはありません（空の状態または未計測）。"

    if os.path.exists(world_state_path):
        try:
            with open(world_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                objs = data.get("objects", [])
                if objs:
                    lines = [
                        f"- [ID #{o.get('id')}] 名称: {o.get('name', '物体')}, "
                        f"色: {o.get('color', '不明')}, "
                        f"座標: (X={o.get('x_mm', 0):.1f}mm, Y={o.get('y_mm', 0):.1f}mm, Z={o.get('z_mm', 0):.1f}mm), "
                        f"角度: {o.get('angle_deg', 0):.1f}°"
                        for o in objs
                    ]
                    objects_summary = "【机の上にある検知済み物体リスト】:\n" + "\n".join(lines)
        except Exception:
            pass

    context_prompt = f"""
[現在のタチコマ物理環境ステータス]
- 実機ロボットアーム: {arm_status}
- カメラデバイス: {camera_status}
- 机上認識状況:
{objects_summary}
※ユーザーから机の上の状況や接続状態を尋ねられた場合は、上記の実測情報を基に正確に回答してください。存在しない物体をでっち上げないでください。
"""
    return context_prompt


def execute_tasks_sync(tasks_list):
    """実機サーボへの物理搬送シーケンス（緊急停止割り込み対応・ブロッキング同期関数）"""
    global follower, abort_arm_flag
    abort_arm_flag = False

    if follower is None or not getattr(follower, "is_open", False):
        return "⚠️ 実機アームが接続されていません（プレビューのみモード）"

    try:
        for sid in SERVO_IDS:
            follower.set_torque(sid, True)

        home_rad = get_home_radians()
        for idx, task in enumerate(tasks_list, 1):
            if abort_arm_flag:
                return "🛑 緊急停止ボタンにより搬送タスクを中断し、脱力しました。"

            p = task.get("pick")
            d = task.get("place")
            success, frames, log_text = generator.generate_from_coords(p, d)
            if not success:
                continue

            for _, rad_positions in frames:
                if abort_arm_flag:
                    return "🛑 緊急停止ボタンにより動作を即時中断し、脱力しました。"
                for sid in SERVO_IDS:
                    raw_val = radian_to_raw(sid, rad_positions[sid])
                    follower.write_position(sid, raw_val)
                time.sleep(0.02)

        return "✅ すべての搬送タスクが完了しました。"
    except Exception as e:
        return f"❌ 実行中エラー: {e}"
    finally:
        for sid in SERVO_IDS:
            try:
                follower.set_torque(sid, False)
            except Exception:
                pass


# ==============================================================================
# 緊急制御パネル UI (スマホからのワンタップ停止・再開)
# ==============================================================================
class SystemControlView(View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="🛑 アーム緊急停止 (脱力)", style=discord.ButtonStyle.danger, custom_id="sys_stop_arm", row=0)
    async def stop_arm(self, interaction: discord.Interaction, button: Button):
        global is_busy, abort_arm_flag, follower
        await interaction.response.defer()
        abort_arm_flag = True
        is_busy = False

        if follower is not None and getattr(follower, "is_open", False):
            for sid in SERVO_IDS:
                try:
                    follower.set_torque(sid, False)
                except Exception:
                    pass

        await interaction.followup.send("🛑 **ロボットアームに緊急停止・脱力指令を送信しました。**")

    @discord.ui.button(label="🛑 Webアプリを停止", style=discord.ButtonStyle.danger, custom_id="sys_stop_service", row=0)
    async def stop_service(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await asyncio.to_thread(subprocess.run, ["sudo", "systemctl", "stop", PROTECTED_SERVICE])
        await interaction.followup.send(f"🛑 `{PROTECTED_SERVICE}.service` を緊急停止しました。")

    @discord.ui.button(label="▶️ Webアプリを再開", style=discord.ButtonStyle.success, custom_id="sys_start_service", row=1)
    async def start_service(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await asyncio.to_thread(subprocess.run, ["sudo", "systemctl", "start", PROTECTED_SERVICE])
        res = await asyncio.to_thread(
            subprocess.run,
            ["sudo", "systemctl", "is-active", PROTECTED_SERVICE],
            capture_output=True,
            text=True,
        )
        await interaction.followup.send(f"▶️ `{PROTECTED_SERVICE}.service` を再開しました（状態: `{res.stdout.strip()}`）。")

    @discord.ui.button(label="🔌 外部隔離 (VPN切断)", style=discord.ButtonStyle.secondary, custom_id="sys_isolate_net", row=1)
    async def isolate_net(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await interaction.followup.send(
            "🚨 **3秒後に Tailscale VPN を切断し、サーバーを外部ネットワークから完全隔離します。**\n"
            "※復旧にはサーバー本体へ物理コンソールを接続する必要があります。"
        )
        await asyncio.sleep(3)
        await asyncio.to_thread(subprocess.run, ["sudo", "tailscale", "down"])


# ==============================================================================
# 段階的防衛 (Tiered Defense) バックグラウンド監視ロジック
# ==============================================================================
def check_system_security_sync():
    """同期スレッドでシステム状態を検査し、(警告リスト, 自動防衛アクションリスト) を返す"""
    warnings = []
    auto_actions = []

    # --- [レベル1: 警告監視] ---
    usage = shutil.disk_usage("/")
    disk_pct = (usage.used / usage.total) * 100
    if disk_pct > DISK_USAGE_THRESHOLD:
        warnings.append(f"💾 **ディスク残量警告**: 使用率 `{disk_pct:.1f}%`")

    load1, _, _ = os.getloadavg()
    if load1 > CPU_LOAD_THRESHOLD:
        warnings.append(f"🔥 **CPU高負荷**: 1分平均ロード `{load1:.2f}`")

    ssh_check = subprocess.run(
        ["journalctl", "-u", "ssh", "--since", "2 minutes ago"],
        capture_output=True,
        text=True,
    )
    failed_ssh = sum(1 for line in ssh_check.stdout.splitlines() if "Failed password" in line)
    if failed_ssh > 0:
        warnings.append(f"🚨 **SSH不正ログイン試行**: 直近2分間で `{failed_ssh}` 件の失敗を検知")

    # --- [レベル2: 身元不明コードの検知 ＆ 自動停止 (フェイルセーフ)] ---
    suspicious_dirs = ("/tmp/", "/var/tmp/", "/dev/shm/")
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            p_info = proc.info
            exe_path = p_info.get("exe") or ""
            p_name = p_info.get("name") or ""

            if any(exe_path.startswith(d) for d in suspicious_dirs) and p_name not in SAFE_PROCESS_NAMES:
                pid = p_info["pid"]
                proc.kill()
                subprocess.run(["sudo", "systemctl", "stop", PROTECTED_SERVICE])
                auto_actions.append(
                    f"🛡️ **不審プロセスを自動強制終了しました**\n"
                    f"・対象: `{p_name}` (PID: `{pid}`)\n"
                    f"・実行パス: `{exe_path}`\n"
                    f"・保護措置: `{PROTECTED_SERVICE}.service` を自動停止しました。"
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return warnings, auto_actions


@tasks.loop(minutes=2)
async def security_monitor_task():
    global last_alert_time
    if not ALERT_CHANNEL_ID:
        return

    channel = bot.get_channel(ALERT_CHANNEL_ID)
    if channel is None:
        return

    warnings, auto_actions = await asyncio.to_thread(check_system_security_sync)

    if auto_actions:
        embed = discord.Embed(
            title="🚨 【自動防衛発動】不審な挙動を検知・自動停止しました",
            description="\n\n".join(auto_actions),
            color=0xFF0000,
        )
        embed.set_footer(text="安全確認後、下のボタンからサービス再開またはネットワーク隔離を選択してください。")
        await channel.send(embed=embed, view=SystemControlView())
        last_alert_time = time.time()
        return

    if warnings and (time.time() - last_alert_time > ALERT_COOLDOWN_SEC):
        embed = discord.Embed(
            title="⚠️ システム・セキュリティ警告",
            description="\n".join(warnings),
            color=0xF59E0B,
        )
        embed.set_footer(text="異常がある場合は下のボタンから手動で緊急停止できます。")
        await channel.send(embed=embed, view=SystemControlView())
        last_alert_time = time.time()


# ==============================================================================
# ロボットタスク確認ボタン UI
# ==============================================================================
class TaskConfirmView(View):
    def __init__(self, tasks_list, user_id):
        super().__init__(timeout=120.0)
        self.tasks = tasks_list
        self.user_id = user_id

    @discord.ui.button(label="🤖 実機で実行する", style=discord.ButtonStyle.green, emoji="▶️")
    async def confirm(self, interaction: discord.Interaction, button: Button):
        global is_busy
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ 指示を出した本人のみ操作できます。", ephemeral=True)
            return

        if is_busy:
            await interaction.response.send_message("⚠️ 現在別のアクションを実行中です。", ephemeral=True)
            return

        await interaction.response.defer()

        is_busy = True
        for child in self.children:
            child.disabled = True

        await interaction.edit_original_response(
            content="🦾 **TACHIKOMA が物理タスクを実行中... (緊急時は `!control` パネルから停止可能)**",
            view=self,
        )

        result_msg = await asyncio.to_thread(execute_tasks_sync, self.tasks)
        is_busy = False

        await interaction.followup.send(result_msg)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary, emoji="⏹️")
    async def cancel(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ 指示を出した本人のみ操作できます。", ephemeral=True)
            return

        await interaction.response.defer()

        for child in self.children:
            child.disabled = True
        await interaction.edit_original_response(content="🛑 **タスクをキャンセルしました。**", view=self)


@bot.event
async def on_ready():
    bot.add_view(SystemControlView())

    if not security_monitor_task.is_running():
        security_monitor_task.start()

    guild_info = f" (対象Guild: {TARGET_GUILD_ID})" if TARGET_GUILD_ID else " (全サーバー対象)"
    print(f"🤖 TACHIKOMA Discord Bot 起動完了: {bot.user.name} ({bot.user.id}){guild_info}")
    print(f"🛡️ 段階的防衛モニタリング稼働中 (通知先Ch: {ALERT_CHANNEL_ID})")


@bot.event
async def on_message(message: discord.Message):
    global is_busy
    if message.author.bot:
        return

    if TARGET_GUILD_ID:
        if message.guild is None or message.guild.id != TARGET_GUILD_ID:
            return

    user_text = message.content.strip()
    if not user_text:
        return

    if user_text in ("!control", "!status"):
        usage = shutil.disk_usage("/")
        disk_pct = (usage.used / usage.total) * 100
        load1, _, _ = os.getloadavg()
        svc_res = await asyncio.to_thread(
            subprocess.run,
            ["sudo", "systemctl", "is-active", PROTECTED_SERVICE],
            capture_output=True,
            text=True,
        )
        embed = discord.Embed(
            title="🛠️ システム状態監視 ＆ 緊急制御パネル",
            description=(
                f"• **CPU 1分負荷**: `{load1:.2f}`\n"
                f"• **ディスク使用率**: `{disk_pct:.1f}%`\n"
                f"• **保護対象サービス (`{PROTECTED_SERVICE}`)**: `{svc_res.stdout.strip()}`\n"
                f"• **アーム状態**: `{'動作中' if is_busy else '待機中'}`"
            ),
            color=0x38BDF8,
        )
        await message.reply(embed=embed, view=SystemControlView())
        return

    if is_busy:
        await message.reply("⚠️ 現在タスク実行中です。緊急停止したい場合は `!control` と入力して停止ボタンを押してください。")
        return

    async with message.channel.typing():
        hw_context = get_hardware_and_world_context()
        full_prompt = f"{hw_context}\n\n[ユーザーからの発話]:\n{user_text}"

        plan = await asyncio.to_thread(planner.plan, full_prompt)

        reply_text = plan.get("reply_text", "了解しました。")
        tasks_list = plan.get("tasks") or []

        if not tasks_list:
            await message.reply(f"💬 **TACHIKOMA:** {reply_text}")
            return

        embed = discord.Embed(
            title="📋 TACHIKOMA タスク実行プラン",
            description=f"💬 {reply_text}\n\n**立案されたサブタスク ({len(tasks_list)}件):**",
            color=0x38BDF8,
        )

        for i, t in enumerate(tasks_list, 1):
            p = t.get("pick", {})
            d = t.get("place", {})
            desc = t.get("description", f"物体移動 #{i}")

            p_r = p.get("r", 0.0) * 100.0
            p_th = p.get("theta_deg", 0.0)
            p_z = p.get("z", 0.015) * 1000.0
            p_roll = p.get("wrist_roll_deg", p.get("roll_deg", 0.0))

            d_r = d.get("r", 0.0) * 100.0
            d_th = d.get("theta_deg", 0.0)
            d_z = d.get("z", 0.015) * 1000.0
            d_roll = d.get("wrist_roll_deg", d.get("roll_deg", 0.0))

            field_val = (
                f"• **Pick** : $r$={p_r:4.1f}cm, $\\theta$={p_th:+5.1f}°, $z$={p_z:4.1f}mm, Roll(ID5)={p_roll:+5.1f}°\n"
                f"• **Place**: $r$={d_r:4.1f}cm, $\\theta$={d_th:+5.1f}°, $z$={d_z:4.1f}mm, Roll(ID5)={d_roll:+5.1f}°"
            )

            embed.add_field(
                name=f"Step {i}: {desc}",
                value=field_val,
                inline=False,
            )

        view = TaskConfirmView(tasks_list=tasks_list, user_id=message.author.id)
        await message.reply(embed=embed, view=view)


def main():
    global follower
    parser = argparse.ArgumentParser(description="TACHIKOMA Discord Bridge")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    args = parser.parse_args()

    if args.arm:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"⚠️ 実機接続失敗: {e}")
            follower = None

    if not DISCORD_TOKEN:
        print("❌ エラー: .env に DISCORD_BOT_TOKEN が設定されていません。")
        return

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()