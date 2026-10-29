"""
==============================================================================
TACHIKOMA Discord Bridge (interfaces/discord_bot.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムの Layer 1 (インターフェース層) を担当する
Discord Bot クライアントです。

【所属レイヤー】
Layer 1: インターフェース層 (入出力・UI・通知)

【設計方針と責務分離】
1. UI・通知特化 (薄いインターフェース):
   - システム監視・自律防衛ループ、OS遮断操作、Gemini AI診断の実体はすべて
     Layer 3 の専門サービス (services/sentinel/) に完全委譲。
   - モデル指定も config/model_config.py から一括参照し、ハードコードを排除。
2. 段階的防衛通知 (Tiered Defense):
   - discord.ext.tasks による 2分間隔の巡回監視ループ。
   - レベル1 (警告通知: クールタイムあり) / レベル2 (自動防衛: クールタイム無視・即時報告)。
3. スマホ最適化 UI (SystemControlView):
   - 3段グリッド配置による片手操作対応。
   - 永続化 View (timeout=None) により Bot 再起動後も既存メッセージのボタンが有効。
4. ノートPC / ローカル開発環境保護:
   - 外部ハードウェアや Linux デーモンが存在しない開発機でもエラー落ちせず、
     安全に起動・UI表示テストが完結。
==============================================================================
"""

import os
import io
import sys
import time
import glob
import json
import shutil
import subprocess
import asyncio
import argparse
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any

import psutil
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
from discord.ui import View, Button

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

# --- 内部モジュール (新アーキテクチャ連携) ---
from config.model_config import get_candidate_models, get_primary_model
from services.sentinel.system_actions import (
    emergency_stop_arm,
    stop_service,
    start_service,
    get_service_status,
    isolate_network,
    DEFAULT_PROTECTED_SERVICE,
)
from services.sentinel.system_monitor import (
    check_system_security_sync,
    run_ai_security_diagnosis_sync,
    ALERT_COOLDOWN_SEC,
)

# --- ハードウェア・アーム既存モジュール ---
try:
    from config.joint_config import FOLLOWER_PORT, BAUDRATE, SERVO_IDS
    from core.llm_planner import LLMTaskPlanner
    from drivers.sts3215 import STS3215Driver
    from services.arm_worker.motion_generator import ParametricMotionGenerator
    from services.arm_worker.kinematics import raw_to_radian, radian_to_raw, get_home_radians
except ImportError:
    FOLLOWER_PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000
    SERVO_IDS = [1, 2, 3, 4, 5, 6]
    LLMTaskPlanner = None
    STS3215Driver = None
    ParametricMotionGenerator = None


# ==============================================================================
# ロギングユーティリティ
# ==============================================================================
def dlog(tag: str, msg: str, level: str = "INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    prefix_map = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERR": "❌", "DEBUG": "🔍"}
    icon = prefix_map.get(level.upper(), "•")
    print(f"[{now}] {icon} [{tag}] {msg}", flush=True)


def derr(tag: str, context_msg: str, exc: Exception = None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now}] ❌ [ERROR in {tag}] {context_msg}", flush=True)
    if exc is not None:
        print(f"Detail: {exc}", flush=True)


# ==============================================================================
# 環境変数・状態管理
# ==============================================================================
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
TARGET_GUILD_ID = int(os.getenv("DISCORD_GUILD_ID")) if os.getenv("DISCORD_GUILD_ID") else None
ALERT_CHANNEL_ID = int(os.getenv("DISCORD_ALERT_CHANNEL_ID")) if os.getenv("DISCORD_ALERT_CHANNEL_ID") else None

CAMERA_DEVICE_INDEX = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
CAMERA_DEVICE_PATH = os.getenv("CAMERA_DEVICE_PATH", f"/dev/video{CAMERA_DEVICE_INDEX}")
STREAM_EXTERNAL_URL = os.getenv("STREAM_EXTERNAL_URL", "").strip()

# 実行状態フラグ
is_busy = False
abort_arm_flag = False
last_alert_time = 0.0

# コアインスタンスの初期化
planner = LLMTaskPlanner() if LLMTaskPlanner else None
generator = ParametricMotionGenerator() if ParametricMotionGenerator else None
follower = None

# Discord クライアントの初期化
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ==============================================================================
# 1. 接続ハードウェア走査・物理コンテキスト生成 (State Manager 移行準備)
# ==============================================================================
def get_connected_hardware_list() -> List[str]:
    hw_items = []
    # (1) アーム接続確認
    if follower is not None and getattr(follower, "is_open", False):
        hw_items.append(f"🦾 **アーム**: 接続中・通信可能 (`{FOLLOWER_PORT}`)")
    elif os.path.exists(FOLLOWER_PORT):
        hw_items.append(f"🦾 **アーム**: ポート認識済・未オープン (`{FOLLOWER_PORT}`)")
    else:
        hw_items.append(f"🦾 **アーム**: 未接続 (`{FOLLOWER_PORT}` 非存在)")

    # (2) カメラ確認
    video_devs = sorted(glob.glob("/dev/video*"))
    if video_devs:
        hw_items.append(f"📷 **カメラ**: 認識済 (`{', '.join(video_devs)}`)")
    else:
        hw_items.append("📷 **カメラ**: 未接続 (`/dev/video*` なし)")

    # (3) 外部ストレージ確認
    try:
        res = subprocess.run(["lsblk", "-J", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,RM"], capture_output=True, text=True, timeout=3)
        data = json.loads(res.stdout)
        exts = []
        def scan_blocks(blocks):
            for b in blocks:
                mnt = b.get("mountpoint")
                if mnt and mnt not in ("/", "/boot", "/boot/efi", "[SWAP]") and not str(mnt).startswith("/snap"):
                    exts.append(f"`/dev/{b.get('name')}` ({b.get('size')} -> `{mnt}`)")
                if "children" in b:
                    scan_blocks(b["children"])
        scan_blocks(data.get("blockdevices", []))
        hw_items.append(f"💾 **外部ストレージ**: {', '.join(exts)}" if exts else "💾 **外部ストレージ**: なし (内蔵SSDのみ)")
    except Exception:
        hw_items.append("💾 **外部ストレージ**: 取得スキップ (ノートPC/非Linux)")

    return hw_items


def get_hardware_and_world_context() -> str:
    hw_list = get_connected_hardware_list()
    hw_summary = "\n".join(f"- {item}" for item in hw_list)

    world_state_path = os.path.join(BASE_DIR, "config", "current_world_state.json")
    objects_summary = "現在、机の上に検知された物体データはありません（空の状態または未計測）。"

    if os.path.exists(world_state_path):
        try:
            with open(world_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                objs = data.get("objects", [])
                if objs:
                    lines = [
                        f"- [ID #{o.get('id')}] 名称: {o.get('name', '物体')}, 色: {o.get('color', '不明')}, "
                        f"座標: (X={o.get('x_mm', 0):.1f}mm, Y={o.get('y_mm', 0):.1f}mm, Z={o.get('z_mm', 0):.1f}mm)"
                        for o in objs
                    ]
                    objects_summary = "【机の上にある検知済み物体リスト】:\n" + "\n".join(lines)
        except Exception as e:
            derr("WORLD_CTX", f"{world_state_path} 読み込み失敗", e)

    return f"""
[現在のタチコマ物理環境ステータス]
{hw_summary}
- 机上認識状況:
{objects_summary}
※机の上の状況や接続状態を尋ねられた場合は、上記の実測情報を基に正確に回答してください。
"""


# ==============================================================================
# 2. 実機サーボ搬送シーケンス (Motion Executor 移行準備)
# ==============================================================================
def execute_tasks_sync(tasks_list: List[Dict[str, Any]]) -> str:
    global follower, abort_arm_flag
    abort_arm_flag = False

    if follower is None or not getattr(follower, "is_open", False):
        dlog("ARM_EXEC", "実機アーム未接続のためシミュレーション動作", "WARN")
        return "⚠️ 実機アームが接続されていません（プレビューのみモード）"

    try:
        dlog("ARM_EXEC", f"搬送シーケンス開始 (全 {len(tasks_list)} ステップ)", "INFO")
        for sid in SERVO_IDS:
            follower.set_torque(sid, True)

        for idx, task in enumerate(tasks_list, 1):
            if abort_arm_flag:
                emergency_stop_arm(follower)
                return "🛑 緊急停止ボタンによりタスクを即座に中断し、脱力しました。"

            p = task.get("pick")
            d = task.get("place")
            if not generator:
                continue
            success, frames, _ = generator.generate_from_coords(p, d)
            if not success:
                continue

            for _, rad_positions in frames:
                if abort_arm_flag:
                    emergency_stop_arm(follower)
                    return "🛑 緊急停止ボタンにより動作を即時中断し、脱力しました。"
                for sid in SERVO_IDS:
                    raw_val = radian_to_raw(sid, rad_positions[sid])
                    follower.write_position(sid, raw_val)
                time.sleep(0.02)

        return "✅ すべての搬送タスクが正常に完了しました。"
    except Exception as e:
        derr("ARM_EXEC", "実機搬送中に例外発生", e)
        return f"❌ 実行中エラー: {e}"
    finally:
        emergency_stop_arm(follower)


# ==============================================================================
# 3. スマホ最適化 緊急制御パネル UI (SystemControlView)
# ==============================================================================
class SystemControlView(View):
    def __init__(self):
        super().__init__(timeout=None)
        if STREAM_EXTERNAL_URL:
            self.add_item(Button(label="📹 ライブ", style=discord.ButtonStyle.link, url=STREAM_EXTERNAL_URL, row=0))

    @discord.ui.button(label="📸 撮影", style=discord.ButtonStyle.secondary, custom_id="sys_cam_snap", row=0)
    async def snap_camera(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[📸 撮影] 押下 (User: {interaction.user})", "INFO")
        await interaction.response.send_message("📸 **カメラ撮影を要求しました（Phase 2で本格統合予定）**")

    @discord.ui.button(label="🧠 AI診断", style=discord.ButtonStyle.primary, custom_id="sys_ai_diag", row=0)
    async def ai_diagnosis(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[🧠 AI診断] 押下 (User: {interaction.user})", "INFO")
        await interaction.response.send_message(
            "🧠 **AIセキュリティ診断を開始しました。**\n"
            "7系統のシステムメトリクスを収集し、Gemini による解析を実行しています..."
        )

        async with interaction.channel.typing():
            # Sentinel モジュールへ処理を完全委譲
            success, diag_text, used_model, elapsed = await asyncio.to_thread(run_ai_security_diagnosis_sync)

        embed = discord.Embed(
            title="🧠 Gemini システム・セキュリティ診断レポート",
            description=diag_text[:4000],
            color=0x8B5CF6 if success else 0xEF4444,
        )
        embed.set_footer(text=f"使用モデル: {used_model} | 所要時間: {elapsed:.2f}秒 | Sentinel Layer 3")
        await interaction.followup.send(content="✅ **AIセキュリティ診断が完了しました！**", embed=embed)

    @discord.ui.button(label="🛑 アーム", style=discord.ButtonStyle.danger, custom_id="sys_stop_arm", row=1)
    async def stop_arm(self, interaction: discord.Interaction, button: Button):
        global is_busy, abort_arm_flag, follower
        dlog("UI_BTN", "[🛑 アーム] 緊急停止押下", "WARN")
        await interaction.response.defer()
        abort_arm_flag = True
        is_busy = False

        # Sentinel のアーム緊急停止を呼び出し
        success, msg = emergency_stop_arm(follower)
        await interaction.followup.send(msg)

    @discord.ui.button(label="🛑 Web", style=discord.ButtonStyle.danger, custom_id="sys_stop_service", row=1)
    async def stop_web_service(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[🛑 Web] サービス停止押下 ({DEFAULT_PROTECTED_SERVICE})", "WARN")
        await interaction.response.defer()
        success, msg = await asyncio.to_thread(stop_service, DEFAULT_PROTECTED_SERVICE)
        await interaction.followup.send(msg)

    @discord.ui.button(label="▶️ Web", style=discord.ButtonStyle.success, custom_id="sys_start_service", row=1)
    async def start_web_service(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[▶️ Web] サービス再開押下 ({DEFAULT_PROTECTED_SERVICE})", "INFO")
        await interaction.response.defer()
        success, msg = await asyncio.to_thread(start_service, DEFAULT_PROTECTED_SERVICE)
        await interaction.followup.send(msg)

    @discord.ui.button(label="🔌 外部隔離", style=discord.ButtonStyle.secondary, custom_id="sys_isolate_net", row=2)
    async def isolate_net(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", "[🔌 外部隔離] 押下", "WARN")
        await interaction.response.defer()
        await interaction.followup.send(
            "🚨 **3秒後に Tailscale VPN を切断し、外部ネットワークから完全隔離します。**\n"
            "※復旧にはサーバー本体へ物理コンソールを接続する必要があります。"
        )
        await asyncio.sleep(3)
        success, msg = await asyncio.to_thread(isolate_network)
        dlog("UI_BTN", f"ネットワーク隔離完了: {msg}", "OK")


# ==============================================================================
# 4. 段階的セキュリティ監視タスク (Sentinel 連携)
# ==============================================================================
@tasks.loop(minutes=2)
async def security_monitor_task():
    global last_alert_time
    if not ALERT_CHANNEL_ID:
        return

    channel = bot.get_channel(ALERT_CHANNEL_ID)
    if channel is None:
        return

    # Sentinel モジュールの検査関数を呼び出し
    warnings, auto_actions = await asyncio.to_thread(check_system_security_sync)

    if auto_actions:
        embed = discord.Embed(
            title="🚨 【自動防衛発動】不審な挙動を検知・自動停止しました",
            description="\n\n".join(auto_actions),
            color=0xFF0000,
        )
        embed.set_footer(text="下のパネルからサービス復旧または外部隔離を選択してください。")
        await channel.send(embed=embed, view=SystemControlView())
        last_alert_time = time.time()
        return

    if warnings and (time.time() - last_alert_time > ALERT_COOLDOWN_SEC):
        embed = discord.Embed(
            title="⚠️ システム・セキュリティ警告",
            description="\n".join(warnings),
            color=0xF59E0B,
        )
        embed.set_footer(text="異常がある場合は下のパネルから手動で緊急停止やAI診断を実行できます。")
        await channel.send(embed=embed, view=SystemControlView())
        last_alert_time = time.time()


# ==============================================================================
# 5. ステータスパネル Embed 生成
# ==============================================================================
def build_status_embed_sync() -> discord.Embed:
    usage = shutil.disk_usage("/")
    disk_pct = (usage.used / usage.total) * 100
    load_val = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    mem = psutil.virtual_memory()

    is_active, svc_state = get_service_status(DEFAULT_PROTECTED_SERVICE)
    hw_items = get_connected_hardware_list()
    hw_block = "\n".join(f"• {item}" for item in hw_items)

    candidates = get_candidate_models()

    desc = (
        f"**【システムメトリクス】**\n"
        f"• **CPU 1分負荷**: `{load_val:.2f}`\n"
        f"• **メモリ使用率**: `{mem.percent:.1f}%` (`{mem.used // (1024**2)}MB` / `{mem.total // (1024**2)}MB`)\n"
        f"• **ディスク使用率**: `{disk_pct:.1f}%`\n"
        f"• **保護対象サービス (`{DEFAULT_PROTECTED_SERVICE}`)**: `{svc_state}`\n"
        f"• **アーム状態**: `{'動作中' if is_busy else '待機中'}`\n\n"
        f"**【接続ハードウェア一覧】**\n"
        f"{hw_block}"
    )

    embed = discord.Embed(title="🛠️ システム状態監視 ＆ 緊急制御パネル", description=desc, color=0x38BDF8)
    embed.set_footer(text=f"AIモデル優先順: {' -> '.join(candidates[:2])}")
    return embed


# ==============================================================================
# 6. タスク確認 UI (TaskConfirmView)
# ==============================================================================
class TaskConfirmView(View):
    def __init__(self, tasks_list: List[Dict[str, Any]], user_id: int):
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
            content="🦾 **TACHIKOMA が物理タスクを実行中... (緊急停止は `!control` から可能)**",
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


# ==============================================================================
# 7. イベントハンドラ
# ==============================================================================
@bot.event
async def on_ready():
    bot.add_view(SystemControlView())
    if not security_monitor_task.is_running():
        security_monitor_task.start()

    guild_info = f" (対象Guild: {TARGET_GUILD_ID})" if TARGET_GUILD_ID else " (全サーバー対象)"
    dlog("BOT_READY", f"TACHIKOMA Discord Bot 起動完了: {bot.user.name} ({bot.user.id}){guild_info}", "OK")
    dlog("BOT_READY", f"Sentinel監視稼働中 (通知先Ch: {ALERT_CHANNEL_ID}, 保護SVC: {DEFAULT_PROTECTED_SERVICE})", "INFO")


@bot.event
async def on_message(message: discord.Message):
    global is_busy
    if message.author.bot:
        return

    if TARGET_GUILD_ID and (message.guild is None or message.guild.id != TARGET_GUILD_ID):
        return

    user_text = message.content.strip()
    if not user_text:
        return

    dlog("DISCORD_MSG", f"受信 [{message.channel} / {message.author}]: {user_text}", "INFO")

    try:
        # コントロールパネルの表示要求
        if user_text in ("!control", "!status"):
            embed = await asyncio.to_thread(build_status_embed_sync)
            await message.reply(embed=embed, view=SystemControlView())
            return

        if is_busy:
            await message.reply("⚠️ 現在タスク実行中です。緊急停止したい場合は `!control` と入力して停止ボタンを押してください。")
            return

        # 自然言語プランニング (LLMTaskPlanner)
        if planner is None:
            await message.reply("⚠️ プランナーがロードされていません。")
            return

        async with message.channel.typing():
            hw_context = await asyncio.to_thread(get_hardware_and_world_context)
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
                    f"• **Pick** : $r$={p_r:4.1f}cm, $\\theta$={p_th:+5.1f}°, $z$={p_z:4.1f}mm, Roll={p_roll:+5.1f}°\n"
                    f"• **Place**: $r$={d_r:4.1f}cm, $\\theta$={d_th:+5.1f}°, $z$={d_z:4.1f}mm, Roll={d_roll:+5.1f}°"
                )
                embed.add_field(name=f"Step {i}: {desc}", value=field_val, inline=False)

            view = TaskConfirmView(tasks_list=tasks_list, user_id=message.author.id)
            await message.reply(embed=embed, view=view)

    except Exception as e:
        derr("ON_MESSAGE", f"メッセージ処理中に例外発生: {user_text}", e)
        await message.reply(f"❌ **内部エラーが発生しました (`{type(e).__name__}`)**: `{e}`")


# ==============================================================================
# メインエントリーポイント ＆ ノートPC単体テスト
# ==============================================================================
def main():
    global follower
    parser = argparse.ArgumentParser(description="TACHIKOMA Discord Bridge (Layer 1)")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    parser.add_argument("--test-ui", action="store_true", help="ノートPC用: Discordに接続せずUI Embed生成を単体テスト")
    args = parser.parse_args()

    if args.test_ui:
        print("=" * 70)
        print("🧪 [TEST] interfaces/discord_bot.py UI・コンテキスト単体生成テスト")
        print("=" * 70)
        ctx = get_hardware_and_world_context()
        print("1. 生成されたハードウェアコンテキスト:")
        print(ctx[:300] + "...\n")
        embed = build_status_embed_sync()
        print(f"2. 生成された Embed タイトル: {embed.title}")
        print(f"   Embed 説明文字数: {len(embed.description)} chars")
        print(f"   Embed フッター: {embed.footer.text}")
        print("=" * 70)
        print("🎉 UI生成テストが正常に完了しました！")
        return

    if args.arm and STS3215Driver:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            dlog("STARTUP", f"実機フォロワー接続完了 ({FOLLOWER_PORT})", "OK")
        except Exception as e:
            derr("STARTUP", f"実機フォロワー接続失敗: {e}")
            follower = None

    if not DISCORD_TOKEN:
        dlog("STARTUP", ".env に DISCORD_BOT_TOKEN が未設定のため Bot 接続はスキップします (テスト完了)。", "WARN")
        return

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()