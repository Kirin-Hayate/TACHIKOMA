"""
==============================================================================
TACHIKOMA Discord Bridge エージェント (src/tachikoma_discord_bot.py)
==============================================================================
【概要】
Discord サーバー経由で TACHIKOMA との自然言語対話および実機マニピュレーションを行う Bot。
HeroBox (Celeron J4125 / ヘッドレス) 上での常駐運用に最適化されています。

【主要機能】
1. サーバー (Guild) 単位のアクセス制御:
   - DISCORD_GUILD_ID で指定された専用サーバー内であれば、全チャンネルおよび
     新規作成されたスレッドから自由に会話可能。外部サーバーや DM からの誤爆を遮断。

2. ハードウェア・ワールドステートの動的コンテキスト注入:
   - 実機アーム（/dev/ttyACM0 等）の接続状態
   - USB Webカメラ（/dev/video0 等）の稼働状態
   - 机上物体認識結果 (config/current_world_state.json)
   をメッセージ受信ごとに動的取得し、LLM のプロンプトへ挿入してハルシネーションを防止。

3. 4変数ステップ表示:
   - Pick / Place の各目標を [r (cm), theta (deg), z (mm), Roll(ID5) (deg)] で Embed 表示。

4. 非同期イベント駆動 ＆ 実機安全ゲート:
   - discord.py のイベントループを阻害しないよう、プランニング・軌道生成・シリアル制御は
     asyncio.to_thread で非同期化。
   - インタラクティブボタン（実行 / キャンセル）による承認ゲートと排他制御 (is_busy)。
==============================================================================
"""

import os
import sys
import json
import asyncio
import argparse
from dotenv import load_dotenv
import discord
from discord.ext import commands
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

# 設定読み込み
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# 対象サーバーID (指定サーバー内であれば全チャンネル・スレッドで応答可能)
TARGET_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
if TARGET_GUILD_ID:
    TARGET_GUILD_ID = int(TARGET_GUILD_ID)

# ロボット状態管理
is_busy = False
planner = LLMTaskPlanner()
generator = ParametricMotionGenerator()
follower = None

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


def get_hardware_and_world_context():
    """現在の実機接続状態、カメラ状態、机上物体情報を取得してプロンプト用テキストを生成"""
    global follower

    # 1. 実機アーム状態
    arm_status = "接続中 (動作可能)" if (follower is not None and getattr(follower, 'is_open', False)) else "未接続 (実機動作不可・シミュレーションのみ)"

    # 2. カメラ状態判定
    camera_status = "未接続 / 不明"
    if os.path.exists("/dev/video0"):
        camera_status = "接続中 (/dev/video0 検出済み)"

    # 3. 机の上の状況 (current_world_state.json)
    world_state_path = os.path.join(BASE_DIR, "config", "current_world_state.json")
    objects_summary = "現在、机の上に検知された物体データはありません（空の状態または未計測）。"

    if os.path.exists(world_state_path):
        try:
            with open(world_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                objs = data.get("objects", [])
                if objs:
                    lines = [f"- [ID #{o.get('id')}] 名称: {o.get('name', '物体')}, 色: {o.get('color', '不明')}, 座標: (X={o.get('x_mm', 0):.1f}mm, Y={o.get('y_mm', 0):.1f}mm, Z={o.get('z_mm', 0):.1f}mm), 角度: {o.get('angle_deg', 0):.1f}°" for o in objs]
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


def execute_tasks_sync(tasks):
    """実機サーボへの物理搬送シーケンス（ブロッキング同期関数）"""
    global follower
    if follower is None or not getattr(follower, 'is_open', False):
        return "⚠️ 実機アームが接続されていません（プレビューのみモード）"

    try:
        # トルクON
        for sid in SERVO_IDS:
            follower.set_torque(sid, True)

        home_rad = get_home_radians()
        total = len(tasks)
        for idx, task in enumerate(tasks, 1):
            p = task.get('pick')
            d = task.get('place')
            # 軌道生成
            success, frames, log_text = generator.generate_from_coords(p, d)
            if not success:
                continue

            # フレーム再生
            for _, rad_positions in frames:
                for sid in SERVO_IDS:
                    raw_val = radian_to_raw(sid, rad_positions[sid])
                    follower.write_position(sid, raw_val)
                import time
                time.sleep(0.02)

        return "✅ すべての搬送タスクが完了しました。"
    except Exception as e:
        return f"❌ 実行中エラー: {e}"
    finally:
        for sid in SERVO_IDS:
            follower.set_torque(sid, False)


class TaskConfirmView(View):
    """Discord UI ボタン（実行 / キャンセル）"""
    def __init__(self, tasks, user_id):
        super().__init__(timeout=60.0)
        self.tasks = tasks
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

        is_busy = True
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(content="🦾 **TACHIKOMA が物理タスクを実行中...**", view=self)

        result_msg = await asyncio.to_thread(execute_tasks_sync, self.tasks)
        is_busy = False

        await interaction.followup.send(result_msg)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary, emoji="⏹️")
    async def cancel(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ 指示を出した本人のみ操作できます。", ephemeral=True)
            return

        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(content="🛑 **タスクをキャンセルしました。**", view=self)


@bot.event
async def on_ready():
    guild_info = f" (対象Guild: {TARGET_GUILD_ID})" if TARGET_GUILD_ID else " (全サーバー対象)"
    print(f"🤖 TACHIKOMA Discord Bot 起動完了: {bot.user.name} ({bot.user.id}){guild_info}")


@bot.event
async def on_message(message: discord.Message):
    global is_busy
    if message.author.bot:
        return

    # サーバーID制限: 指定サーバー以外のメッセージやダイレクトメッセージ(DM)を無視
    if TARGET_GUILD_ID:
        if message.guild is None or message.guild.id != TARGET_GUILD_ID:
            return

    user_text = message.content.strip()
    if not user_text:
        return

    if is_busy:
        await message.reply("⚠️ 現在タスク実行中です。完了までお待ちください。")
        return

    async with message.channel.typing():
        # ハードウェア状況・机上物体情報を注入したプロンプトを構築
        hw_context = get_hardware_and_world_context()
        full_prompt = f"{hw_context}\n\n[ユーザーからの発話]:\n{user_text}"

        # LLM タスク立案を実行
        plan = await asyncio.to_thread(planner.plan, full_prompt)

        reply_text = plan.get("reply_text", "了解しました。")
        tasks = plan.get("tasks") or []

        # 物理タスクがない場合（雑談・状態照会など）
        if not tasks:
            await message.reply(f"💬 **TACHIKOMA:** {reply_text}")
            return

        # 搬送タスクがある場合: 4変数形式で Embed を構成
        embed = discord.Embed(
            title="📋 TACHIKOMA タスク実行プラン",
            description=f"💬 {reply_text}\n\n**立案されたサブタスク ({len(tasks)}件):**",
            color=0x38bdf8
        )

        for i, t in enumerate(tasks, 1):
            p = t.get('pick', {})
            d = t.get('place', {})
            desc = t.get('description', f"物体移動 #{i}")

            # 4変数の抽出 (r [cm], theta [deg], z [mm], Roll(ID5) [deg])
            p_r = p.get('r', 0.0) * 100.0
            p_th = p.get('theta_deg', 0.0)
            p_z = p.get('z', 0.015) * 1000.0
            p_roll = p.get('wrist_roll_deg', p.get('roll_deg', 0.0))

            d_r = d.get('r', 0.0) * 100.0
            d_th = d.get('theta_deg', 0.0)
            d_z = d.get('z', 0.015) * 1000.0
            d_roll = d.get('wrist_roll_deg', d.get('roll_deg', 0.0))

            field_val = (
                f"• **Pick** : $r$={p_r:4.1f}cm, $\\theta$={p_th:+5.1f}°, $z$={p_z:4.1f}mm, Roll(ID5)={p_roll:+5.1f}°\n"
                f"• **Place**: $r$={d_r:4.1f}cm, $\\theta$={d_th:+5.1f}°, $z$={d_z:4.1f}mm, Roll(ID5)={d_roll:+5.1f}°"
            )

            embed.add_field(
                name=f"Step {i}: {desc}",
                value=field_val,
                inline=False
            )

        view = TaskConfirmView(tasks=tasks, user_id=message.author.id)
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