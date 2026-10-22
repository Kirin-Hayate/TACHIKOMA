import os
import sys
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
TARGET_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
if TARGET_CHANNEL_ID:
    TARGET_CHANNEL_ID = int(TARGET_CHANNEL_ID)

# ロボット状態管理
is_busy = False
planner = LLMTaskPlanner()
generator = ParametricMotionGenerator()
follower = None

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


def execute_tasks_sync(tasks):
    """実機サーボへの物理搬送シーケンス（ブロッキング同期関数）"""
    global follower
    if follower is None:
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
                # 制御レート維持
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
        super().__init__(timeout=60.0)  # 60秒でタイムアウト
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
        # ボタンを無効化してメッセージ更新
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(content="🦾 **TACHIKOMA が物理タスクを実行中...**", view=self)

        # 別スレッドでブロッキングな実機搬送を非同期実行
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
    print(f"🤖 TACHIKOMA Discord Bot 起動完了: {bot.user.name} ({bot.user.id})")


@bot.event
async def on_message(message: discord.Message):
    global is_busy
    # Bot自身の発言は無視
    if message.author.bot:
        return

    # 許可チャンネル限定チェック
    if TARGET_CHANNEL_ID and message.channel.id != TARGET_CHANNEL_ID:
        return

    prompt = message.content.strip()
    if not prompt:
        return

    if is_busy:
        await message.reply("⚠️ 現在タスク実行中です。完了までお待ちください。")
        return

    async with message.channel.typing():
        # LLM タスク立案をスレッドで非同期化
        plan = await asyncio.to_thread(planner.plan, prompt)

        reply_text = plan.get("reply_text", "了解しました。")
        tasks = plan.get("tasks") or []

        # 物理タスクがない場合（雑談・照会など）
        if not tasks:
            await message.reply(f"💬 **TACHIKOMA:** {reply_text}")
            return

        # 搬送タスクがある場合: Embed と承認ボタンを作成
        embed = discord.Embed(
            title="📋 TACHIKOMA タスク実行プラン",
            description=f"💬 {reply_text}\n\n**立案されたサブタスク ({len(tasks)}件):**",
            color=0x38bdf8
        )
        for i, t in enumerate(tasks, 1):
            p = t.get('pick', {})
            d = t.get('place', {})
            desc = t.get('description', f"物体移動 #{i}")
            embed.add_field(
                name=f"Step {i}: {desc}",
                value=f"• Pick: r={p.get('r',0)*100:.1f}cm, θ={p.get('theta_deg',0):+.1f}°\n• Place: r={d.get('r',0)*100:.1f}cm, θ={d.get('theta_deg',0):+.1f}°",
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