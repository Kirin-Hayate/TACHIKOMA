"""
==============================================================================
TACHIKOMA Discord Bridge (interfaces/discord_bot.py) [Orchestrator統合版]
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムの Layer 1 (インターフェース層) を担当する
Discord Bot クライアントです。

【所属レイヤー】
Layer 1: インターフェース層 (入出力・UI・通知)

【設計方針と責務分離】
1. UI・通知特化 (薄いインターフェース):
   - 会話・タスク分解・IK事前検証・搬送制御・再スキャンは Layer 2 (core/orchestrator.py) に委譲。
   - サーバー防衛・メトリクス監視・AI診断は Layer 3 (services/sentinel/) に委譲。
2. 承認ワークフロー (TaskConfirmView):
   - 立案された Pick & Place サブタスク一覧、IK到達性、自動生成されたモーションCSVを表示。
   - ユーザーのボタン押下により安全に実行開始。
3. 段階的防衛通知 (Tiered Defense):
   - discord.ext.tasks による 2分間隔の巡回監視ループ。
4. ノートPC / ローカル開発環境保護:
   - 実機未接続時はドライランとして安全に稼働し、--test-ui でUI検証が完結。
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

# --- 司令塔・Worker モジュール ---
from core.orchestrator import TachikomaOrchestrator
from services.arm_worker.motion_executor import MotionExecutor

try:
    from config.joint_config import FOLLOWER_PORT, BAUDRATE, SERVO_IDS
    from drivers.sts3215 import STS3215Driver
except ImportError:
    FOLLOWER_PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000
    SERVO_IDS = [1, 2, 3, 4, 5, 6]
    STS3215Driver = None


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
STREAM_EXTERNAL_URL = os.getenv("STREAM_EXTERNAL_URL", "").strip()

# 実行状態フラグ
is_busy = False
last_alert_time = 0.0

# ハードウェアドライバ・司令塔オーケストレーター初期化
follower: Optional[Any] = None
motion_executor: Optional[MotionExecutor] = None
orchestrator: Optional[TachikomaOrchestrator] = None

# Discord クライアントの初期化
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ==============================================================================
# 1. 接続ハードウェア走査・物理コンテキスト生成
# ==============================================================================
def get_connected_hardware_list() -> List[str]:
    hw_items = []
    # (1) アーム接続確認
    if follower is not None and getattr(follower, "is_open", False):
        hw_items.append(f"🦾 **アーム**: 接続中・通信可能 (`{FOLLOWER_PORT}`)")
    elif os.path.exists(FOLLOWER_PORT):
        hw_items.append(f"🦾 **アーム**: ポート認識済・未オープン (`{FOLLOWER_PORT}`)")
    else:
        hw_items.append(f"🦾 **アーム**: 未接続 (`{FOLLOWER_PORT}` 非存在 / ドライラン)")

    # (2) カメラ確認
    video_devs = sorted(glob.glob("/dev/video*"))
    if video_devs:
        hw_items.append(f"📷 **カメラ**: 認識済 (`{', '.join(video_devs)}`)")
    else:
        hw_items.append("📷 **カメラ**: 未接続 (プレースホルダーモード)")

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
    """現在のハードウェア接続状態と最新ワールドステートの要約文を生成する"""
    hw_list = get_connected_hardware_list()
    hw_summary = "\n".join(f"- {item}" for item in hw_list)

    world_state = orchestrator.get_current_world_state() if orchestrator else {}
    objs = world_state.get("objects", [])
    updated_at = world_state.get("updated_at", "未計測")
    total_objs = world_state.get("total_objects", len(objs))

    if objs:
        lines = [
            f"- [ID #{o.get('id')}] 名称: {o.get('name', '物体')}, 色: {o.get('color', '不明')}, "
            f"座標: (X={o.get('x_mm', 0):.1f}mm, Y={o.get('y_mm', 0):.1f}mm, Z={o.get('z_mm', 0):.1f}mm)"
            for o in objs
        ]
        objects_summary = f"【机の上にある検知済み物体リスト (最終更新: {updated_at}, 計{total_objs}個)】:\n" + "\n".join(lines)
    else:
        objects_summary = f"現在、机の上に検知された物体データはありません (最終確認: {updated_at})。"

    return f"""
[現在のタチコマ物理環境ステータス]
{hw_summary}
- 机上認識状況:
{objects_summary}
※机の上の状況や確認日時、接続状態を尋ねられた場合は、上記の実測情報を基に回答してください。
"""

# ==============================================================================
# 2. スマホ最適化 緊急制御パネル UI (SystemControlView)
# ==============================================================================
class SystemControlView(View):
    def __init__(self):
        super().__init__(timeout=None)
        if STREAM_EXTERNAL_URL:
            self.add_item(Button(label="📹 ライブ", style=discord.ButtonStyle.link, url=STREAM_EXTERNAL_URL, row=0))

    @discord.ui.button(label="📸 撮影/再スキャン", style=discord.ButtonStyle.secondary, custom_id="sys_cam_snap", row=0)
    async def snap_camera(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[📸 撮影/再スキャン] 押下 (User: {interaction.user})", "INFO")
        await interaction.response.defer()
        if orchestrator is None:
            await interaction.followup.send("⚠️ オーケストレーターが初期化されていません。")
            return

        # 1. カメラ画像 (生フレーム or マーカー注釈付き) を取得
        jpeg_bytes, snap_err = await asyncio.to_thread(orchestrator.vision.capture_annotated_snapshot)
        
        # 2. 机上認識・ワールドステート更新 (ArUcoマーカーがあれば更新)
        success, new_state, msg = await asyncio.to_thread(orchestrator.refresh_vision, None, False)
        obj_count = new_state.get("total_objects", 0)
        updated_at = new_state.get("updated_at", "不明")

        content = f"📸 **机上スキャン実行結果**\n• {msg}\n• 認識物体数: `{obj_count}` 個 (最終更新: `{updated_at}`)"

        # 画像が取得できていれば添付して送信
        if jpeg_bytes:
            file = discord.File(io.BytesIO(jpeg_bytes), filename="tabletop_snapshot.jpg")
            await interaction.followup.send(content=content, file=file)
        else:
            await interaction.followup.send(content=f"{content}\n⚠️ 画像の取得に失敗しました: {snap_err}")

    @discord.ui.button(label="🧠 AI診断", style=discord.ButtonStyle.primary, custom_id="sys_ai_diag", row=0)
    async def ai_diagnosis(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[🧠 AI診断] 押下 (User: {interaction.user})", "INFO")
        await interaction.response.send_message(
            "🧠 **AIセキュリティ診断を開始しました。**\n"
            "7系統のシステムメトリクスを収集し、Gemini による解析を実行しています..."
        )

        async with interaction.channel.typing():
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
        global is_busy
        dlog("UI_BTN", "[🛑 アーム] 緊急停止押下", "WARN")
        await interaction.response.defer()
        is_busy = False

        if orchestrator is not None:
            orchestrator.trigger_abort()
        success, msg = emergency_stop_arm(follower)
        await interaction.followup.send(f"🛑 **緊急停止実行**: {msg}")

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
# 3. 段階的セキュリティ監視タスク (Sentinel 連携)
# ==============================================================================
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
# 4. ステータスパネル Embed 生成
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
    world_state = orchestrator.get_current_world_state() if orchestrator else {}
    obj_count = world_state.get("total_objects", 0)

    desc = (
        f"**【システムメトリクス】**\n"
        f"• **CPU 1分負荷**: `{load_val:.2f}`\n"
        f"• **メモリ使用率**: `{mem.percent:.1f}%` (`{mem.used // (1024**2)}MB` / `{mem.total // (1024**2)}MB`)\n"
        f"• **ディスク使用率**: `{disk_pct:.1f}%`\n"
        f"• **保護対象サービス (`{DEFAULT_PROTECTED_SERVICE}`)**: `{svc_state}`\n"
        f"• **アーム状態**: `{'動作中' if is_busy else '待機中'}`\n"
        f"• **机上認識物体**: `{obj_count}` 個検知\n\n"
        f"**【接続ハードウェア一覧】**\n"
        f"{hw_block}"
    )

    embed = discord.Embed(title="🛠️ システム状態監視 ＆ 緊急制御パネル", description=desc, color=0x38BDF8)
    embed.set_footer(text=f"AIモデル優先順: {' -> '.join(candidates[:2])}")
    return embed


# ==============================================================================
# 5. タスク確認 UI (TaskConfirmView - Orchestrator連携)
# ==============================================================================
class TaskConfirmView(View):
    def __init__(self, user_id: int, preview_csvs: Optional[List[str]] = None):
        super().__init__(timeout=120.0)
        self.user_id = user_id
        self.preview_csvs = preview_csvs or []

    @discord.ui.button(label="🤖 実機で実行する", style=discord.ButtonStyle.green, emoji="▶️")
    async def confirm(self, interaction: discord.Interaction, button: Button):
        global is_busy
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ 指示を出した本人のみ操作できます。", ephemeral=True)
            return

        if is_busy:
            await interaction.response.send_message("⚠️ 現在別のアクションを実行中です。", ephemeral=True)
            return

        if orchestrator is None:
            await interaction.response.send_message("⚠️ オーケストレーターがロードされていません。", ephemeral=True)
            return

        await interaction.response.defer()
        is_busy = True
        for child in self.children:
            child.disabled = True

        csv_info = f"\n📄 検証CSV: `{os.path.basename(self.preview_csvs[0])}`" if self.preview_csvs else ""
        await interaction.edit_original_response(
            content=f"🦾 **TACHIKOMA が物理タスクを実行中...**{csv_info}\n*(緊急停止は `!stop` または `!control` から可能)*",
            view=self,
        )

        # 司令塔 orchestrator に実行を委譲 (完了後に自動机上再スキャン)
        all_ok, exec_logs, updated_state = await asyncio.to_thread(
            orchestrator.execute_approved_plan,
            None,
            True
        )
        is_busy = False

        status_icon = "✅" if all_ok else "⚠️"
        result_embed = discord.Embed(
            title=f"{status_icon} 物理搬送シーケンス完了報告",
            description="\n".join(exec_logs),
            color=0x22C55E if all_ok else 0xEAB308
        )
        result_embed.set_footer(text=f"最新机上物体数: {updated_state.get('total_objects', 0)} 個")
        await interaction.followup.send(embed=result_embed)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary, emoji="⏹️")
    async def cancel(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ 指示を出した本人のみ操作できます。", ephemeral=True)
            return

        await interaction.response.defer()
        if orchestrator is not None:
            orchestrator.pending_plan = None

        for child in self.children:
            child.disabled = True
        await interaction.edit_original_response(content="🛑 **タスクをキャンセルしました。**", view=self)


# ==============================================================================
# 6. イベントハンドラ
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

        # 緊急停止コマンド
        if user_text in ("!stop", "!abort"):
            is_busy = False
            if orchestrator is not None:
                orchestrator.trigger_abort()
            _, msg = emergency_stop_arm(follower)
            await message.reply(f"🛑 **緊急停止指令を実行しました**: {msg}")
            return

        # 机上強制再スキャンコマンド
        if user_text == "!scan":
            async with message.channel.typing():
                if orchestrator is None:
                    await message.reply("⚠️ オーケストレーターが初期化されていません。")
                    return
                success, new_state, scan_msg = await asyncio.to_thread(orchestrator.refresh_vision, None, False)
                await message.reply(f"📸 **机上スキャン完了**: {scan_msg} (物体数: `{new_state.get('total_objects', 0)}`)")
            return

        if is_busy:
            await message.reply("⚠️ 現在タスク実行中です。緊急停止したい場合は `!stop` または `!control` を使用してください。")
            return

        if orchestrator is None:
            await message.reply("⚠️ オーケストレーターが初期化されていません。")
            return

        # 司令塔 orchestrator に指示を丸ごと委譲
        async with message.channel.typing():
            hw_context = await asyncio.to_thread(get_hardware_and_world_context)
            augmented_instruction = f"{hw_context}\n\n[ユーザーの発話]:\n{user_text}"

            res = await asyncio.to_thread(orchestrator.process_instruction, augmented_instruction, True)

            reply_text = res.get("reply_text", "了解いたしました！")
            tasks_list = res.get("tasks") or []
            ik_feasible = res.get("ik_feasible", True)
            preview_csvs = res.get("preview_csvs", [])

            # 物理動作を伴わない雑談・質問の場合
            if not tasks_list:
                await message.reply(f"💬 **TACHIKOMA:** {reply_text}")
                return

            # IK 幾何学的解が見つからなかった場合
            if not ik_feasible:
                await message.reply(f"💬 **TACHIKOMA:** {reply_text}")
                return

            # 搬送タスク計画の Embed 構築
            embed = discord.Embed(
                title="📋 TACHIKOMA タスク実行プラン (IK検証済)",
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

                d_r = d.get("r", 0.0) * 100.0
                d_th = d.get("theta_deg", 0.0)
                d_z = d.get("z", 0.015) * 1000.0

                field_val = (
                    f"• **Pick** : $r$={p_r:4.1f}cm, $\\theta$={p_th:+5.1f}°, $z$={p_z:4.1f}mm\n"
                    f"• **Place**: $r$={d_r:4.1f}cm, $\\theta$={d_th:+5.1f}°, $z$={d_z:4.1f}mm"
                )
                embed.add_field(name=f"Step {i}: {desc}", value=field_val, inline=False)

            if preview_csvs:
                csv_names = [f"`{os.path.basename(p)}`" for p in preview_csvs]
                embed.add_field(name="📄 検証用モーションCSV", value=", ".join(csv_names), inline=False)

            view = TaskConfirmView(user_id=message.author.id, preview_csvs=preview_csvs)
            await message.reply(embed=embed, view=view)

    except Exception as e:
        derr("ON_MESSAGE", f"メッセージ処理中に例外発生: {user_text}", e)
        await message.reply(f"❌ **内部エラーが発生しました (`{type(e).__name__}`)**: `{e}`")


# ==============================================================================
# メインエントリーポイント ＆ ノートPC単体テスト
# ==============================================================================
def main():
    global follower, motion_executor, orchestrator

    parser = argparse.ArgumentParser(description="TACHIKOMA Discord Bridge (Layer 1)")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    parser.add_argument("--test-ui", action="store_true", help="ノートPC用: Discordに接続せずUI・司令塔連携を単体テスト")
    args = parser.parse_args()

    # 実機アームの接続初期化
    if args.arm and STS3215Driver:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            dlog("STARTUP", f"実機フォロワー接続完了 ({FOLLOWER_PORT})", "OK")
        except Exception as e:
            derr("STARTUP", f"実機フォロワー接続失敗: {e}")
            follower = None

    # MotionExecutor および TachikomaOrchestrator のインスタンス化
    motion_executor = MotionExecutor(follower_driver=follower)
    orchestrator = TachikomaOrchestrator(motion_executor=motion_executor)

    if args.test_ui:
        print("=" * 70)
        print("🧪 [TEST] interfaces/discord_bot.py UI・司令塔統合 単体テスト")
        print("=" * 70)
        embed = build_status_embed_sync()
        print(f"1. 生成されたステータス Embed タイトル: {embed.title}")
        print(f"   Embed 説明文字数: {len(embed.description)} chars")
        print(f"   Embed フッター: {embed.footer.text}")

        print("\n2. オーケストレーター経由の自然言語処理テスト:")
        res = orchestrator.process_instruction("赤い木製ブロックを右奥に片付けて！")
        print(f"   応答発話: {res['reply_text']}")
        print(f"   サブタスク数: {len(res['tasks'])}")
        print(f"   IK到達性: {res['ik_feasible']}")
        print(f"   生成プレビューCSV: {res['preview_csvs']}")

        print("\n" + "=" * 70)
        print("🎉 discord_bot.py の統合単体テストが正常に完了しました！")
        print("=" * 70)
        return

    if not DISCORD_TOKEN:
        dlog("STARTUP", ".env に DISCORD_BOT_TOKEN が未設定のため Bot 接続はスキップします (テスト完了)。", "WARN")
        return

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()