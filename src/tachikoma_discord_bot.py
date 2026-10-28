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

【追加・拡張された主要機能】
1. 接続ハードウェア自動走査・列挙 (lsusb / lsblk / /dev/video* / アームポート)
2. カメラ排他制御スナップショット (📸 撮影) ＆ 軽量MJPEGストリーミング配信 (📹 ライブ)
3. Cockpit相当の7系統テレメトリ収集 ＆ GeminiモデルカスケードによるAIセキュリティ診断 (🧠 AI診断)
4. スマホ画面に最適化した省スペース3段グリッドUI (SystemControlView)
5. ターミナル即時デバッグ用の詳細トレースバック・タイムスタンプ付きログ出力
==============================================================================
"""

"""
==============================================================================
TACHIKOMA Discord Bridge ＆ システム自律防衛エージェント (src/tachikoma_discord_bot.py)
[高速フォールバック・Thinking無効化・進捗ライブ表示・カメラ即時断念 統合版]
==============================================================================
"""

import os
import io
import sys
import time
import json
import glob
import shutil
import subprocess
import threading
import asyncio
import argparse
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import psutil
import cv2
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
# デバッグログ出力ユーティリティ
# ==============================================================================
def dlog(tag: str, msg: str, level: str = "INFO"):
    """タイムスタンプ・タグ付きでターミナルへ即座にフラッシュ出力する"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    prefix_map = {
        "INFO": "ℹ️",
        "OK": "✅",
        "WARN": "⚠️",
        "ERR": "❌",
        "DEBUG": "🔍",
    }
    icon = prefix_map.get(level.upper(), "•")
    print(f"[{now}] {icon} [{tag}] {msg}", flush=True)


def derr(tag: str, context_msg: str, exc: Exception = None, show_tb: bool = True):
    """例外発生時にエラー概要をターミナルへ明示的に出力する"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"\n{'='*78}", flush=True)
    print(f"[{now}] ❌ [ERROR in {tag}] {context_msg}", flush=True)
    if exc is not None:
        print(f"Exception Type : {type(exc).__name__}", flush=True)
        print(f"Exception Detail: {exc}", flush=True)
    if show_tb:
        print("-" * 78, flush=True)
        traceback.print_exc(file=sys.stdout)
    print(f"{'='*78}\n", flush=True)


# ==============================================================================
# 環境変数・設定読み込み
# ==============================================================================
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

TARGET_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
if TARGET_GUILD_ID:
    TARGET_GUILD_ID = int(TARGET_GUILD_ID)

ALERT_CHANNEL_ID = os.getenv("DISCORD_ALERT_CHANNEL_ID")
if ALERT_CHANNEL_ID:
    ALERT_CHANNEL_ID = int(ALERT_CHANNEL_ID)

PROTECTED_SERVICE = os.getenv("PROTECTED_SERVICE_NAME", "my_web_app")

_env_safe_procs = os.getenv("SAFE_PROCESS_NAMES", "")
if _env_safe_procs.strip():
    SAFE_PROCESS_NAMES = {p.strip() for p in _env_safe_procs.split(",") if p.strip()}
else:
    SAFE_PROCESS_NAMES = {
        "python", "python3", "bash", "sh", "git", "pip", "sshd",
        "tailscaled", "systemd", "apt", "dpkg"
    }

DISK_USAGE_THRESHOLD = float(os.getenv("DISK_USAGE_THRESHOLD", "90.0"))
CPU_LOAD_THRESHOLD = float(os.getenv("CPU_LOAD_THRESHOLD", "3.8"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "900"))

CAMERA_DEVICE_INDEX = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
CAMERA_DEVICE_PATH = os.getenv("CAMERA_DEVICE_PATH", f"/dev/video{CAMERA_DEVICE_INDEX}")
STREAM_PORT = int(os.getenv("STREAM_PORT", "8088"))
STREAM_FPS = float(os.getenv("STREAM_FPS", "3.0"))
STREAM_EXTERNAL_URL = os.getenv("STREAM_EXTERNAL_URL", "").strip()

# 無料枠0の gemini-4-argon を除外し、高速なFlash系モデルをデフォルトに設定
_default_models = "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-2.5-flash"
_env_models = os.getenv("GEMINI_MODEL_CASCADE", _default_models)
GEMINI_MODEL_CASCADE = [
    m.strip() for m in _env_models.split(",")
    if m.strip() and "argon" not in m.strip().lower()
]
if not GEMINI_MODEL_CASCADE:
    GEMINI_MODEL_CASCADE = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

is_busy = False
abort_arm_flag = False
last_alert_time = 0.0
camera_lock = threading.Lock()

planner = LLMTaskPlanner()
generator = ParametricMotionGenerator()
follower = None

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ==============================================================================
# 1. 接続ハードウェア走査・列挙 ＆ 物理コンテキスト生成
# ==============================================================================
def get_connected_hardware_list():
    """接続中の機器（アーム・カメラ・外部ストレージ・USB）を箇条書きリストで取得"""
    dlog("HW_SCAN", "接続ハードウェアの走査を開始...", "DEBUG")
    hw_items = []

    # (1) ロボットアーム接続確認
    try:
        if follower is not None and getattr(follower, "is_open", False):
            hw_items.append(f"🦾 **アーム**: 接続中・通信可能 (`{FOLLOWER_PORT}`)")
        elif os.path.exists(FOLLOWER_PORT):
            hw_items.append(f"🦾 **アーム**: ポート認識済・未オープン (`{FOLLOWER_PORT}`)")
        else:
            hw_items.append(f"🦾 **アーム**: 未接続 (`{FOLLOWER_PORT}` 非存在)")
    except Exception as e:
        derr("HW_SCAN", "アーム接続状態の確認中にエラー", e, show_tb=False)
        hw_items.append(f"🦾 **アーム**: 状態取得エラー (`{e}`)")

    # (2) カメラデバイス確認 (/dev/video*)
    try:
        video_devs = sorted(glob.glob("/dev/video*"))
        if video_devs:
            readable = os.access(CAMERA_DEVICE_PATH, os.R_OK | os.W_OK)
            perm_str = "権限OK" if readable else "⚠️権限不足(Permission denied)"
            hw_items.append(f"📷 **カメラ**: 認識済 (`{', '.join(video_devs)}` / {perm_str})")
        else:
            hw_items.append("📷 **カメラ**: 未接続 (`/dev/video*` なし)")
    except Exception as e:
        derr("HW_SCAN", "カメラデバイスの走査中にエラー", e, show_tb=False)
        hw_items.append(f"📷 **カメラ**: 走査エラー (`{e}`)")

    # (3) 外部ストレージ確認
    try:
        res = subprocess.run(
            ["lsblk", "-J", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,RM"],
            capture_output=True, text=True, timeout=4
        )
        data = json.loads(res.stdout)
        ext_storages = []

        def scan_blocks(blocks):
            for b in blocks:
                mnt = b.get("mountpoint")
                rm = b.get("rm")
                b_type = b.get("type")
                name = b.get("name")
                size = b.get("size")
                if mnt and mnt not in ("/", "/boot", "/boot/efi", "[SWAP]") and not str(mnt).startswith("/snap"):
                    ext_storages.append(f"`/dev/{name}` ({size} -> `{mnt}`)")
                elif rm in (True, "1", 1) and b_type in ("part", "disk") and not mnt:
                    ext_storages.append(f"`/dev/{name}` ({size}, 未マウント)")
                if "children" in b:
                    scan_blocks(b["children"])

        scan_blocks(data.get("blockdevices", []))
        if ext_storages:
            hw_items.append(f"💾 **外部ストレージ**: {', '.join(ext_storages)}")
        else:
            hw_items.append("💾 **外部ストレージ**: なし (内蔵SSDのみ)")
    except Exception as e:
        derr("HW_SCAN", "lsblk による外部ストレージ走査中にエラー", e, show_tb=False)
        hw_items.append("💾 **外部ストレージ**: 取得失敗")

    # (4) 主要USBデバイス
    try:
        usb_res = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=4)
        usb_names = []
        for line in usb_res.stdout.splitlines():
            if "Linux Foundation" in line or "root hub" in line.lower():
                continue
            parts = line.split("ID ")
            if len(parts) > 1:
                desc = parts[1].split(" ", 1)
                if len(desc) > 1:
                    usb_names.append(desc[1].strip())
        if usb_names:
            hw_items.append(f"🔌 **USB機器**: {', '.join(usb_names[:5])}")
        else:
            hw_items.append("🔌 **USB機器**: 外部USB機器なし")
    except Exception as e:
        derr("HW_SCAN", "lsusb によるUSB機器走査中にエラー", e, show_tb=False)
        hw_items.append("🔌 **USB機器**: 取得失敗")

    return hw_items


def get_hardware_and_world_context():
    """現在の実機接続状態、カメラ状態、机上物体情報を取得してプロンプト用テキストを生成"""
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
                        f"- [ID #{o.get('id')}] 名称: {o.get('name', '物体')}, "
                        f"色: {o.get('color', '不明')}, "
                        f"座標: (X={o.get('x_mm', 0):.1f}mm, Y={o.get('y_mm', 0):.1f}mm, Z={o.get('z_mm', 0):.1f}mm), "
                        f"角度: {o.get('angle_deg', 0):.1f}°"
                        for o in objs
                    ]
                    objects_summary = "【机の上にある検知済み物体リスト】:\n" + "\n".join(lines)
        except Exception as e:
            derr("WORLD_CTX", f"{world_state_path} の読み込みエラー", e, show_tb=False)

    return f"""
[現在のタチコマ物理環境ステータス]
{hw_summary}
- 机上認識状況:
{objects_summary}
※ユーザーから机の上の状況や接続状態を尋ねられた場合は、上記の実測情報を基に正確に回答してください。存在しない物体をでっち上げないでください。
"""


# ==============================================================================
# 2. カメラ機能 (静止画スナップショット ＆ エラー時即時断念つき MJPEG サーバー)
# ==============================================================================
def make_error_jpeg_bytes(err_text: str) -> bytes:
    """カメラエラー時にスマホ画面が真っ暗にならないよう、状況を説明するプレースホルダー画像を生成"""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (25, 25, 30)
    cv2.putText(img, "CAMERA OFFLINE / ERROR", (95, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 255), 2)
    cv2.putText(img, "Stream stopped (Fail-Fast Mode)", (125, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    short_err = (err_text or "Check device or permissions")[:55]
    cv2.putText(img, short_err, (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


def capture_jpeg_frame_sync(log_error: bool = True):
    """カメラを1フレームだけ開いてJPEGバイト列を取得し、即座に解放する"""
    if not os.path.exists(CAMERA_DEVICE_PATH):
        msg = f"カメラデバイス ({CAMERA_DEVICE_PATH}) が存在しません。"
        if log_error:
            dlog("CAMERA", msg, "WARN")
        return None, msg

    if not os.access(CAMERA_DEVICE_PATH, os.R_OK | os.W_OK):
        msg = (
            f"カメラ ({CAMERA_DEVICE_PATH}) のアクセス権限がありません (Permission denied)。\n"
            f"ターミナルで `sudo setfacl -m u:$USER:rw /dev/video*` を実行してください。"
        )
        if log_error:
            dlog("CAMERA", msg.replace("\n", " "), "ERR")
        return None, msg

    with camera_lock:
        cap = None
        try:
            # OpenCVの不要な大量バックエンド警告を抑制するため V4L2 のみで試行
            cap = cv2.VideoCapture(CAMERA_DEVICE_PATH, cv2.CAP_V4L2)
            if not cap.isOpened():
                msg = f"カメラ ({CAMERA_DEVICE_PATH}) をオープンできませんでした（他プロセスが使用中の可能性があります）。"
                if log_error:
                    dlog("CAMERA", msg, "ERR")
                return None, msg

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            for _ in range(3):
                cap.read()

            ret, frame = cap.read()
            if not ret or frame is None:
                msg = "カメラフレームの読み取り(cap.read)に失敗しました。"
                if log_error:
                    dlog("CAMERA", msg, "ERR")
                return None, msg

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                msg = "フレームのJPEGエンコードに失敗しました。"
                if log_error:
                    dlog("CAMERA", msg, "ERR")
                return None, msg

            return buf.tobytes(), None
        except Exception as e:
            if log_error:
                derr("CAMERA", "カメラ撮影処理中に例外が発生しました", e, show_tb=False)
            return None, f"カメラ例外: {e}"
        finally:
            if cap is not None:
                cap.release()


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """スマホブラウザ向け軽量MJPEGストリーム配信ハンドラ (エラー時は1回で即座に終了)"""
    def log_message(self, format, *args):
        pass  # アクセスログによるターミナル汚染を防止

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (
                "<html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                "<title>TACHIKOMA Live Camera</title>"
                "<style>body{background:#111;color:#eee;text-align:center;font-family:sans-serif;margin:0;padding:16px;}"
                "img{max-width:100%;height:auto;border:2px solid #38bdf8;border-radius:8px;}"
                ".btn{display:inline-block;margin-top:12px;padding:8px 16px;background:#38bdf8;color:#000;"
                "text-decoration:none;border-radius:6px;font-weight:bold;}</style></head>"
                "<body><h3>📹 TACHIKOMA Live View</h3><img src='/stream.mjpg' /><br/>"
                "<a class='btn' href='/'>🔄 再読み込み</a></body></html>"
            )
            self.wfile.write(html.encode("utf-8"))

        elif self.path.startswith("/stream.mjpg"):
            dlog("MJPEG_STREAM", f"クライアント ({self.client_address[0]}) がストリームに接続しました", "INFO")
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            interval = 1.0 / max(0.5, STREAM_FPS)
            try:
                while True:
                    start_t = time.time()
                    # ループ内では同じエラーを連投しないよう log_error=False で呼び出し、ここで1回だけ出す
                    jpg_bytes, err = capture_jpeg_frame_sync(log_error=False)
                    if not jpg_bytes:
                        dlog("MJPEG_STREAM", f"カメラ接続不可のためストリーム配信を即時中断します (理由: {err})", "WARN")
                        # スマホ画面が真っ暗にならないよう、エラー案内画像を1枚だけ送ってループを終了する
                        err_img = make_error_jpeg_bytes(err)
                        if err_img:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(err_img)))
                            self.end_headers()
                            self.wfile.write(err_img)
                            self.wfile.write(b"\r\n")
                        break  # 1回失敗したら再試行せず即座に諦める

                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg_bytes)))
                    self.end_headers()
                    self.wfile.write(jpg_bytes)
                    self.wfile.write(b"\r\n")

                    elapsed = time.time() - start_t
                    time.sleep(max(0.05, interval - elapsed))
            except (BrokenPipeError, ConnectionResetError):
                dlog("MJPEG_STREAM", f"クライアント ({self.client_address[0]}) がストリーム視聴を閉じました", "INFO")
            except Exception as e:
                derr("MJPEG_STREAM", "MJPEG配信中にエラーが発生しました", e, show_tb=False)
        else:
            self.send_error(404)


def start_mjpeg_server_daemon():
    """バックグラウンドスレッドでMJPEGサーバーを起動"""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", STREAM_PORT), MJPEGStreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        dlog("MJPEG_SERVER", f"カメラ配信サーバー起動完了: http://127.0.0.1:{STREAM_PORT} (FPS={STREAM_FPS})", "OK")
        if STREAM_EXTERNAL_URL:
            dlog("MJPEG_SERVER", f"外部アクセスURL (Tailscale Serve): {STREAM_EXTERNAL_URL}", "INFO")
    except Exception as e:
        derr("MJPEG_SERVER", f"ポート {STREAM_PORT} でのMJPEGサーバー起動に失敗しました", e, show_tb=False)


# ==============================================================================
# 3. 7系統メトリクス収集 ＆ Gemini 高速カスケード診断 (リトライ無効化・Thinking OFF)
# ==============================================================================
def _run_cmd_safe(cmd_list, timeout=5, label="CMD"):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
        out = res.stdout.strip()
        if res.returncode != 0 and not out:
            err_txt = res.stderr.strip()
            return f"(コマンド終了コード {res.returncode}: {err_txt or '出力なし'})"
        return out or "(該当データなし)"
    except Exception as e:
        return f"(取得エラー: {e})"


def collect_7_security_metrics_sync():
    """Cockpit相当の7系統システム・セキュリティデータを一括収集"""
    dlog("AI_DIAG", "7系統のシステム・セキュリティメトリクス収集を開始...", "INFO")

    ext_conns = []
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == "ESTABLISHED" and c.raddr:
                ext_conns.append(f"PID:{c.pid} {c.laddr.ip}:{c.laddr.port} -> {c.raddr.ip}:{c.raddr.port}")
    except Exception:
        ext_conns.append(_run_cmd_safe(["ss", "-tunp", "state", "established"], label="ss_established"))
    sec1 = "\n".join(ext_conns[:20]) if ext_conns else "外部ESTABLISHED接続なし"

    sec2 = _run_cmd_safe(
        ["journalctl", "-u", "ssh", "-u", "cockpit", "--since", "1 hour ago", "-p", "warning", "--no-pager", "-n", "20"],
        label="auth_log"
    )
    ps_raw = _run_cmd_safe(["ps", "-eo", "pid,user,%cpu,%mem,comm", "--sort=-%cpu"], label="top_ps")
    sec3 = "\n".join(ps_raw.splitlines()[:11])
    sec4 = _run_cmd_safe(["systemctl", "--failed", "--no-pager"], label="systemd_failed")
    sec5 = _run_cmd_safe(["journalctl", "_COMM=sudo", "--since", "1 hour ago", "--no-pager", "-n", "15"], label="sudo_log")
    sec6 = _run_cmd_safe(["journalctl", "-p", "err..emerg", "--since", "1 hour ago", "--no-pager", "-n", "15"], label="err_log")

    temps = []
    try:
        s_temps = psutil.sensors_temperatures()
        for name, entries in s_temps.items():
            for e in entries:
                temps.append(f"{name}({e.label or 'core'}): {e.current}°C")
    except Exception:
        pass

    usage = shutil.disk_usage("/")
    disk_info = f"Root Disk: {(usage.used / usage.total) * 100:.1f}% used ({usage.free // (1024**3)}GB free)"
    sec7 = (", ".join(temps) if temps else "温度センサー値なし") + f"\n{disk_info}"

    report = f"""
[1. 外部TCP接続(ESTABLISHED)]
{sec1}

[2. 認証・ログイン警告ログ(直近1時間)]
{sec2}

[3. リソース消費上位プロセス(Top 10)]
{sec3}

[4. 失敗サービス(systemctl --failed)]
{sec4}

[5. 直近のsudo特権実行履歴(直近1時間)]
{sec5}

[6. OS・カーネルエラーログ(直近1時間)]
{sec6}

[7. ハードウェア温度・ストレージ健全性]
{sec7}
""".strip()

    dlog("AI_DIAG", f"メトリクス収集完了 (総文字数: {len(report)} chars)", "OK")
    return report


def call_single_gemini_model_sync(model_name: str, prompt: str) -> str:
    """
    指定した単一モデルに対して、内部自動リトライ(tenacity)を無効化 ＆ Thinking OFF で高速リクエストを送る。
    失敗時は86秒待たずに即座に例外を投げて次のモデルへフォールバックさせる。
    """
    from google import genai
    from google.genai import types

    # SDK内部の自動リトライ(最大5回・約86秒の待機)を無効化(attempts=1)し、タイムアウトを15秒に制限
    http_opts = None
    try:
        http_opts = types.HttpOptions(
            timeout=15000,  # 15秒 (ミリ秒指定)
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except Exception:
        try:
            http_opts = types.HttpOptions(timeout=15000)
        except Exception:
            http_opts = None

    client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts) if http_opts else genai.Client(api_key=GEMINI_API_KEY)

    # Thinkingモードを明示的にOFF (thinking_budget=0) にして応答速度を最大化
    gen_config = None
    try:
        gen_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
        )
    except Exception:
        gen_config = None

    try:
        if gen_config is not None:
            resp = client.models.generate_content(model=model_name, contents=prompt, config=gen_config)
        else:
            resp = client.models.generate_content(model=model_name, contents=prompt)
    except Exception as first_err:
        # モデルが thinking_config 未対応だった場合のみ config なしで1回だけ即時呼び出し
        if "thinking" in str(first_err).lower():
            resp = client.models.generate_content(model=model_name, contents=prompt)
        else:
            raise first_err

    if resp and resp.text:
        return resp.text.strip()
    raise RuntimeError(f"モデル {model_name} から空の応答が返されました。")


# ==============================================================================
# 4. 実機アーム制御シーケンス
# ==============================================================================
def execute_tasks_sync(tasks_list):
    """実機サーボへの物理搬送シーケンス（緊急停止割り込み対応）"""
    global follower, abort_arm_flag
    abort_arm_flag = False

    if follower is None or not getattr(follower, "is_open", False):
        dlog("ARM_EXEC", "実機アーム未接続のためプレビューモードとしてスキップしました", "WARN")
        return "⚠️ 実機アームが接続されていません（プレビューのみモード）"

    try:
        dlog("ARM_EXEC", f"搬送タスク開始 (全 {len(tasks_list)} ステップ)", "INFO")
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
        derr("ARM_EXEC", "実機サーボ搬送シーケンス中に例外が発生しました", e)
        return f"❌ 実行中エラー: {e}"
    finally:
        for sid in SERVO_IDS:
            try:
                follower.set_torque(sid, False)
            except Exception:
                pass


# ==============================================================================
# 5. コンパクト緊急制御パネル UI (SystemControlView)
# ==============================================================================
class SystemControlView(View):
    """
    省スペース3段グリッド配置:
    Row 0 (観測・診断): [ 📸 撮影 ] [ 📹 ライブ ] [ 🧠 AI診断 ]
    Row 1 (停止・再開): [ 🛑 アーム ] [ 🛑 Web ] [ ▶️ Web ]
    Row 2 (最終防壁)  : [ 🔌 外部隔離 ]
    """
    def __init__(self):
        super().__init__(timeout=None)
        if STREAM_EXTERNAL_URL:
            self.add_item(
                Button(
                    label="📹 ライブ",
                    style=discord.ButtonStyle.link,
                    url=STREAM_EXTERNAL_URL,
                    row=0,
                )
            )

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item):
        derr("UI_VIEW", f"ボタン ({getattr(item, 'label', item)}) の処理中に例外が発生しました", error)
        err_msg = f"❌ **ボタン処理エラー (`{type(error).__name__}`)**: `{error}`"
        try:
            if interaction.response.is_done():
                await interaction.followup.send(err_msg, ephemeral=True)
            else:
                await interaction.response.send_message(err_msg, ephemeral=True)
        except Exception:
            pass

    @discord.ui.button(label="📸 撮影", style=discord.ButtonStyle.secondary, custom_id="sys_cam_snap", row=0)
    async def snap_camera(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[📸 撮影] が押されました (User: {interaction.user})", "INFO")
        await interaction.response.send_message("📸 **カメラ撮影を開始しました...**")

        async with interaction.channel.typing():
            jpg_bytes, err = await asyncio.to_thread(capture_jpeg_frame_sync, True)
            if err or not jpg_bytes:
                await interaction.followup.send(f"⚠️ **撮影失敗**: {err}")
                return
            file = discord.File(io.BytesIO(jpg_bytes), filename="snapshot.jpg")

        dlog("UI_BTN", "スナップショット撮影・Discord送信完了", "OK")
        await interaction.followup.send("✅ **現在のカメラ視界**", file=file)

    @discord.ui.button(label="🧠 AI診断", style=discord.ButtonStyle.primary, custom_id="sys_ai_diag", row=0)
    async def ai_diagnosis(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[🧠 AI診断] が押されました (User: {interaction.user})", "INFO")

        if not GEMINI_API_KEY:
            await interaction.response.send_message("❌ `.env` に `GEMINI_API_KEY` が設定されていません。", ephemeral=True)
            return

        # 1. ボタン押下直後に「開始メッセージ」を即座に確定返信する
        await interaction.response.send_message(
            "🧠 **AIセキュリティ診断を開始しました。**\n"
            "7系統のシステムログ収集とGeminiによる解析を行っています。完了まで数秒お待ちください..."
        )

        t0 = time.time()
        diag_text = None
        used_model = "Failed"
        errors = []

        # 2. 診断処理が終わるまで「TACHIKOMA が入力中...」を表示し続ける
        async with interaction.channel.typing():
            raw_metrics = await asyncio.to_thread(collect_7_security_metrics_sync)
            prompt = f"""
あなたはLinuxサーバー（Ubuntu 24.04 / Tailscale環境）の専属セキュリティ・インフラ診断AIです。
以下の7系統の実測テレメトリデータを分析し、スマホ画面で読みやすいよう簡潔な日本語の箇条書きで診断結果を出力してください。

【出力フォーマット】
1. **総合判定**: （🟢安全 / 🟡要注意 / 🔴危険 のいずれかと一言理由）
2. **通信・認証の安全性**: （不審な外部IP接続や不正ログイン試行の有無）
3. **プロセス・サービスの健全性**: （暴走プロセスやfailedサービスの有無）
4. **ハードウェア・OS状態**: （温度・ディスク・エラーログの評価と推奨アクション）

【実測テレメトリデータ】
{raw_metrics}
"""

            for model_name in GEMINI_MODEL_CASCADE:
                dlog("AI_DIAG", f"Gemini API 呼び出し試行中... (Model: {model_name})", "INFO")
                m_start = time.time()
                try:
                    diag_text = await asyncio.to_thread(call_single_gemini_model_sync, model_name, prompt)
                    used_model = model_name
                    elapsed_m = time.time() - m_start
                    dlog("AI_DIAG", f"Gemini 診断成功! (Model: {model_name}, 所要時間: {elapsed_m:.2f}秒)", "OK")
                    break
                except Exception as e:
                    elapsed_m = time.time() - m_start
                    short_reason = str(e).splitlines()[0][:120]
                    derr("AI_DIAG", f"モデル '{model_name}' 失敗 ({elapsed_m:.2f}秒) -> 次のモデルへ即時フォールバックします", e, show_tb=False)
                    errors.append(f"• `{model_name}`: {short_reason}")

        total_sec = time.time() - t0
        if diag_text is None:
            err_block = "\n".join(errors)
            diag_text = f"❌ **すべてのモデル候補でリクエストが失敗しました。**\n{err_block}"

        embed = discord.Embed(
            title="🧠 Gemini システム・セキュリティ診断レポート",
            description=diag_text[:4000],
            color=0x8B5CF6 if used_model != "Failed" else 0xEF4444,
        )
        embed.set_footer(
            text=f"使用モデル: {used_model} | 処理時間: {total_sec:.1f}秒 | 対象: ①通信 ②認証 ③PS ④SVC ⑤sudo ⑥エラー ⑦温度/SSD"
        )

        # 3. 入力中表示が消えると同時に、診断結果レポートを新しいメッセージとして送信
        await interaction.followup.send(content="✅ **AIセキュリティ診断が完了しました！**", embed=embed)

    @discord.ui.button(label="🛑 アーム", style=discord.ButtonStyle.danger, custom_id="sys_stop_arm", row=1)
    async def stop_arm(self, interaction: discord.Interaction, button: Button):
        global is_busy, abort_arm_flag, follower
        dlog("UI_BTN", f"[🛑 アーム] 緊急停止が押されました (User: {interaction.user})", "WARN")
        await interaction.response.defer()
        abort_arm_flag = True
        is_busy = False

        if follower is not None and getattr(follower, "is_open", False):
            for sid in SERVO_IDS:
                try:
                    follower.set_torque(sid, False)
                except Exception as e:
                    derr("UI_BTN", f"サーボ ID:{sid} のトルク解除中にエラー", e, show_tb=False)

        await interaction.followup.send("🛑 **ロボットアームに緊急停止・脱力指令を送信しました。**")

    @discord.ui.button(label="🛑 Web", style=discord.ButtonStyle.danger, custom_id="sys_stop_service", row=1)
    async def stop_service(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[🛑 Web] サービス停止が押されました (Target: {PROTECTED_SERVICE})", "WARN")
        await interaction.response.defer()
        res = await asyncio.to_thread(
            subprocess.run,
            ["sudo", "systemctl", "stop", PROTECTED_SERVICE],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            await interaction.followup.send(f"⚠️ `{PROTECTED_SERVICE}.service` 停止エラー:\n```\n{res.stderr.strip()}\n```")
            return
        await interaction.followup.send(f"🛑 `{PROTECTED_SERVICE}.service` を緊急停止しました。")

    @discord.ui.button(label="▶️ Web", style=discord.ButtonStyle.success, custom_id="sys_start_service", row=1)
    async def start_service(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[▶️ Web] サービス再開が押されました (Target: {PROTECTED_SERVICE})", "INFO")
        await interaction.response.defer()
        await asyncio.to_thread(
            subprocess.run,
            ["sudo", "systemctl", "start", PROTECTED_SERVICE],
            capture_output=True, text=True
        )
        res = await asyncio.to_thread(
            subprocess.run,
            ["sudo", "systemctl", "is-active", PROTECTED_SERVICE],
            capture_output=True,
            text=True,
        )
        state_str = res.stdout.strip()
        await interaction.followup.send(f"▶️ `{PROTECTED_SERVICE}.service` を再開しました（状態: `{state_str}`）。")

    @discord.ui.button(label="🔌 外部隔離", style=discord.ButtonStyle.secondary, custom_id="sys_isolate_net", row=2)
    async def isolate_net(self, interaction: discord.Interaction, button: Button):
        dlog("UI_BTN", f"[🔌 外部隔離] が押されました！3秒後に tailscale down を実行します", "WARN")
        await interaction.response.defer()
        await interaction.followup.send(
            "🚨 **3秒後に Tailscale VPN を切断し、サーバーを外部ネットワークから完全隔離します。**\n"
            "※復旧にはサーバー本体へ物理コンソールを接続する必要があります。"
        )
        await asyncio.sleep(3)
        await asyncio.to_thread(subprocess.run, ["sudo", "tailscale", "down"])


# ==============================================================================
# 6. 段階的防衛 (Tiered Defense) バックグラウンド監視ロジック
# ==============================================================================
def check_system_security_sync():
    warnings = []
    auto_actions = []

    try:
        usage = shutil.disk_usage("/")
        disk_pct = (usage.used / usage.total) * 100
        if disk_pct > DISK_USAGE_THRESHOLD:
            warnings.append(f"💾 **ディスク残量警告**: 使用率 `{disk_pct:.1f}%` (閾値: `{DISK_USAGE_THRESHOLD}%`)")

        load1, _, _ = os.getloadavg()
        if load1 > CPU_LOAD_THRESHOLD:
            warnings.append(f"🔥 **CPU高負荷**: 1分平均ロード `{load1:.2f}` (閾値: `{CPU_LOAD_THRESHOLD}`)")

        ssh_check = subprocess.run(
            ["journalctl", "-u", "ssh", "--since", "2 minutes ago", "--no-pager"],
            capture_output=True,
            text=True,
        )
        failed_ssh = sum(1 for line in ssh_check.stdout.splitlines() if "Failed password" in line)
        if failed_ssh > 0:
            warnings.append(f"🚨 **SSH不正ログイン試行**: 直近2分間で `{failed_ssh}` 件の失敗を検知")

        suspicious_dirs = ("/tmp/", "/var/tmp/", "/dev/shm/")
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                p_info = proc.info
                exe_path = p_info.get("exe") or ""
                p_name = p_info.get("name") or ""

                if any(exe_path.startswith(d) for d in suspicious_dirs) and p_name not in SAFE_PROCESS_NAMES:
                    pid = p_info["pid"]
                    dlog("SEC_DEFENSE", f"不審プロセス検知! PID={pid}, Name={p_name}, Path={exe_path} -> 即時Kill実行", "ERR")
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
    except Exception as e:
        derr("SEC_MONITOR", "定期セキュリティ監視スキャン中にエラー", e, show_tb=False)

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
        embed.set_footer(text="異常がある場合は下のボタンから手動で緊急停止・AI診断を実行できます。")
        await channel.send(embed=embed, view=SystemControlView())
        last_alert_time = time.time()


# ==============================================================================
# 7. ロボットタスク確認ボタン UI ＆ ステータスパネル生成
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


def build_status_embed_sync():
    usage = shutil.disk_usage("/")
    disk_pct = (usage.used / usage.total) * 100
    load1, _, _ = os.getloadavg()
    mem = psutil.virtual_memory()

    svc_res = subprocess.run(
        ["sudo", "systemctl", "is-active", PROTECTED_SERVICE],
        capture_output=True,
        text=True,
    )
    hw_items = get_connected_hardware_list()
    hw_block = "\n".join(f"• {item}" for item in hw_items)

    desc = (
        f"**【システムメトリクス】**\n"
        f"• **CPU 1分負荷**: `{load1:.2f}`\n"
        f"• **メモリ使用率**: `{mem.percent:.1f}%` (`{mem.used // (1024**2)}MB` / `{mem.total // (1024**2)}MB`)\n"
        f"• **ディスク使用率**: `{disk_pct:.1f}%`\n"
        f"• **保護対象サービス (`{PROTECTED_SERVICE}`)**: `{svc_res.stdout.strip()}`\n"
        f"• **アーム状態**: `{'動作中' if is_busy else '待機中'}`\n\n"
        f"**【接続ハードウェア一覧】**\n"
        f"{hw_block}"
    )

    embed = discord.Embed(
        title="🛠️ システム状態監視 ＆ 緊急制御パネル",
        description=desc,
        color=0x38BDF8,
    )
    embed.set_footer(text=f"AI診断モデル優先順: {' -> '.join(GEMINI_MODEL_CASCADE)}")
    return embed


# ==============================================================================
# 8. Discord Bot イベントハンドラ ＆ エントリーポイント
# ==============================================================================
@bot.event
async def on_ready():
    bot.add_view(SystemControlView())

    if not security_monitor_task.is_running():
        security_monitor_task.start()

    guild_info = f" (対象Guild: {TARGET_GUILD_ID})" if TARGET_GUILD_ID else " (全サーバー対象)"
    dlog("BOT_READY", f"TACHIKOMA Discord Bot 起動完了: {bot.user.name} (ID: {bot.user.id}){guild_info}", "OK")
    dlog("BOT_READY", f"段階的防衛モニタリング稼働中 (通知先Ch: {ALERT_CHANNEL_ID}, 保護SVC: {PROTECTED_SERVICE})", "INFO")
    dlog("BOT_READY", f"AI診断カスケードモデル (高速・Thinking OFF): {GEMINI_MODEL_CASCADE}", "INFO")


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

    dlog("DISCORD_MSG", f"受信 [{message.channel} / {message.author}]: {user_text}", "INFO")

    try:
        if user_text in ("!control", "!status"):
            embed = await asyncio.to_thread(build_status_embed_sync)
            await message.reply(embed=embed, view=SystemControlView())
            dlog("DISCORD_MSG", "!control パネルを送信しました", "OK")
            return

        if is_busy:
            await message.reply("⚠️ 現在タスク実行中です。緊急停止したい場合は `!control` と入力して停止ボタンを押してください。")
            return

        async with message.channel.typing():
            t_start = time.time()
            hw_context = await asyncio.to_thread(get_hardware_and_world_context)
            full_prompt = f"{hw_context}\n\n[ユーザーからの発話]:\n{user_text}"

            dlog("LLM_PLAN", "LLMTaskPlanner.plan() を呼び出し中...", "DEBUG")
            plan = await asyncio.to_thread(planner.plan, full_prompt)
            elapsed = time.time() - t_start

            reply_text = plan.get("reply_text", "了解しました。")
            tasks_list = plan.get("tasks") or []
            dlog("LLM_PLAN", f"プラン生成完了 (所要時間: {elapsed:.2f}秒, サブタスク数: {len(tasks_list)})", "OK")

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

    except Exception as e:
        derr("ON_MESSAGE", f"メッセージ処理中に例外が発生しました (入力: {user_text})", e)
        await message.reply(f"❌ **内部エラーが発生しました (`{type(e).__name__}`)**: `{e}`")


def main():
    global follower
    parser = argparse.ArgumentParser(description="TACHIKOMA Discord Bridge")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    parser.add_argument("--no-stream", action="store_true", help="MJPEGカメラ配信サーバーを無効化")
    args = parser.parse_args()

    dlog("STARTUP", "=== TACHIKOMA Discord Bot 起動シーケンス開始 ===", "INFO")
    dlog("STARTUP", f"BASE_DIR: {BASE_DIR}", "DEBUG")
    dlog("STARTUP", f"CAMERA_DEVICE_PATH: {CAMERA_DEVICE_PATH} (Index: {CAMERA_DEVICE_INDEX})", "DEBUG")

    if args.arm:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            dlog("STARTUP", f"実機フォロワー接続完了 ({FOLLOWER_PORT})", "OK")
        except Exception as e:
            derr("STARTUP", f"実機フォロワー ({FOLLOWER_PORT}) の接続に失敗しました", e, show_tb=False)
            follower = None
    else:
        dlog("STARTUP", "--arm フラグ未指定のため、アーム未接続(シミュレーション/監視モード)で起動します", "INFO")

    if not args.no_stream:
        start_mjpeg_server_daemon()
    else:
        dlog("STARTUP", "--no-stream フラグによりMJPEG配信サーバーをスキップしました", "INFO")

    if not DISCORD_TOKEN:
        dlog("STARTUP", ".env に DISCORD_BOT_TOKEN が設定されていません。終了します。", "ERR")
        return

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()