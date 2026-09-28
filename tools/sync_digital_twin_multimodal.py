"""
==============================================================================
マルチモーダル・デジタルツイン同期ツール (Gemini / Qwen 切替対応版)
(tools/sync_digital_twin_multimodal.py)
==============================================================================
【機能】
1. カメラ画像から全物体を OpenCV で検出し、物理座標・傾き・ミリ寸法を計測。
2. [C] キー入力で全物体をコラージュ台紙化して VLM に一括推論リクエスト。
3. 実行時引数に応じて推論エンジンを切り替え・フォールバック：
   - --gemini        : Gemini API によるクラウド一括同定
   - --qwen          : ローカル Ollama (qwen2.5vl:3b) による完全オフライン同定
   - --gemini --qwen : Gemini 試行 ➔ 全モデル失敗時に Qwen へ自動フォールバック
4. 計測されたミリ寸法 (長辺x短辺) に合わせて MuJoCo の直方体形状を動的変形。
5. MuJoCo 画面上の各物体の直上に物体名ラベルを 3D オーバーレイ表示。

【操作】
  - [C]     : 全物体を同定し MuJoCo に直方体形状＆ラベルを反映
  - [B]     : 机面背景の記憶 (高精度差分)
  - [SPACE] : マーカー正射影の再計算
  - [Q/ESC] : 終了
==============================================================================
"""

import sys
import os
import time
import json
import math
import re
import argparse
import base64
import urllib.request
import urllib.error
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.kinematics import get_home_radians

# Google GenAI SDK
from google import genai
from google.genai import types

import threading
from contextlib import contextmanager

# キャンバス・スロット設定
CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015  # 直方体の厚み (15mm)
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

MIN_AREA_PX = 600
MAX_AREA_PX = 30000

# Ollama 設定
OLLAMA_API_URL = "http://localhost:11434/api/generate"
QWEN_MODEL_NAME = "qwen2.5vl:3b"

# Gemini 試行モデル候補
CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]

# 操作要求フラグ
REQ_SAVE_BG = False
REQ_RECALIB = False
REQ_CLASSIFY = False
REQ_QUIT = False


# ==============================================================================
# 1. Pydantic スキーマ
# ==============================================================================
class IdentifiedItem(BaseModel):
    id: int = Field(description="画像内のタイル番号 (#0, #1 等)")
    category: str = Field(description="物体の一般名 (例: pen, wooden block, eraser, mouse, ruler)")
    color: str = Field(description="物体の主要な色 (例: red, blue, silver, natural wood, black)")
    description: str = Field(description="簡潔な特徴表現")

class CollageClassificationResult(BaseModel):
    items: List[IdentifiedItem] = Field(description="同定結果リスト")


# ==============================================================================
# 2. 幾何計算 & 座標系ユーティリティ
# ==============================================================================
def create_marker_mask(size: int = CANVAS_SIZE, margin: int = MARGIN) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pad = margin + 15
    mask[pad:size - pad, pad:size - pad] = 255
    return mask


def pixel_to_robot_phys_xy(u: float, v: float, projector: VisionProjector) -> Tuple[float, float]:
    p0 = projector.marker_phys_xy[0]
    p1 = projector.marker_phys_xy[1]
    p2 = projector.marker_phys_xy[2]
    p3 = projector.marker_phys_xy[3]

    s = (u - MARGIN) / INNER_SPAN_PX
    t = (v - MARGIN) / INNER_SPAN_PX

    top = (1.0 - s) * p0 + s * p1
    bottom = (1.0 - s) * p2 + s * p3
    phys_xy = (1.0 - t) * top + t * bottom
    return float(phys_xy[0]), float(phys_xy[1])


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def build_adaptive_collage(crops: List[np.ndarray], tile_size: int = 160) -> Tuple[np.ndarray, int, int]:
    n = len(crops)
    if n == 0:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8), 0, 0

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    collage = np.full((rows * tile_size, cols * tile_size, 3), 255, dtype=np.uint8)

    for idx, crop in enumerate(crops):
        r, c = idx // cols, idx % cols
        y_off, x_off = r * tile_size, c * tile_size

        pad = 8
        cell_w, cell_h = tile_size - pad * 2, tile_size - pad * 2
        ch, cw = crop.shape[:2]
        scale = min(cell_w / cw, cell_h / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        oy = y_off + pad + (cell_h - nh) // 2
        ox = x_off + pad + (cell_w - nw) // 2
        collage[oy:oy + nh, ox:ox + nw] = resized

        cv2.rectangle(collage, (x_off, y_off), (x_off + tile_size, y_off + tile_size), (210, 210, 210), 1)
        tag = f"#{idx}"
        cv2.rectangle(collage, (x_off + 4, y_off + 4), (x_off + 52, y_off + 28), (0, 0, 0), -1)
        cv2.putText(collage, tag, (x_off + 8, y_off + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return collage, rows, cols


# ==============================================================================
# 3. Gemini API 問い合わせ
# ==============================================================================
def query_gemini_attributes(collage_bgr: np.ndarray, num_items: int) -> Dict[int, Dict[str, str]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"❌ '{ENV_PATH}' または環境変数内に 'GEMINI_API_KEY' が見つかりませんでした。")
        return {}

    client = genai.Client(api_key=api_key)
    img_rgb = cv2.cvtColor(collage_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    prompt = (
        f"This composite image contains {num_items} tabletop object crops arranged in a grid (#0, #1, ...).\n"
        "Identify each item's canonical category name (e.g. 'pen', 'wooden block', 'eraser', 'mouse', 'ruler') "
        "and primary dominant color (e.g. 'red', 'blue', 'silver', 'black', 'natural wood').\n"
        "Return structured JSON matching the schema."
    )

    for model_name in CANDIDATE_MODELS:
        for attempt in range(2):
            try:
                print(f"⚡ [Gemini: {model_name}] へ推論リクエスト中... (試行 {attempt + 1}/2)")
                t0 = time.time()

                with progress_timer(f"Gemini ({model_name}) 応答待機中"):
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[pil_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=CollageClassificationResult,
                            temperature=0.1
                        ),
                    )
                print(f"✅ Gemini 推論完了 ({model_name}, 所要時間: {time.time() - t0:.2f} 秒)")
                data = json.loads(response.text)
                res = {}
                for item in data.get("items", []):
                    res[item["id"]] = {
                        "category": item.get("category", "unknown"),
                        "color": item.get("color", "unknown"),
                        "description": item.get("description", "")
                    }
                return res
            except Exception as e:
                err_msg = str(e)
                print(f"⚠️ {model_name} エラー: {err_msg}")
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break

    return {}


# ==============================================================================
# 4. Qwen2.5-VL ローカル問い合わせ (Ollama)
# ==============================================================================
def query_qwen_attributes(collage_bgr: np.ndarray, num_items: int) -> Dict[int, Dict[str, str]]:
    """コラージュ画像を Ollama (qwen2.5vl:3b) に投げ、JSON 形式で一括同定"""
    success, buffer = cv2.imencode(".jpg", collage_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return {}
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    prompt = (
        f"This composite image shows {num_items} tabletop object crops labeled #0 to #{num_items - 1}.\n"
        "Identify each item's canonical category name (e.g. 'pen', 'wooden block', 'eraser', 'mouse', 'ruler') "
        "and primary color (e.g. 'red', 'blue', 'silver', 'black', 'natural wood').\n"
        "Output ONLY a raw JSON array matching this exact format:\n"
        '[{"id": 0, "category": "pen", "color": "red"}, {"id": 1, "category": "mouse", "color": "black"}]\n'
        "Do not include any explanation or markdown tags."
    )

    request_payload = {
        "model": QWEN_MODEL_NAME,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512
        }
    }

    req_data = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    print(f"🦙 [Local Qwen: {QWEN_MODEL_NAME}] へ推論リクエスト中...")
    t0 = time.time()
    try:
        with progress_timer(f"Qwen 推論中"):
            with urllib.request.urlopen(req, timeout=90) as response:
                res_body = json.loads(response.read().decode("utf-8"))
            raw_text = res_body.get("response", "").strip()
            
            print(f"✅ Qwen 推論完了 (所要時間: {time.time() - t0:.2f} 秒)")

            # JSON 部分の抽出
            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                items_data = json.loads(json_match.group(0))
                res = {}
                for item in items_data:
                    res[int(item.get("id", 0))] = {
                        "category": item.get("category", "unknown"),
                        "color": item.get("color", "unknown"),
                        "description": ""
                    }
                return res
            else:
                print(f"⚠️ Qwen の出力から JSON 配列を抽出できませんでした: {raw_text[:100]}...")
                return {}
    except Exception as e:
        print(f"⚠️ Qwen 推論エラー: {e}")
        return {}

@contextmanager
def progress_timer(label: str = "推論処理中"):
    """API呼び出しや推論中に1秒間隔でコンソールに経過秒数を上書き表示する"""
    stop_event = threading.Event()
    start_time = time.time()

    def _worker():
        while not stop_event.wait(1.0):
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r   ⏳ {label}... 経過: {elapsed:.1f} 秒")
            sys.stdout.flush()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        sys.stdout.write("\r" + " " * 45 + "\r")  # プログレス表示行をクリア
        sys.stdout.flush()
# ==============================================================================
# 5. 統合認識ディスパッチャ (Gemini / Qwen / ハイブリッドフォールバック)
# ==============================================================================

def dispatch_classification(collage_bgr: np.ndarray, num_items: int, use_gemini: bool, use_qwen: bool) -> Dict[int, Dict[str, str]]:
    # 1. Gemini が有効な場合
    if use_gemini:
        attrs = query_gemini_attributes(collage_bgr, num_items)
        if attrs:
            return attrs
        if not use_qwen:
            print("❌ Gemini の推論に失敗しました (--qwen が指定されていないため中断)。")
            return {}
        print("🔄 Gemini の推論に失敗したため、ローカル Qwen にフォールバックします...")

    # 2. Qwen が有効な場合（直接指定またはフォールバック）
    if use_qwen:
        return query_qwen_attributes(collage_bgr, num_items)

    return {}


# ==============================================================================
# 6. OpenCV 物体・幾何特徴抽出
# ==============================================================================
def extract_objects_with_geometry(warped_img: np.ndarray, bg_gray: Optional[np.ndarray], valid_mask: np.ndarray, projector: VisionProjector):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if bg_gray is not None:
        diff = cv2.absdiff(blurred, bg_gray)
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 6
        )

    thresh = cv2.bitwise_and(thresh, thresh, mask=valid_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    h_img, w_img = warped_img.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w < h:
            w, h = h, w
            angle += 90.0

        while angle > 90.0:
            angle -= 180.0
        while angle <= -90.0:
            angle += 180.0

        box_pts = np.int32(cv2.boxPoints(rect))
        x_mm, y_mm = pixel_to_robot_phys_xy(cx, cy, projector)

        bx, by, bw, bh = cv2.boundingRect(cnt)
        pad = 10
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(w_img, bx + bw + pad), min(h_img, by + bh + pad)
        crop = warped_img[y1:y2, x1:x2]

        detected.append({
            "u": cx,
            "v": cy,
            "phys_xy": (round(x_mm, 1), round(y_mm, 1)),
            "size_mm": (round(w, 1), round(h, 1)),
            "angle_deg": round(angle, 1),
            "box_pts": box_pts,
            "crop": crop
        })

    return detected, thresh


def custom_sim_key_callback(keycode: int):
    global REQ_SAVE_BG, REQ_RECALIB, REQ_CLASSIFY, REQ_QUIT
    if keycode in (66, 98):  # B, b
        REQ_SAVE_BG = True
    elif keycode in (67, 99):  # C, c
        REQ_CLASSIFY = True
    elif keycode == 32:  # Space
        REQ_RECALIB = True
    elif keycode in (81, 113, 256):  # Q, q, ESC
        REQ_QUIT = True


def draw_aruco_markers_in_mujoco(sim, projector, marker_size_m: float = 0.04):
    if sim.viewer is None:
        return

    half_s = marker_size_m / 2.0
    half_th = 0.0002

    for marker_idx in range(4):
        phys_x, phys_y = projector.marker_phys_xy[marker_idx]
        mj_x = -phys_y / 1000.0
        mj_y = -phys_x / 1000.0
        mj_z = half_th

        # 白地プレート
        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([half_s, half_s, half_th], dtype=np.float64),
                pos=np.array([mj_x, mj_y, mj_z], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.9, 0.9, 0.9, 0.95], dtype=np.float32)
            )
            sim.viewer.user_scn.ngeom += 1

        # 中央黒正方形
        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([half_s * 0.6, half_s * 0.6, half_th * 1.5], dtype=np.float64),
                pos=np.array([mj_x, mj_y, mj_z + 0.0001], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32)
            )
            sim.viewer.user_scn.ngeom += 1

        # ID ラベル
        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            lbl_pos = np.array([mj_x, mj_y, mj_z + 0.025], dtype=np.float64)
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=np.zeros(3),
                pos=lbl_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.3, 0.8, 1.0, 1.0], dtype=np.float32)
            )
            sim.viewer.user_scn.geoms[ng].label = f"ArUco #{marker_idx}".encode("utf-8")
            sim.viewer.user_scn.ngeom += 1


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    global REQ_SAVE_BG, REQ_RECALIB, REQ_CLASSIFY, REQ_QUIT

    # 引数パース
    parser = argparse.ArgumentParser(description="Tachikoma Multimodal Digital Twin")
    parser.add_argument("--gemini", action="store_true", help="Gemini API による認識を有効化")
    parser.add_argument("--qwen", action="store_true", help="ローカル Qwen2.5-VL による認識を有効化")
    args = parser.parse_args()

    use_gemini = args.gemini
    use_qwen = args.qwen

    # 両方未指定の場合はハイブリッド（Gemini優先 ➔ Qwenフォールバック）をデフォルトとする
    if not use_gemini and not use_qwen:
        use_gemini = True
        use_qwen = True

    mode_desc = []
    if use_gemini and use_qwen:
        mode_desc.append("Gemini (Fallback to Qwen)")
    elif use_gemini:
        mode_desc.append("Gemini Only")
    elif use_qwen:
        mode_desc.append("Qwen Only (Offline)")

    print("==================================================")
    print(" 🌐 マルチモーダル・デジタルツイン同期システム")
    print(f" ⚙️ 認識エンジン: {mode_desc[0]}")
    print("==================================================")
    print("【操作】")
    print("  [C]     : 全物体を同定し MuJoCo に直方体形状＆ラベルを反映")
    print("  [B]     : 机面背景の記憶 (高精度差分)")
    print("  [SPACE] : マーカー正射影の再計算")
    print("  [Q/ESC] : 終了")
    print("--------------------------------------------------")

    # 1. MuJoCo ビューア初期化
    sim = MujocoSimViewer()
    home_rad = get_home_radians()
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass

    sim.viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=custom_sim_key_callback
    )

    # 各スロットのアドレスと geom_id を解決（新旧両方の命名規則に対応）
    slot_info = []
    for i in range(MAX_SLOTS):
        bname = f"obj_block_{i}"
        gname = f"geom_obj_{i}"
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, bname)
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, gname)

        if bid == -1:
            bname = f"jenga_block_{i}"
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid != -1:
                gid = sim.model.body_geomadr[bid]

        qpos_adr = None
        if bid != -1:
            jnt_adr = sim.model.body_jntadr[bid]
            if jnt_adr != -1:
                qpos_adr = sim.model.jnt_qposadr[jnt_adr]

        slot_info.append({"bid": bid, "gid": gid, "qpos_adr": qpos_adr})

    valid_count = sum(1 for s in slot_info if s["qpos_adr"] is not None and s["gid"] != -1)
    print(f"✅ MuJoCo 内に {valid_count} 個の物体スロットを検出・バインドしました。")
    if valid_count == 0:
        print("❌ 警告: 有効な物体スロットが MuJoCo モデル内に見つかりません！assets/so100_scene.xml を確認してください。")

    # 2. カメラ初期化
    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    valid_mask = create_marker_mask(CANVAS_SIZE, MARGIN)
    bg_gray = None

    object_profiles: Dict[int, dict] = {}
    cv2.namedWindow("Digital Twin Multi-modal Profiler")

    try:
        while sim.is_running() and not REQ_QUIT:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            if warped is None:
                cv2.imshow("Digital Twin Multi-modal Profiler", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                    break
                continue

            # キー要求の処理
            if REQ_SAVE_BG:
                cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)
                object_profiles.clear()
                REQ_SAVE_BG = False
                print("📸 背景画像を更新しました。")

            if REQ_RECALIB:
                projector.update_homography(frame)
                REQ_RECALIB = False
                print("🔄 キャリブレーションを更新しました。")

            detected_objs, _ = extract_objects_with_geometry(warped, bg_gray, valid_mask, projector)
            annotated = warped.copy()

            # [C] キーによる物体一括同定
            if REQ_CLASSIFY:
                REQ_CLASSIFY = False
                if detected_objs:
                    crops = [obj["crop"] for obj in detected_objs if obj["crop"].size > 0]
                    collage_img, _, _ = build_adaptive_collage(crops, tile_size=160)
                    ai_attrs = dispatch_classification(collage_img, len(crops), use_gemini, use_qwen)

                    object_profiles.clear()
                    print("\n📦 === 物体同定結果一覧 ===")
                    for idx, obj in enumerate(detected_objs):
                        attr = ai_attrs.get(idx, {"category": "object", "color": "", "description": ""})
                        display_name = f"{attr['color']} {attr['category']}".strip()
                        object_profiles[idx] = {
                            "category": attr["category"],
                            "color": attr["color"],
                            "display_name": display_name
                        }
                        print(f"  [#{idx}] {display_name} ({obj['size_mm'][0]}x{obj['size_mm'][1]}mm)")

            # 1. OpenCV 画面へのバウンディングボックス＆ラベル描画
            for i, obj in enumerate(detected_objs):
                prof = object_profiles.get(i, {})
                label_name = prof.get("display_name", f"#{i}")

                cv2.drawContours(annotated, [obj["box_pts"]], 0, (0, 255, 0), 2)
                u_pt, v_pt = int(obj["u"]), int(obj["v"])
                cv2.circle(annotated, (u_pt, v_pt), 4, (0, 0, 255), -1)

                tag = f"#{i}: {label_name}"
                cv2.putText(annotated, tag, (u_pt - 40, v_pt - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, tag, (u_pt - 40, v_pt - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # 2. MuJoCo 空間への動的反映
            if sim.viewer is not None:
                sim.viewer.user_scn.ngeom = 0

            draw_aruco_markers_in_mujoco(sim, projector, marker_size_m=0.04)

            for i in range(MAX_SLOTS):
                sinfo = slot_info[i]
                qadr = sinfo["qpos_adr"]
                gid = sinfo["gid"]
                if qadr is None or gid == -1:
                    continue

                if i < len(detected_objs):
                    obj = detected_objs[i]
                    x_mm, y_mm = obj["phys_xy"]
                    major_mm, minor_mm = obj["size_mm"]

                    mj_x = -y_mm / 1000.0
                    mj_y = -x_mm / 1000.0
                    mj_z = HALF_Z

                    yaw_rad = math.radians(-obj["angle_deg"])
                    quat = euler_yaw_to_quat(yaw_rad)

                    # 位置・姿勢
                    sim.data.qpos[qadr:qadr + 3] = [mj_x, mj_y, mj_z]
                    sim.data.qpos[qadr + 3:qadr + 7] = quat

                    # 直方体サイズ (half-size)
                    half_x = max(0.005, (major_mm / 1000.0) / 2.0)
                    half_y = max(0.005, (minor_mm / 1000.0) / 2.0)
                    sim.model.geom_size[gid] = [half_x, half_y, HALF_Z]

                    # 3Dラベル描画
                    prof = object_profiles.get(i, {})
                    label_name = prof.get("display_name", f"#{i}")

                    if sim.viewer is not None and sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
                        ngeom = sim.viewer.user_scn.ngeom
                        label_pos = np.array([mj_x, mj_y, mj_z + 0.03], dtype=np.float64)
                        mujoco.mjv_initGeom(
                            sim.viewer.user_scn.geoms[ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.zeros(3),
                            pos=label_pos,
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1.0, 1.0, 0.2, 1.0], dtype=np.float32)
                        )
                        sim.viewer.user_scn.geoms[ngeom].label = label_name.encode("utf-8")
                        sim.viewer.user_scn.ngeom += 1
                else:
                    # 机の下へ退避
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]
                    sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            mode_text = "Diff" if bg_gray is not None else "Adaptive"
            engine_text = "Gemini+Qwen" if (use_gemini and use_qwen) else ("Gemini" if use_gemini else "Qwen")
            cv2.putText(annotated, f"Detected: {len(detected_objs)} | Engine: {engine_text} | Press [C] to Identify",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Digital Twin Multi-modal Profiler", annotated)

            k = cv2.waitKey(1) & 0xFF
            if k in [ord('q'), ord('Q'), 27]:
                break
            elif k in [ord('b'), ord('B')]:
                REQ_SAVE_BG = True
            elif k in [ord('c'), ord('C')]:
                REQ_CLASSIFY = True
            elif k == 32:  # SPACE
                REQ_RECALIB = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()