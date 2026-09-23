"""
==============================================================================
ハイブリッド物体同定・詳細プロファイリングツール
(tools/classify_objects_gemini.py)
==============================================================================
【統合データモデル】
1. セマンティクス (Gemini):
   - category    : 一般カテゴリ (pen, wooden block, mouse, scissors 等)
   - color       : 主要色 (red, blue, natural wood, black 等)
   - description : 識別用短文 (red ballpoint pen with white cap 等)
2. 幾何計測 (OpenCV):
   - phys_pos_mm : 机面物理中心座標 [X_mm, Y_mm]
   - size_mm     : 実測長軸・短軸寸法 [長辺_mm, 短辺_mm]
   - angle_deg   : 把持角度 yaw (-90〜+90 deg)
==============================================================================
"""

import sys
import os
import time
import json
import math
import cv2
import numpy as np
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

# Google GenAI SDK
from google import genai
from google.genai import types

# 保存・キャンバス設定
CAPTURE_DIR = os.path.join(BASE_DIR, "data", "camera_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)
COLLAGE_SAVE_PATH = os.path.join(CAPTURE_DIR, "gemini_collage_input.jpg")

CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400

MIN_AREA_PX = 600
MAX_AREA_PX = 30000

CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


# ==============================================================================
# 1. Pydantic スキーマ定義（カテゴリ・色・詳細特徴）
# ==============================================================================
class IdentifiedItem(BaseModel):
    id: int = Field(description="画像内のタイル番号 (#0, #1 等の整数)")
    category: str = Field(description="物体の一般カテゴリ名 (例: pen, wooden block, eraser, mouse, scissors)")
    color: str = Field(description="物体の主要な色 (例: red, blue, natural wood, black, white, transparent)")
    description: str = Field(description="他の同一カテゴリ品と明確に区別できる特徴表現 (例: blue oil-based ballpoint pen with rubber grip)")

class CollageClassificationResult(BaseModel):
    items: List[IdentifiedItem] = Field(description="全タイルの同定結果リスト")


# ==============================================================================
# 2. 幾何計算ユーティリティ (OpenCV)
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


def build_adaptive_collage(crops: List[np.ndarray], tile_size: int = 160) -> Tuple[np.ndarray, int, int]:
    n = len(crops)
    if n == 0:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8), 0, 0

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    collage_h = rows * tile_size
    collage_w = cols * tile_size
    collage = np.full((collage_h, collage_w, 3), 255, dtype=np.uint8)

    for idx, crop in enumerate(crops):
        r = idx // cols
        c = idx % cols

        y_offset = r * tile_size
        x_offset = c * tile_size

        pad = 8
        cell_w = tile_size - pad * 2
        cell_h = tile_size - pad * 2

        ch, cw = crop.shape[:2]
        scale = min(cell_w / cw, cell_h / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        oy = y_offset + pad + (cell_h - nh) // 2
        ox = x_offset + pad + (cell_w - nw) // 2
        collage[oy:oy + nh, ox:ox + nw] = resized

        cv2.rectangle(collage, (x_offset, y_offset), (x_offset + tile_size, y_offset + tile_size), (210, 210, 210), 1)

        tag = f"#{idx}"
        cv2.rectangle(collage, (x_offset + 4, y_offset + 4), (x_offset + 52, y_offset + 28), (0, 0, 0), -1)
        cv2.putText(collage, tag, (x_offset + 8, y_offset + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return collage, rows, cols


# ==============================================================================
# 3. Gemini 呼び出し (耐障害リトライ + 構造化属性抽出)
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
        f"This composite image contains {num_items} tabletop object crops in a grid.\n"
        "Each tile has its label index (#0, #1, #2, ...) at the top-left.\n"
        "For each item, identify:\n"
        "1. category: short canonical noun (e.g. 'pen', 'wooden block', 'mouse', 'eraser', 'scissors')\n"
        "2. color: primary dominant color (e.g. 'red', 'dark blue', 'black', 'natural wood', 'white')\n"
        "3. description: a clear, distinctive description distinguishing it from other similar items.\n"
        "Return structured JSON matching the schema."
    )

    for model_name in CANDIDATE_MODELS:
        for attempt in range(2):
            try:
                print(f"⚡ [{model_name}] へ推論リクエスト中... (試行 {attempt + 1}/2)")
                t0 = time.time()
                response = client.models.generate_content(
                    model=model_name,
                    contents=[pil_image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CollageClassificationResult,
                        temperature=0.1
                    ),
                )
                elapsed = time.time() - t0
                print(f"✅ 推論完了 (使用モデル: {model_name}, 所要時間: {elapsed:.2f} 秒)")

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
# 4. 物体検出 & 物理計測パイプライン (OpenCV)
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

    # 正射影内側 400px が 400mm 相当のため 1px ≈ 1mm
    px_to_mm = 1.0

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

        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)

        # 物理中心座標の計算
        x_mm, y_mm = pixel_to_robot_phys_xy(cx, cy, projector)

        # 実測ミリ寸法 (長軸, 短軸)
        major_mm = round(w * px_to_mm, 1)
        minor_mm = round(h * px_to_mm, 1)

        bx, by, bw, bh = cv2.boundingRect(cnt)
        pad = 12
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w_img, bx + bw + pad)
        y2 = min(h_img, by + bh + pad)
        crop = warped_img[y1:y2, x1:x2]

        detected.append({
            "u": cx,
            "v": cy,
            "phys_xy": (round(x_mm, 1), round(y_mm, 1)),
            "size_mm": (major_mm, minor_mm),
            "angle_deg": round(angle, 1),
            "box_pts": box_pts,
            "crop": crop
        })

    return detected, thresh


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    print("==================================================")
    print(" 🏷️ 机上物体マルチモーダルプロファイリング")
    print("==================================================")
    print("【操作】")
    print("  [C]     : 全物体をコラージュ化して Gemini で属性抽出")
    print("  [B]     : 現在の机面を背景として記憶（背景差分更新）")
    print("  [SPACE] : 4隅マーカーから正射影を再計算")
    print("  [Q/ESC] : 終了")
    print("--------------------------------------------------")

    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    valid_mask = create_marker_mask(CANVAS_SIZE, MARGIN)
    bg_gray = None

    # 統合プロファイルキャッシュ (index -> dict)
    object_profiles: Dict[int, dict] = {}

    cv2.namedWindow("Multimodal Object Profiler")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            if warped is None:
                cv2.imshow("Multimodal Object Profiler", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                    break
                continue

            detected_objs, _ = extract_objects_with_geometry(warped, bg_gray, valid_mask, projector)
            annotated = warped.copy()

            # 画面への重畳描画
            for i, obj in enumerate(detected_objs):
                prof = object_profiles.get(i, {})
                category = prof.get("category", f"item_{i}")
                color = prof.get("color", "")

                cv2.drawContours(annotated, [obj["box_pts"]], 0, (0, 255, 0), 2)
                cv2.circle(annotated, (int(obj["u"]), int(obj["v"])), 4, (0, 0, 255), -1)

                # ラベルテキスト作成 (例: "#0: red pen (145x12mm)")
                tag = f"#{i}: {color + ' ' if color else ''}{category}"
                geom_info = f"{obj['size_mm'][0]:.0f}x{obj['size_mm'][1]:.0f}mm {obj['angle_deg']:+.0f}deg"

                u_pt, v_pt = int(obj["u"]), int(obj["v"])
                cv2.putText(annotated, tag, (u_pt - 45, v_pt - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, tag, (u_pt - 45, v_pt - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(annotated, geom_info, (u_pt - 45, v_pt - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, geom_info, (u_pt - 45, v_pt - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1, cv2.LINE_AA)

            mode_text = "Diff" if bg_gray is not None else "Adaptive"
            cv2.putText(annotated, f"Detected: {len(detected_objs)} | Mode: {mode_text} | Press [C] to Analyze",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Multimodal Object Profiler", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('b'), ord('B')]:
                cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)
                object_profiles.clear()
                print("📸 背景画像を更新しました。")
            elif key == 32:  # SPACE
                projector.update_homography(frame)
                print("🔄 キャリブレーションを更新しました。")
            elif key in [ord('c'), ord('C')]:
                if not detected_objs:
                    print("⚠️ 物体が検出されていません。")
                    continue

                crops = [obj["crop"] for obj in detected_objs if obj["crop"].size > 0]
                collage_img, rows, cols = build_adaptive_collage(crops, tile_size=160)
                cv2.imwrite(COLLAGE_SAVE_PATH, collage_img)
                print(f"\n🖼️ {len(crops)} 個の物体コラージュ画像を生成: {COLLAGE_SAVE_PATH}")

                # Gemini 問い合わせ (1リクエスト)
                ai_attrs = query_gemini_attributes(collage_bgr=collage_img, num_items=len(crops))

                # OpenCV 幾何データと Gemini 属性データを完全統合
                object_profiles.clear()
                print("\n📦 === 机上物体統合プロファイル ===")
                for idx, obj in enumerate(detected_objs):
                    attr = ai_attrs.get(idx, {"category": "unknown", "color": "unknown", "description": ""})
                    full_profile = {
                        "id": idx,
                        "category": attr["category"],
                        "color": attr["color"],
                        "description": attr["description"],
                        "phys_xy_mm": obj["phys_xy"],
                        "size_mm": obj["size_mm"],
                        "angle_deg": obj["angle_deg"]
                    }
                    object_profiles[idx] = full_profile

                    print(
                        f"  [#{idx}] {full_profile['color']} {full_profile['category']} "
                        f"| 寸法: {full_profile['size_mm'][0]}x{full_profile['size_mm'][1]} mm "
                        f"| 位置: X={full_profile['phys_xy_mm'][0]}, Y={full_profile['phys_xy_mm'][1]} mm "
                        f"| 角度: {full_profile['angle_deg']:+.1f}° "
                        f"| 特徴: {full_profile['description']}"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()