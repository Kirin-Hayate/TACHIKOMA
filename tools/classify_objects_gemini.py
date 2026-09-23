"""
==============================================================================
ハイブリッド物体同定ツール (tools/classify_tabletop_objects_gemini.py)
【グリッド連結コラージュ方式】
==============================================================================
【仕組み】
1. OpenCV の背景差分で机上の全物体を 30fps で追従し、各物体をクロップ。
2. [C] キー入力時:
   - 検出数 N に応じた最適な縦横比の白い台紙画像を動的生成。
   - 各サムネイルに識別番号 (#0, #1, ...) を印字して台紙に配置。
   - 生成された「1枚のコラージュ画像」のみを Gemini API へ 1 リクエストで送信。
3. レートリミット (RPM 制限) を回避しつつ、1秒程度で全物体の名称を特定。

【操作】
  - [C]     : 現在検出中の全物体を 1 枚にまとめて Gemini で分類
  - [B]     : 現在の机面を背景として記憶（背景差分更新）
  - [SPACE] : 4隅マーカーから正射影を再計算
  - [Q/ESC] : 終了
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

# プロジェクトルートと環境変数の設定
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
MIN_AREA_PX = 600
MAX_AREA_PX = 25000

MODEL_ID = "gemini-3.1-flash-lite"


# ==============================================================================
# 1. Pydantic 出力スキーマ定義 (番号と物体名のペアリスト)
# ==============================================================================
class IdentifiedItem(BaseModel):
    id: int = Field(description="画像内のタイル番号 (#0, #1 等の数値)")
    label: str = Field(description="物体の簡潔な名称 (例: wooden block, jenga, mouse, scissors, pen)")

class CollageClassificationResult(BaseModel):
    items: List[IdentifiedItem] = Field(description="コラージュ画像内の全アイテムの同定結果リスト")


# ==============================================================================
# 2. 動的コラージュ台紙の生成 (OpenCV)
# ==============================================================================
def build_adaptive_collage(crops: List[np.ndarray], tile_size: int = 150) -> Tuple[np.ndarray, int, int]:
    """
    検出数 N に応じて最適な行数・列数を計算し、
    各物体に '#0', '#1' と番号を焼き込んだ 1 枚の白い台紙画像を生成する。
    """
    n = len(crops)
    if n == 0:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8), 0, 0

    # 縦横比が正方形に近くなるよう行・列を算出 (例: 1個->1x1, 2個->1x2, 3〜4個->2x2, 5〜6個->2x3)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # 白い台紙を新規作成
    collage_h = rows * tile_size
    collage_w = cols * tile_size
    collage = np.full((collage_h, collage_w, 3), 255, dtype=np.uint8)

    for idx, crop in enumerate(crops):
        r = idx // cols
        c = idx % cols

        y_offset = r * tile_size
        x_offset = c * tile_size

        # 各タイルのセル領域 (マージンを持たせる)
        pad = 8
        cell_w = tile_size - pad * 2
        cell_h = tile_size - pad * 2

        # 元クロップのアスペクト比を維持してセルに収まるよう縮小
        ch, cw = crop.shape[:2]
        scale = min(cell_w / cw, cell_h / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        # セルの中央に配置
        oy = y_offset + pad + (cell_h - nh) // 2
        ox = x_offset + pad + (cell_w - nw) // 2
        collage[oy:oy + nh, ox:ox + nw] = resized

        # タイルの枠線を描画 (薄いグレー)
        cv2.rectangle(collage, (x_offset, y_offset), (x_offset + tile_size, y_offset + tile_size), (200, 200, 200), 1)

        # 各タイルの左上に識別番号をスタンプ印字 (#0, #1 ...)
        tag = f"#{idx}"
        cv2.rectangle(collage, (x_offset + 4, y_offset + 4), (x_offset + 48, y_offset + 28), (0, 0, 0), -1)
        cv2.putText(collage, tag, (x_offset + 8, y_offset + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return collage, rows, cols


# ==============================================================================
# 3. Gemini API 問い合わせ (1 リクエストで全物体を同定)
# ==============================================================================
# 試行するモデル候補（優先度順）
CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]

def query_gemini_collage(collage_bgr: np.ndarray, num_items: int) -> Dict[int, str]:
    """
    1枚のコラージュ画像を送信。
    503 が返ってきた場合は自動で指数待機リトライ & 予備モデルへ切り替える堅牢設計。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"❌ '{ENV_PATH}' または環境変数内に 'GEMINI_API_KEY' が見つかりませんでした。")
        return {}

    client = genai.Client(api_key=api_key)

    img_rgb = cv2.cvtColor(collage_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    prompt = (
        f"This single image contains {num_items} cropped tabletop objects arranged in a grid.\n"
        "Each object tile is clearly labeled with its index (#0, #1, #2, etc.) at the top-left corner.\n"
        "Please identify what each physical object is (e.g. 'wooden block', 'jenga block', 'mouse', 'pen', 'scissors', 'stapler').\n"
        "Return a JSON list containing the index ID and its concise name for all tiles."
    )

    # 複数モデルでリトライを試行
    for model_name in CANDIDATE_MODELS:
        for attempt in range(2):  # 各モデル最大2回試行
            try:
                print(f"⚡ [{model_name}] へ送信中... (試行 {attempt + 1}/2)")
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
                result_dict = {}
                for item in data.get("items", []):
                    result_dict[item["id"]] = item["label"]
                return result_dict

            except Exception as e:
                err_msg = str(e)
                print(f"⚠️ {model_name} エラー: {err_msg}")
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(1.5 * (attempt + 1))  # 少し待って再試行
                else:
                    break  # 404など回復不能なエラーは別モデルへ即スキップ

    print("❌ すべてのモデル候補で推論に失敗しました。")
    return {}


# ==============================================================================
# 4. OpenCV 机上物体輪郭・クロップ抽出
# ==============================================================================
def create_marker_mask(size: int = CANVAS_SIZE, margin: int = MARGIN) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pad = margin + 15
    mask[pad:size - pad, pad:size - pad] = 255
    return mask


def extract_objects_and_crops(warped_img: np.ndarray, bg_gray: Optional[np.ndarray], valid_mask: np.ndarray):
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

        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)

        # クロップ画像用の矩形 (AABB) + パディング
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
            "angle_deg": angle,
            "box_pts": box_pts,
            "crop": crop
        })

    return detected, thresh


# ==============================================================================
# メインループ
# ==============================================================================
def main():
    print("==================================================")
    print(" 🏷️ 動的コラージュ式 机上物体同定 (OpenCV + Gemini)")
    print("==================================================")
    print("【操作】")
    print("  [C]     : 全物体を 1 枚の台紙にまとめて Gemini で同定")
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

    # 物体ラベルキャッシュ (index -> label)
    cached_labels: Dict[int, str] = {}

    cv2.namedWindow("Tabletop Object Classifier")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            if warped is None:
                cv2.imshow("Tabletop Object Classifier", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                    break
                continue

            detected_objs, _ = extract_objects_and_crops(warped, bg_gray, valid_mask)
            annotated = warped.copy()

            # 検出結果の描画
            for i, obj in enumerate(detected_objs):
                label = cached_labels.get(i, f"item_{i}")

                # 回転矩形
                cv2.drawContours(annotated, [obj["box_pts"]], 0, (0, 255, 0), 2)
                cv2.circle(annotated, (int(obj["u"]), int(obj["v"])), 4, (0, 0, 255), -1)

                # ラベル表示
                tag = f"#{i}: {label}"
                cv2.putText(annotated, tag, (int(obj["u"]) - 40, int(obj["v"]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, tag, (int(obj["u"]) - 40, int(obj["v"]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            mode_text = "Diff" if bg_gray is not None else "Adaptive"
            cv2.putText(annotated, f"Detected: {len(detected_objs)} | Mode: {mode_text} | Press [C] to Classify",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Tabletop Object Classifier", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('b'), ord('B')]:
                cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)
                cached_labels.clear()
                print("📸 背景画像を更新しました。")
            elif key == 32:  # SPACE
                projector.update_homography(frame)
                print("🔄 キャリブレーションを更新しました。")
            elif key in [ord('c'), ord('C')]:
                if not detected_objs:
                    print("⚠️ 分類対象の物体が机上に見つかりません。")
                    continue

                crops = [obj["crop"] for obj in detected_objs if obj["crop"].size > 0]

                # 1. 検出数に合わせて動的コラージュ台紙を生成
                collage_img, rows, cols = build_adaptive_collage(crops, tile_size=160)
                cv2.imwrite(COLLAGE_SAVE_PATH, collage_img)
                print(f"\n🖼️ {len(crops)} 個の物体から {rows}x{cols} のコラージュ画像を生成しました: {COLLAGE_SAVE_PATH}")

                # 2. 1リクエストで Gemini に送信
                results = query_gemini_collage(collage_img, len(crops))

                # 3. 結果の反映
                cached_labels.clear()
                print("\n🏷️ === 物体同定結果一覧 ===")
                for idx in range(len(crops)):
                    name = results.get(idx, "unknown")
                    cached_labels[idx] = name
                    print(f"  [#{idx}] -> {name}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()