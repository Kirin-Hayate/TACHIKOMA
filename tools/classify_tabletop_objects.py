"""
==============================================================================
ハイブリッド物体同定ツール (tools/classify_tabletop_objects.py)
==============================================================================
【役割】
1. OpenCV の背景差分・輪郭抽出により、30fps で机上物体の位置・向きを即時特定。
2. 各物体領域のサムネイル画像を自動クロップ。
3. [C] キー入力時に Gemini (gemini-3.6-flash) を呼び出し、各サムネイルが何であるかを分類同定。
4. 物体ラベル（例: wooden block, mouse, pen など）を画面上に維持・表示。

【操作】
  - [C]     : 現在検出されている全物体を Gemini に投げて分類・同定
  - [B]     : 現在の机面を背景として記憶（背景差分更新）
  - [SPACE] : 4隅マーカーから正射影を再計算
  - [Q/ESC] : 終了
==============================================================================
"""

import sys
import os
import time
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

# キャンバス設定
CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400

MIN_AREA_PX = 600
MAX_AREA_PX = 25000

MODEL_ID = "gemini-3.6-flash"


# ==============================================================================
# 1. Pydantic スキーマ定義
# ==============================================================================
class ObjectLabel(BaseModel):
    label: str = Field(description="物体の簡潔な英語名称 (例: wooden block, mouse, scissors, pen, stapler)")

class BatchClassification(BaseModel):
    items: List[ObjectLabel] = Field(description="各切り出し画像に対応するラベルのリスト")


# ==============================================================================
# 2. Gemini API 問い合わせ (クロップ画像の一括分類)
# ==============================================================================
def classify_crops_with_gemini(crop_images: List[np.ndarray]) -> List[str]:
    """切り出された複数の物体画像を Gemini に一括送信して分類結果を取得"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ 'GEMINI_API_KEY' が設定されていません。")
        return ["unknown"] * len(crop_images)

    client = genai.Client(api_key=api_key)

    # クロップ画像を PIL Image のリストに変換
    contents = []
    prompt = (
        f"Here are {len(crop_images)} cropped images of small tabletop items.\n"
        "Identify each item accurately with a concise name (e.g., 'wooden block', 'computer mouse', 'pen', 'scissors').\n"
        "Return the classification list corresponding to the images in order."
    )
    contents.append(prompt)

    for img in crop_images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        contents.append(Image.fromarray(rgb))

    print(f"⚡ {len(crop_images)} 個の物体サムネイルを Gemini に送信中...")
    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchClassification,
                temperature=0.1
            ),
        )
        elapsed = time.time() - t0
        print(f"✅ 分類完了 (所要時間: {elapsed:.2f} 秒)")

        import json
        data = json.loads(response.text)
        labels = [item.get("label", "unknown") for item in data.get("items", [])]

        # 返答数整合
        while len(labels) < len(crop_images):
            labels.append("unknown")
        return labels[:len(crop_images)]

    except Exception as e:
        print(f"❌ Gemini 分類エラー: {e}")
        return ["unknown"] * len(crop_images)


# ==============================================================================
# 3. OpenCV 物体検出 ＆ クロップ画像生成
# ==============================================================================
def create_marker_mask(size: int = CANVAS_SIZE, margin: int = MARGIN) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pad = margin + 15
    mask[pad:size - pad, pad:size - pad] = 255
    return mask


def extract_objects_and_crops(warped_img: np.ndarray, bg_gray: Optional[np.ndarray], valid_mask: np.ndarray):
    """輪郭検出および各物体のバウンディングボックス／クロップ画像を抽出"""
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

        # クロップ用外接矩形 (AABB) + パディング
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
            "crop": crop,
            "aabb": (x1, y1, x2, y2)
        })

    return detected, thresh


def main():
    print("==================================================")
    print(" 🏷️ ハイブリッド物体同定ツール (OpenCV + Gemini)")
    print("==================================================")
    print("【操作】")
    print("  [C]     : 現在検出されている物体を Gemini で分類・同定")
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

    # 物体ラベルのキャッシュ辞書 (位置近傍追従)
    cached_labels = {}  # slot_idx -> str

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

            # 物体描画
            for i, obj in enumerate(detected_objs):
                label = cached_labels.get(i, f"object_{i}")

                # 回転矩形
                cv2.drawContours(annotated, [obj["box_pts"]], 0, (0, 255, 0), 2)
                cv2.circle(annotated, (int(obj["u"]), int(obj["v"])), 4, (0, 0, 255), -1)

                # ラベル描画
                tag = f"#{i}: {label}"
                cv2.putText(annotated, tag, (int(obj["u"]) - 40, int(obj["v"]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, tag, (int(obj["u"]) - 40, int(obj["v"]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # 操作ガイド
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
                labels = classify_crops_with_gemini(crops)

                print("\n🏷️ === 物体分類結果 ===")
                for idx, lbl in enumerate(labels):
                    cached_labels[idx] = lbl
                    print(f"  Item #{idx}: {lbl}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()