"""
==============================================================================
Gemini 高速物体検出・グラウンディング実験ツール (tools/detect_objects_gemini.py)
==============================================================================
【役割】
1. プログラム起動時に毎回 Web カメラから最新フレームを自動取得。
2. 4隅の ArUco マーカーを認識して机面の真上正射影画像 (500x500) を生成。
3. Gemini Flash に画像を送信し、検出された把持対象物体の BBox [ymin, xmin, ymax, xmax] を取得。
4. 正射影画像上に BBox と把持中心点を重畳表示。
5. [SPACE] を押すたびに何度でも最新の配置を再撮影・再検出可能。
==============================================================================
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from typing import List , Optional
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

# 保存ディレクトリ
CAPTURE_DIR = os.path.join(BASE_DIR, "data", "camera_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)
WARPED_IMAGE_PATH = os.path.join(CAPTURE_DIR, "topdown_warped.jpg")
RESULT_IMAGE_PATH = os.path.join(CAPTURE_DIR, "gemini_detected_result.jpg")

# 使用モデル
MODEL_ID = "gemini-3.6-flash"


# ==============================================================================
# 1. Pydantic スキーマ定義
# ==============================================================================
class BoundingBox2D(BaseModel):
    name: str = Field(description="物体の名称 (例: wooden block, jenga, scissors, pen, mouse)")
    box_2d: List[int] = Field(
        description="正規化バウンディングボックス座標 [ymin, xmin, ymax, xmax] (0〜1000の整数)"
    )

class TabletopDetections(BaseModel):
    objects: List[BoundingBox2D] = Field(description="机上で検出された把持可能物体のリスト")


# ==============================================================================
# 2. Gemini API 問い合わせ
# ==============================================================================
def query_gemini_vision(image_bgr: np.ndarray) -> List[dict]:
    """Gemini を呼び出して机上物体の BBox 一覧を取得"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"❌ '{ENV_PATH}' または環境変数内に 'GEMINI_API_KEY' が見つかりませんでした。")
        return []

    client = genai.Client(api_key=api_key)

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    prompt = (
        "You are an expert robotic perception system. "
        "Detect all graspable physical objects placed on this top-down table surface (e.g. wooden blocks, jenga, pens, scissors, mouse). "
        "Do NOT detect the four square ArUco markers at the four corners. "
        "Return the tight 2D bounding boxes using normalized coordinates [ymin, xmin, ymax, xmax] scaled to [0, 1000]."
    )

    print(f"⚡ {MODEL_ID} へ推論リクエスト中...")
    start_time = time.time()

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TabletopDetections,
                temperature=0.1
            ),
        )
        elapsed = time.time() - start_time
        print(f"✅ 推論完了 (所要時間: {elapsed:.2f} 秒)")

        data = json.loads(response.text)
        return data.get("objects", [])

    except Exception as e:
        print(f"❌ Gemini API 呼び出しエラー: {e}")
        return []


# ==============================================================================
# 3. 検出結果の可視化
# ==============================================================================
def draw_detections(image_bgr: np.ndarray, detected_objects: List[dict]) -> np.ndarray:
    """正射影画像上に BBox と中心ピクセル座標を描画"""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    print("\n🔍 === 検出された物体一覧 ===")
    if not detected_objects:
        print("  (把持可能な物体は見つかりませんでした)")

    for i, obj in enumerate(detected_objects):
        name = obj.get("name", f"object_{i}")
        box = obj.get("box_2d", [])
        if len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box

        px_ymin = int((ymin / 1000.0) * h)
        px_xmin = int((xmin / 1000.0) * w)
        px_ymax = int((ymax / 1000.0) * h)
        px_xmax = int((xmax / 1000.0) * w)

        center_u = int((px_xmin + px_xmax) / 2)
        center_v = int((px_ymin + px_ymax) / 2)

        print(f"  [{i+1}] {name:<14} | BBox: ({px_xmin}, {px_ymin}) -> ({px_xmax}, {px_ymax}) | 中心: ({center_u}, {center_v}) px")

        cv2.rectangle(annotated, (px_xmin, px_ymin), (px_xmax, px_ymax), (0, 255, 0), 2)
        cv2.circle(annotated, (center_u, center_v), 5, (0, 0, 255), -1)

        label = f"{name}"
        cv2.putText(annotated, label, (px_xmin, max(20, px_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, (px_xmin, max(20, px_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated


def capture_topdown_frame(cap, projector, max_attempts=30) -> Optional[np.ndarray]:
    """カメラから最新フレームを読み取り、正射影画像を生成する"""
    print("📷 カメラから最新フレームを取得中...")
    warped = None
    for _ in range(max_attempts):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        if projector.update_homography(frame):
            warped = projector.warp_to_topdown(frame, out_w=500, out_h=500)
            break
        time.sleep(0.05)
    return warped


# ==============================================================================
# メインループ
# ==============================================================================
def main():
    print("==================================================")
    print(" 🚀 Gemini 机上物体高速グラウンディング実験")
    print("==================================================")
    print("【操作】")
    print("  - [SPACE] : 最新のカメラ映像で再検出")
    print("  - [Q] / [ESC] : 終了")
    print("--------------------------------------------------")

    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 初回検出の実行
    def run_detection_pipeline():
        target_img = capture_topdown_frame(cap, projector)
        if target_img is None:
            print("❌ 正射影画像の生成に失敗しました（4隅マーカーを認識できませんでした）。")
            return None

        cv2.imwrite(WARPED_IMAGE_PATH, target_img)

        detected = query_gemini_vision(target_img)
        result_img = draw_detections(target_img, detected)
        cv2.imwrite(RESULT_IMAGE_PATH, result_img)
        return result_img

    current_result = run_detection_pipeline()

    try:
        while True:
            if current_result is not None:
                cv2.imshow("Gemini Tabletop Object Detection", current_result)
            else:
                # 取得失敗時はブランク画面に案内
                blank = np.zeros((500, 500, 3), dtype=np.uint8)
                cv2.putText(blank, "Marker detection failed.", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(blank, "Check markers and press SPACE", (50, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Gemini Tabletop Object Detection", blank)

            key = cv2.waitKey(30) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == 32:  # SPACE
                print("\n🔄 新しい配置を再撮影・再検出します...")
                current_result = run_detection_pipeline()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()