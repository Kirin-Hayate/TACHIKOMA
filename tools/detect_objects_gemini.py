"""
==============================================================================
Gemini 2.5 Flash 物体検出・高速グラウンディングツール (tools/detect_objects_gemini.py)
==============================================================================
【役割】
1. 机面正射影画像 (topdown_warped.jpg またはカメラ映像) を取得。
2. Google GenAI SDK 経由で Gemini 2.5 Flash に画像を送信。
3. 厳密な JSON スキーマにより、机上の物体名と [ymin, xmin, ymax, xmax] (0〜1000) を即時抽出。
4. 正射影画像上にミリ精度でバウンディングボックスと中心点を重畳描画して確認。
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
from typing import List

import sys
import os
import time
import json
from dotenv import load_dotenv  # 👈 追加

# プロジェクトルートにある .env を明示的にロード
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)  # 👈 追加（.env を読み込んで os.environ に展開）

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Google GenAI SDK
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector

# Google GenAI SDK
from google import genai
from google.genai import types

# 入力画像パス設定
CAPTURE_DIR = os.path.join(BASE_DIR, "data", "camera_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)
WARPED_IMAGE_PATH = os.path.join(CAPTURE_DIR, "topdown_warped.jpg")


# ==============================================================================
# 1. 出力フォーマットのスキーマ定義 (Pydantic による完全型拘束)
# ==============================================================================
class BoundingBox2D(BaseModel):
    name: str = Field(description="物体の名称 (例: wooden block, jenga, scissors, pen, mouse)")
    box_2d: List[int] = Field(
        description="正規化バウンディングボックス座標 [ymin, xmin, ymax, xmax] (0〜1000の整数)"
    )

class TabletopDetections(BaseModel):
    objects: List[BoundingBox2D] = Field(description="机上で検出された把持可能物体のリスト")


# ==============================================================================
# 2. Gemini 2.5 Flash API への問い合わせ
# ==============================================================================
def query_gemini_vision(image_bgr: np.ndarray) -> List[dict]:
    """Gemini 3.6 Flash を呼び出して机上物体の BBox 一覧を取得"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"❌ '{ENV_PATH}' または環境変数内に 'GEMINI_API_KEY' が見つかりませんでした。")
        return []

    client = genai.Client(api_key=api_key)

    # OpenCV (BGR) から PIL Image (RGB) へ変換
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    prompt = (
        "You are an expert robotic perception system. "
        "Detect all graspable physical objects placed on this top-down table surface (e.g. wooden blocks, jenga, pens, scissors, mouse). "
        "Do NOT detect the four square ArUco markers at the four corners. "
        "Return the tight 2D bounding boxes using normalized coordinates [ymin, xmin, ymax, xmax] scaled to [0, 1000]."
    )

    print("⚡ gemini-3.6-flash へ推論リクエスト中...")
    start_time = time.time()

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
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
# 3. 検出結果の可視化と中心座標の計算
# ==============================================================================
def draw_detections(image_bgr: np.ndarray, detected_objects: List[dict]) -> np.ndarray:
    """正射影画像上に BBox と中心ピクセル座標を描画"""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    print("\n🔍 === 検出された物体一覧 ===")
    for i, obj in enumerate(detected_objects):
        name = obj.get("name", f"object_{i}")
        box = obj.get("box_2d", [])
        if len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box

        # 0〜1000 の正規化座標を実ピクセル座標に変換
        px_ymin = int((ymin / 1000.0) * h)
        px_xmin = int((xmin / 1000.0) * w)
        px_ymax = int((ymax / 1000.0) * h)
        px_xmax = int((xmax / 1000.0) * w)

        # 中心ピクセル (u, v)
        center_u = int((px_xmin + px_xmax) / 2)
        center_v = int((px_ymin + px_ymax) / 2)

        print(f"  [{i+1}] {name:<14} | BBox: ({px_xmin}, {px_ymin}) -> ({px_xmax}, {px_ymax}) | 中心: ({center_u}, {center_v}) px")

        # バウンディングボックス (緑色)
        cv2.rectangle(annotated, (px_xmin, px_ymin), (px_xmax, px_ymax), (0, 255, 0), 2)

        # 把持目標中心点 (赤丸)
        cv2.circle(annotated, (center_u, center_v), 5, (0, 0, 255), -1)

        # ラベル描画（黒アウトライン＋黄色前景文字）
        label = f"{name}"
        cv2.putText(annotated, label, (px_xmin, max(20, px_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, (px_xmin, max(20, px_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    print("==================================================")
    print(" 🚀 Gemini 2.5 Flash 机上物体高速グラウンディング")
    print("==================================================")

    target_img = None

    # 保存済みの正射影画像が存在すれば優先使用
    if os.path.exists(WARPED_IMAGE_PATH):
        print(f"📁 保存済みの正射影画像をロード: {WARPED_IMAGE_PATH}")
        target_img = cv2.imread(WARPED_IMAGE_PATH)

    # 保存画像がない場合はカメラから動的生成
    if target_img is None:
        print("📷 カメラから正射影画像を生成中...")
        projector = VisionProjector()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        for _ in range(30):
            ret, frame = cap.read()
            if ret and projector.update_homography(frame):
                target_img = projector.warp_to_topdown(frame, out_w=500, out_h=500)
                break
            time.sleep(0.05)

        cap.release()

        if target_img is not None:
            cv2.imwrite(WARPED_IMAGE_PATH, target_img)
            print(f"💾 正射影画像を保存しました: {WARPED_IMAGE_PATH}")
        else:
            print("❌ 正射影画像の生成に失敗しました（4隅マーカーを認識できませんでした）。")
            return

    # Gemini 2.5 Flash による推論
    detected_objects = query_gemini_vision(target_img)
    if not detected_objects:
        print("⚠️ 物体が検出されませんでした。")
        return

    # 結果の可視化
    result_img = draw_detections(target_img, detected_objects)

    # 画像保存
    out_path = os.path.join(CAPTURE_DIR, "gemini_detected_result.jpg")
    cv2.imwrite(out_path, result_img)
    print(f"\n💾 結果画像を保存しました: {out_path}")

    # ウィンドウ表示
    cv2.imshow("Gemini 2.5 Flash Object Detection", result_img)
    print("👉 ウィンドウをクリックして任意のキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()