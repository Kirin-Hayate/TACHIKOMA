"""
==============================================================================
ハイブリッド物体同定ツール (tools/classify_tabletop_objects.py) - Qwen版
==============================================================================
【役割】
1. OpenCV の背景差分・輪郭抽出により、30fps で机上物体の位置・向き・輪郭を即時特定。
2. 各物体領域のサムネイル画像を自動クロップ。
3. [C] キー入力時にローカル Ollama (qwen2.5vl:3b) を呼び出し、
   「これは何（wooden block / mouse / pen 等）か？」を同定。
4. クロップ画像のみを投げるため、軽量・低遅延かつ完全オフラインで動作可能。

【操作】
  - [C]     : 現在検出されている物体を Qwen2.5-VL で分類・同定
  - [B]     : 現在の机面を背景として記憶（背景差分更新）
  - [SPACE] : 4隅マーカーから正射影を再計算
  - [Q/ESC] : 終了
==============================================================================
"""

import sys
import os
import time
import json
import base64
import re
import urllib.request
import urllib.error
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector

# Ollama 設定
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"

# キャンバス設定
CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400

MIN_AREA_PX = 600
MAX_AREA_PX = 25000


# ==============================================================================
# 1. 画像 Base64 エンコード
# ==============================================================================
def encode_crop_to_base64(crop_bgr: np.ndarray, target_size: int = 128) -> str:
    """切り出し画像を正方形パディング＆リサイズして Base64 化 (極小トークン化)"""
    h, w = crop_bgr.shape[:2]
    # アスペクト比を維持しつつ長辺を target_size に揃える
    scale = target_size / max(h, w)
    resized = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 黒パディングで正方形にする
    rh, rw = resized.shape[:2]
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - rh) // 2
    x_off = (target_size - rw) // 2
    padded[y_off:y_off + rh, x_off:x_off + rw] = resized

    success, buffer = cv2.imencode(".jpg", padded, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise ValueError("JPEG エンコードに失敗しました。")
    return base64.b64encode(buffer).decode("utf-8")


# ==============================================================================
# 2. Qwen2.5-VL 単一画像クラス分類
# ==============================================================================
def classify_single_crop_qwen(crop_bgr: np.ndarray, timeout_sec: float = 90.0) -> str:
    """小さなクロップ画像を Qwen2.5-VL に渡し、経過時間を表示しながら分類させる"""
    img_b64 = encode_crop_to_base64(crop_bgr, target_size=128)

    prompt_text = (
        "Identify this single tabletop object clearly in 1 to 3 words (e.g. 'wooden block', 'jenga block', 'mouse', 'pen', 'scissors', 'tape dispenser'). "
        "Output ONLY the object name. No explanations, no markdown, no quotes."
    )

    request_payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 16
        }
    }

    req_data = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    # 経過時間をコンソールにリアルタイム表示するウォッチャースレッド
    stop_event = threading.Event()
    start_time = time.time()

    def print_progress():
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r   ⏳ 推論中... 経過: {elapsed:.1f} 秒")
            sys.stdout.flush()
            time.sleep(1.0)

    progress_thread = threading.Thread(target=print_progress, daemon=True)
    progress_thread.start()

    try:
        # タイムアウトを 90 秒に延長
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            raw_text = res_body.get("response", "").strip()
            cleaned = re.sub(r'[\r\n"\'`.]', '', raw_text)
            return cleaned if cleaned else "unknown"
    except Exception as e:
        sys.stdout.write("\n")
        print(f"⚠️ Qwen 推論エラー: {e}")
        return "unknown"
    finally:
        stop_event.set()
        progress_thread.join()
        sys.stdout.write("\r" + " " * 40 + "\r")  # プログレス行をクリア
        sys.stdout.flush()


# ==============================================================================
# 3. OpenCV 検出パイプライン
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

        # クロップ画像用の矩形 (AABB)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        pad = 10
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
    print(f" 🏷️ ハイブリッド物体同定ツール (OpenCV + {MODEL_NAME})")
    print("==================================================")
    print("【操作】")
    print("  [C]     : 検出中の各物体を Qwen2.5-VL で個別分類")
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

    cached_labels = {}  # index -> str

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

                cv2.drawContours(annotated, [obj["box_pts"]], 0, (0, 255, 0), 2)
                cv2.circle(annotated, (int(obj["u"]), int(obj["v"])), 4, (0, 0, 255), -1)

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
                    print("⚠️ 分類対象の物体がありません。")
                    continue

                print(f"\n🧠 Qwen2.5-VL で {len(detected_objs)} 個の物体を順次分類中...")
                cached_labels.clear()

                for idx, obj in enumerate(detected_objs):
                    crop = obj["crop"]
                    if crop.size == 0:
                        continue
                    t0 = time.time()
                    name = classify_single_crop_qwen(crop)
                    elapsed = time.time() - t0
                    cached_labels[idx] = name
                    print(f"  Item #{idx} ({elapsed:.2f}s): {name}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()