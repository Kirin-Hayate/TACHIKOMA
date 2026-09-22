"""
==============================================================================
OpenCV リアルタイム机上物体検出ツール (tools/detect_objects_opencv.py)
==============================================================================
【役割】
1. カメラからリアルタイムにフレームを取得し、正射影ワーピング。
2. 4隅の ArUco マーカー領域を自動マスク除外。
3. 机面と物体の色差（グレースケール差分・適応的二値化）から輪郭を抽出。
4. cv2.minAreaRect により、物体の【中心ピクセル (u, v)】および
   グリッパー把持に必要な【回転角 yaw (deg)】を 30fps で瞬時検出・描画。

【操作】
  - [B] : 現在の「何もない机面」を背景として記憶（背景差分モード更新）
  - [SPACE] : ホモグラフィ行列（4隅マーカー）の再計算
  - [Q] / [ESC] : 終了
==============================================================================
"""

import sys
import os
import time
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector

# 出力正射影サイズ
CANVAS_SIZE = 500
MARGIN = 50

# 検出対象の面積フィルタ (ジェンガの画素サイズ目安)
# 500x500px において、400mm 四方が内側 400px なので 1px ≈ 1mm。
# ジェンガ (約 75mm x 25mm = 1875 mm^2) -> 800〜6000 px^2 を対象
MIN_AREA_PX = 600
MAX_AREA_PX = 15000


def create_marker_mask(size: int = CANVAS_SIZE, margin: int = MARGIN) -> np.ndarray:
    """4隅の ArUco マーカー（外周マージン領域）を除外するための有効領域マスクを生成"""
    mask = np.zeros((size, size), dtype=np.uint8)
    # マーカーが存在する四隅・外周を除き、机の中央作業領域のみを白(255)にする
    pad = margin + 15
    mask[pad:size - pad, pad:size - pad] = 255
    return mask


def extract_oriented_objects(warped_img: np.ndarray, bg_gray: Optional[np.ndarray], valid_mask: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """
    画像処理パイプライン: 二値化 ➔ 輪郭抽出 ➔ 最小外接矩形 (minAreaRect)
    """
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if bg_gray is not None:
        # 背景差分モード
        diff = cv2.absdiff(blurred, bg_gray)
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    else:
        # Otsu 二値化 + 適応的閾値
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 6
        )

    # 4隅マーカー領域を除外
    thresh = cv2.bitwise_and(thresh, thresh, mask=valid_mask)

    # ノイズ除去 (オープニング ＆ クロージング)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    annotated = warped_img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue

        # 回転を考慮した最小外接矩形 (center(x,y), size(w,h), angle)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        # 常に長辺を主軸として角度を正規化 (-90度〜+90度)
        if w < h:
            w, h = h, w
            angle += 90.0

        # アームの水平座標系に合わせた角度整形
        while angle > 90.0:
            angle -= 180.0
        while angle <= -90.0:
            angle += 180.0

        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)

        cx_int, cy_int = int(round(cx)), int(round(cy))

        detected.append({
            "center": (cx_int, cy_int),
            "size": (round(w, 1), round(h, 1)),
            "angle_deg": round(angle, 1),
            "box_pts": box_pts
        })

        # --- 描画 ---
        # 1. 回転矩形 (緑)
        cv2.drawContours(annotated, [box_pts], 0, (0, 255, 0), 2)

        # 2. 中心点 (赤丸)
        cv2.circle(annotated, (cx_int, cy_int), 5, (0, 0, 255), -1)

        # 3. 把持姿勢の主軸ベクトル (青矢印)
        rad = math.radians(angle)
        arrow_len = int(w / 2.0)
        ax = int(cx + arrow_len * math.cos(rad))
        ay = int(cy + arrow_len * math.sin(rad))
        cv2.arrowedLine(annotated, (cx_int, cy_int), (ax, ay), (255, 100, 0), 2, tipLength=0.3)

        # 4. 座標・角度テキスト
        text = f"({cx_int},{cy_int}) {angle:+.1f}deg"
        cv2.putText(annotated, text, (cx_int - 40, cy_int - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (cx_int - 40, cy_int - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated, detected, thresh


def main():
    print("==================================================")
    print(" 👁️ OpenCV リアルタイム机上物体トラッカー (30fps)")
    print("==================================================")
    print("【キーボード操作】")
    print("  [B]     : 現在の机面を背景画像として登録 (高精度差分モード)")
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

    cv2.namedWindow("Real-time Object Tracker")
    cv2.namedWindow("Mask Threshold")

    fps_t = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 初回または未確定時はマーカー検出してワーピング行列を固定
            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            if warped is None:
                # マーカー未認識時は待機
                cv2.imshow("Real-time Object Tracker", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                    break
                continue

            # 物体検出処理
            annotated, objects, thresh = extract_oriented_objects(warped, bg_gray, valid_mask)

            # FPS 計測
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, now - fps_t))
            fps_t = now

            mode_str = "Background Diff" if bg_gray is not None else "Adaptive Thresh"
            cv2.putText(annotated, f"FPS: {fps:.1f} | Mode: {mode_str} | Found: {len(objects)}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Real-time Object Tracker", annotated)
            cv2.imshow("Mask Threshold", thresh)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('b'), ord('B')]:
                # 背景記憶
                cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)
                print("📸 現在の机面を背景として記憶しました。")
            elif key == 32:  # SPACE
                if projector.update_homography(frame):
                    print("🔄 正射影キャリブレーションを更新しました。")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()