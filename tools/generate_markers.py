"""
==============================================================================
ラベル付き ArUco マーカー生成ツール (tools/generate_labeled_markers.py)
==============================================================================
【役割】
机面四隅用の ArUco マーカー (ID 0〜3) の下部に識別用テキスト「ID: X」と
切り取りガイド線を付与し、誤認防止と作業効率を高めた印刷用画像を生成します。
==============================================================================
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "assets/markers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4x4マス、50パターン辞書
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

MARKER_PX = 400
WHITE_MARGIN = 50  # マーカー周囲の白フチ (OpenCVの認識に必須)
LABEL_HEIGHT = 45  # 文字印字用スペース

def create_labeled_marker(marker_id: int) -> np.ndarray:
    # マーカー本体生成
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_PX)
    
    # 下部を広めに取って白フチをパディング
    padded = cv2.copyMakeBorder(
        marker, 
        top=WHITE_MARGIN, 
        bottom=WHITE_MARGIN + LABEL_HEIGHT, 
        left=WHITE_MARGIN, 
        right=WHITE_MARGIN, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255
    )
    
    # 描画用にBGR化
    bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    
    # テキスト描画 ("ID: X")
    text = f"ID: {marker_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (bgr.shape[1] - text_size[0]) // 2
    text_y = bgr.shape[0] - (LABEL_HEIGHT // 2)
    cv2.putText(bgr, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # ハサミで切り抜くための薄い外枠ガイド線
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), (210, 210, 210), 1)
    return bgr

labeled_markers = [create_labeled_marker(i) for i in range(4)]

# 個別ファイルの書き出し
for i, m in enumerate(labeled_markers):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"aruco_labeled_id{i}.png"), m)

# 2x2 まとめて1枚の印刷シート画像を作成
h, w, _ = labeled_markers[0].shape
gap = 30
sheet = np.ones((h * 2 + gap * 3, w * 2 + gap * 3, 3), dtype=np.uint8) * 255

for i in range(4):
    r_idx = i // 2
    c_idx = i % 2
    y0 = gap + r_idx * (h + gap)
    x0 = gap + c_idx * (w + gap)
    sheet[y0:y0 + h, x0:x0 + w] = labeled_markers[i]

sheet_path = os.path.join(OUTPUT_DIR, "aruco_labeled_sheet_4x4_0to3.png")
cv2.imwrite(sheet_path, sheet)
print(f"✅ ラベル付きマーカーの生成が完了しました: {sheet_path}")