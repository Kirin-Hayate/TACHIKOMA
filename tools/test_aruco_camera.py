"""
==============================================================================
ArUco マーカー認識リアルタイムテストツール (tools/test_aruco_camera.py)
==============================================================================
【役割】
C270 Webカメラの映像から ArUco マーカー (DICT_4X4_50) をリアルタイムに検出し、
- 検出されたマーカーの ID
- 4隅の輪郭線
- マーカーの中心ピクセル座標 (u, v)
を画面上にオーバーレイ描画して認識テストを行います。

【操作方法】
  - [Q] キー または [ESC] キー : 終了
  - [S] キー : 現在の検出結果フレームを画像として保存
  - [C] キー : カメラインデックスの切り替え (0 <-> 1)
==============================================================================
"""

import os
import cv2
import numpy as np

# カメラ解像度 (C270 の最大解像度 720p)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 辞書設定 (生成したマーカーと同じ 4x4, 50パターン)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# 保存先ディレクトリ
OUTPUT_DIR = "data/camera_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_aruco_detector():
    """OpenCVのバージョン差分を吸収して検出器を初期化"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    
    # 認識精度向上のためのパラメータ微調整
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # サブピクセル角精緻化
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    
    # OpenCV 4.7 以降の ArucoDetector クラス対応
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        return detector, None, None
    else:
        # 古い OpenCV 向けフォールバック
        return None, aruco_dict, parameters


def detect_markers(image, detector, aruco_dict, parameters):
    """画像からマーカーを検出"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids, rejected


def open_camera(cam_idx: int):
    """Windows向けに DirectShow バックエンドでカメラを開く"""
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # 通常オープンを試行
        cap = cv2.VideoCapture(cam_idx)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    print("==================================================")
    print(" 📷 ArUco マーカー認識リアルタイムテストツール")
    print("==================================================")
    print(" 操作ガイド:")
    print("   [Q] / [ESC] : 終了")
    print("   [S]         : 現在の画面をキャプチャ保存")
    print("   [C]         : カメラ切り替え (0 <-> 1)")
    print("==================================================")

    current_cam_idx = 0
    cap = open_camera(current_cam_idx)

    if not cap.isOpened():
        print(f"⚠️ カメラ (インデックス {current_cam_idx}) が開けませんでした。インデックス 1 を試します...")
        current_cam_idx = 1
        cap = open_camera(current_cam_idx)
        if not cap.isOpened():
            print("❌ 有効なカメラが見つかりませんでした。接続を確認してください。")
            return

    detector, aruco_dict, parameters = init_aruco_detector()
    shot_count = 0

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ カメラ起動成功 (Index: {current_cam_idx}, 解像度: {actual_w}x{actual_h})")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ フレームの取得に失敗しました。")
            break

        display_frame = frame.copy()

        # マーカー検出
        corners, ids, _ = detect_markers(frame, detector, aruco_dict, parameters)

        detected_ids = []
        if ids is not None and len(ids) > 0:
            # OpenCV標準の枠線・ID描画
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                detected_ids.append(marker_id)
                # 4隅の座標から中心座標を計算
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())

                # 中心に赤丸をプロット
                cv2.circle(display_frame, (center_x, center_y), 4, (0, 0, 255), -1)

                # 中心座標テキスト表示
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(
                    display_frame, coord_text, (center_x - 40, center_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA
                )

        # ステータス情報オーバーレイ
        status_text = f"Detected IDs: {sorted(detected_ids)} (Total: {len(detected_ids)}/4)"
        color = (0, 255, 0) if len(detected_ids) == 4 else (0, 165, 255)
        cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # 全4枚の認識状態ガイド
        target_ids = [0, 1, 2, 3]
        guide_text = " | ".join([f"ID{tid}: {'OK' if tid in detected_ids else '--'}" for tid in target_ids])
        cv2.putText(display_frame, guide_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ArUco Camera Test - [Q] Quit, [S] Save", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
            break
        elif key in [ord('s'), ord('S')]:
            shot_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"aruco_test_capture_{shot_count:02d}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"📸 生画像を保存しました: {save_path}")
        elif key in [ord('c'), ord('C')]:
            cap.release()
            current_cam_idx = 1 if current_cam_idx == 0 else 0
            print(f"🔄 カメラをインデックス {current_cam_idx} に切り替えます...")
            cap = open_camera(current_cam_idx)

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 終了しました。")


if __name__ == "__main__":
    main()