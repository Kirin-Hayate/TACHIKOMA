"""
==============================================================================
机面正射影 (トップダウン) リアルタイムプレビュー (tools/test_ground_projection.py)
==============================================================================
【操作方法】
  - [Q] / [ESC] : 終了
  - [SPACE]     : キャリブレーション (ホモグラフィ行列) の強制再計算
  - [S]         : 正射影画像をファイル保存

【引数での座標オーバーライド例】
  python tools/test_ground_projection.py --id0 60,-20 --id1 60,20
==============================================================================
"""

import sys
import os
import argparse
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector

OUTPUT_DIR = "data/camera_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="机面正射影リアルタイムプレビュー")
    parser.add_argument("--cam", type=int, default=0, help="カメラインデックス (デフォルト: 0)")
    parser.add_argument("--id0", type=str, help="ID0の極座標 r_cm,theta_deg (例: 60,-20)")
    parser.add_argument("--id1", type=str, help="ID1の極座標 r_cm,theta_deg (例: 60,20)")
    parser.add_argument("--id2", type=str, help="ID2の極座標 r_cm,theta_deg (例: 27,-60)")
    parser.add_argument("--id3", type=str, help="ID3の極座標 r_cm,theta_deg (例: 26,62)")
    return parser.parse_args()


def main():
    args = parse_args()
    projector = VisionProjector()

    # コマンドライン引数による動的座標上書き
    for mid, arg_val in [(0, args.id0), (1, args.id1), (2, args.id2), (3, args.id3)]:
        if arg_val:
            r_str, th_str = arg_val.split(",")
            projector.set_marker_polar(mid, float(r_str), float(th_str))
            print(f"🔧 ID {mid} の座標を上書き設定: r={float(r_str):.1f}cm, θ={float(th_str):+.1f}°")

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ カメラ (Index: {args.cam}) を開けませんでした。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("==================================================")
    print(" 📷 正射影ワーピングテスト起動")
    print("   [SPACE] : ホモグラフィ変換を再計算")
    print("   [S]     : 現在の正射影画像を保存")
    print("   [Q]     : 終了")
    print("==================================================")

    calibrated = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 初回または未キャリブレーション時に自動で4マーカーを探して更新
        if not calibrated:
            if projector.update_homography(frame):
                calibrated = True
                print("🎯 全4マーカーを検出。正射影ワーピングを開始します。")

        # 真上からの正射影画像を生成 (500x500)
        warped = projector.warp_to_topdown(frame, out_w=500, out_h=500)

        # 生画像側に検出マーカー情報を重畳
        display_raw = frame.copy()
        centers = projector.detect_markers(frame)
        for mid, pt in centers.items():
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(display_raw, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(display_raw, f"ID{mid}", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 画面表示
        cv2.imshow("Original Camera Feed", cv2.resize(display_raw, (640, 360)))
        if warped is not None:
            # 正射影画像上にグリッド線を描画（直交性の確認用）
            grid_img = warped.copy()
            for x in range(50, 451, 100):
                cv2.line(grid_img, (x, 50), (x, 450), (255, 200, 0), 1)
            for y in range(50, 451, 100):
                cv2.line(grid_img, (50, y), (450, y), (255, 200, 0), 1)
            cv2.imshow("Top-Down Ortho View (Warped)", grid_img)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key == 32:  # SPACE
            if projector.update_homography(frame):
                print("🔄 ホモグラフィ行列を最新フレームで再計算しました。")
            else:
                print("⚠️ 4枚のマーカーが全て見えていません。")
        elif key in [ord('s'), ord('S')]:
            if warped is not None:
                save_p = os.path.join(OUTPUT_DIR, "topdown_warped.jpg")
                cv2.imwrite(save_p, warped)
                print(f"💾 正射影画像を保存しました: {save_p}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()