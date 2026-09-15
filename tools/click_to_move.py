"""
==============================================================================
クリック・トゥ・ムーブ 実機テストツール (tools/click_to_move.py)
==============================================================================
【役割】
正射影ウィンドウ (Top-Down View) 上をクリックした地点のピクセル座標から
アームの物理直交座標 (X, Y) および極座標 (r, theta) を逆算し、
モデルBのたわみ補正と動的IKを用いてアーム先端をその位置へ直接移動させます。

【操作方法】
  - 正射影画面上で「左クリック」 : その地点へアームを移動
  - [H] キー                     : ホーム姿勢へ退避
  - [SPACE] キー                 : マーカー位置の再認識・キャリブレーション更新
  - [Q] / [ESC] キー             : 終了
==============================================================================
"""

import sys
import os
import time
import math
import cv2
import numpy as np
import json
import math
from typing import Dict, Tuple, Optional  

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import (
    FOLLOWER_PORT,
    BAUDRATE,
    SERVO_IDS,
    JOINT_CONFIG
)
from core.sts3215 import STS3215Driver
from core.sim_viewer import MujocoSimViewer
from core.kinematics import (
    get_home_radians,
    solve_ik_adaptive_approach,
    raw_to_radian,
    radian_to_raw,
    GRIPPER_CLOSE_RAD
)
from core.vision_projector import VisionProjector

# 正射影キャンバスのサイズ設定 (500x500px, 内側 50〜450px が 400mm 正方形)
CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400  # 450 - 50
PHYS_SPAN_MM = 400.0 # マーカーが作る正方形の1辺 (400mm)

# 目標接地高さ [mm] (机面から 5mm 浮かせた安全高さ)
TARGET_Z_MM = 5.0

# 移動時間 [秒]
MOVE_DURATION = 3.0


def smooth_move_rad(follower, sim, target_rad, current_rad, duration=1.5, steps=45):
    """コサイン S 字加減速による安全補間移動"""
    interval = duration / steps
    for step in range(1, steps + 1):
        ratio = (1.0 - math.cos(step / steps * math.pi)) / 2.0
        step_rad = {
            sid: current_rad[sid] + ratio * (target_rad[sid] - current_rad[sid])
            for sid in SERVO_IDS
        }
        if follower is not None:
            for sid in SERVO_IDS:
                follower.write_position(sid, radian_to_raw(sid, step_rad[sid]))
        if sim is not None:
            sim.update_joints_rad(step_rad)
        time.sleep(interval)
    current_rad.update(target_rad)


def pixel_to_robot_polar(u: float, v: float, projector: VisionProjector) -> Tuple[float, float, float, float]:
    """
    正射影画面上のピクセル (u, v) を、アーム基準の直交座標 (X, Y)[mm] および極座標 (r[cm], theta[deg]) に変換
    
    正射影キャンバスの配置:
      (50, 50)   : ID 0 (奥・左)
      (450, 50)  : ID 1 (奥・右)
      (50, 450)  : ID 2 (手前・左)
      (450, 450) : ID 3 (手前・右)
    """
    # 4マーカーの既知の物理座標 (mm) を取得
    p0 = projector.marker_phys_xy[0]  # [X, Y]
    p1 = projector.marker_phys_xy[1]
    p2 = projector.marker_phys_xy[2]
    p3 = projector.marker_phys_xy[3]

    # 正規化比率 (0.0 〜 1.0)
    # s: 横方向 (左:0.0 -> 右:1.0)
    # t: 縦方向 (奥:0.0 -> 手前:1.0)
    s = (u - MARGIN) / INNER_SPAN_PX
    t = (v - MARGIN) / INNER_SPAN_PX

    # 双線形補間 (Bilinear Interpolation) でアーム基準 (X, Y) mm を算出
    top = (1.0 - s) * p0 + s * p1
    bottom = (1.0 - s) * p2 + s * p3
    phys_xy = (1.0 - t) * top + t * bottom

    x_mm = float(phys_xy[0])
    y_mm = float(phys_xy[1])

    # アーム極座標へ変換
    r_mm = math.sqrt(x_mm**2 + y_mm**2)
    th_rad = math.atan2(y_mm, x_mm)
    th_deg = math.degrees(th_rad)
    r_cm = r_mm / 10.0

    return x_mm, y_mm, r_cm, th_deg


# クリックイベント管理用グローバル変数
clicked_pt = None

def on_mouse_click(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = (x, y)


def main():
    global clicked_pt
    print("==================================================")
    print(" 🎯 TACHIKOMA Click-to-Move 実機検証ツール")
    print("==================================================")

    # 1. ハードウェア & シミュレータ初期化
    follower = None
    try:
        follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
        print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
    except Exception as e:
        print(f"⚠️ 実機接続なし (シミュレーションのみ): {e}")

    sim = None
    try:
        sim = MujocoSimViewer()
        print("✅ 3Dシミュレータ初期化完了")
    except Exception as e:
        print(f"⚠️ 3Dシミュレータ初期化失敗: {e}")

    home_rad = get_home_radians()
    current_rad = dict(home_rad)

    if follower is not None:
        for sid in SERVO_IDS:
            pos = follower.read_position(sid)
            current_rad[sid] = raw_to_radian(sid, pos) if pos is not None else home_rad[sid]
            follower.write_position(sid, radian_to_raw(sid, current_rad[sid]))
            follower.set_torque(sid, True)
        print("✅ 全サーボのトルクを ON にしました。")

    if sim is not None:
        sim.update_joints_rad(current_rad)

    # 2. カメラ & プロジェクター初期化
    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 初期キャリブレーション
    print("⏳ カメラからマーカーを検出中...")
    for _ in range(30):
        ret, frame = cap.read()
        if ret and projector.update_homography(frame):
            break
        time.sleep(0.05)

    if projector.homography_mat is not None:
        print("✅ 4隅マーカーを検出し、ホモグラフィ行列を初期化しました。")
    else:
        print("⚠️ 4枚のマーカーが揃っていません。起動後に[SPACE]キーで再認識してください。")

    # Home姿勢へ移動
    print("🏠 Home 姿勢へ移動します...")
    smooth_move_rad(follower, sim, home_rad, current_rad, duration=2.0)

    cv2.namedWindow("Top-Down Ortho View (Click to Move)")
    cv2.setMouseCallback("Top-Down Ortho View (Click to Move)", on_mouse_click)

    last_target_pt = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 正射影画像の生成
            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            display_warped = warped.copy() if warped is not None else np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

            # ガイドグリッド描画
            for p in range(MARGIN, CANVAS_SIZE - MARGIN + 1, 100):
                cv2.line(display_warped, (p, MARGIN), (p, CANVAS_SIZE - MARGIN), (255, 200, 0), 1)
                cv2.line(display_warped, (MARGIN, p), (CANVAS_SIZE - MARGIN, p), (255, 200, 0), 1)

            # 前回クリック地点の描画
            if last_target_pt is not None:
                cv2.drawMarker(display_warped, last_target_pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # --- マウスクリック時のアーム移動処理 ---
            if clicked_pt is not None:
                ux, vy = clicked_pt
                clicked_pt = None
                last_target_pt = (ux, vy)

                x_mm, y_mm, r_cm, th_deg = pixel_to_robot_robot_polar = pixel_to_robot_polar(ux, vy, projector)
                print(f"\n🖱️ クリック座標検知: Pixel=({ux}, {vy})")
                print(f"   ➔ ロボット座標: X={x_mm:.1f}mm, Y={y_mm:.1f}mm")
                print(f"   ➔ アーム極座標: r={r_cm:.1f}cm, θ={th_deg:+.1f}°, 目標高さ z={TARGET_Z_MM:.1f}mm")

                # IK & たわみ補正（モデルB適用）
                tgt_rad, wp_rad, pitch = solve_ik_adaptive_approach(
                    r_tcp=(r_cm / 100.0),
                    theta_deg=th_deg,
                    z_tcp=(TARGET_Z_MM / 1000.0),
                    gripper_rad=GRIPPER_CLOSE_RAD,
                    enable_sag_compensation=True
                )

                if tgt_rad is None:
                    print("❌ [IK解なし] 指定位置は可動限界外または干渉リスクのため到達できません。")
                else:
                    print(f"🚀 アーム移動中 (進入角: {pitch:.0f}°)...")
                    # 1. 上空経由点へアプローチ
                    smooth_move_rad(follower, sim, wp_rad, current_rad, duration=1.2, steps=35)
                    # 2. 目標高さ（机上5mm）へ降下
                    smooth_move_rad(follower, sim, tgt_rad, current_rad, duration=0.8, steps=25)
                    print("✅ 到達完了。")

            cv2.imshow("Top-Down Ortho View (Click to Move)", display_warped)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == 32:  # SPACE
                if projector.update_homography(frame):
                    print("🔄 最新フレームでキャリブレーションを再計算しました。")
                else:
                    print("⚠️ 4枚のマーカーが全て認識できていません。")
            elif key in [ord('h'), ord('H')]:
                print("🏠 Home 姿勢へ退避します...")
                smooth_move_rad(follower, sim, home_rad, current_rad, duration=1.5)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n🏠 終了処理: Home 姿勢へ復帰...")
        try:
            smooth_move_rad(follower, sim, home_rad, current_rad, duration=1.5)
        except Exception:
            pass

        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
            print("✅ トルクを OFF にし、接続を閉じました。")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()