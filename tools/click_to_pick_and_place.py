"""
==============================================================================
ビジョンベース Pick & Place ツール (tools/click_to_pick_and_place.py)
==============================================================================
【役割】
1. 正射影ウィンドウ上で「Pick 地点」と「Place 地点」を順番にクリックする。
2. 自動で一連の動作シーケンス（上空アプローチ ➔ 15mm降下 ➔ グリッパー把持 
   ➔ 持ち上げ ➔ Place上空 ➔ 15mm降下 ➔ グリッパー開放 ➔ 退避）を生成。
3. まず PC 画面上の MuJoCo シミュレータでプレビュー再生する。
4. ユーザーがターミナルで承認したら、実機アームにその動作を反映させる。
==============================================================================
"""

import sys
import os
import time
import math
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import (
    FOLLOWER_PORT,
    BAUDRATE,
    SERVO_IDS,
)
from core.sts3215 import STS3215Driver
from core.sim_viewer import MujocoSimViewer
from core.kinematics import (
    get_home_radians,
    solve_ik_adaptive_approach,
    raw_to_radian,
    radian_to_raw,
    GRIPPER_CLOSE_RAD,
    GRIPPER_OPEN_RAD
)
from core.vision_projector import VisionProjector

# キャンバス設定
CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400

# Pick / Place における目標高さ [mm]（デフォルト15mm）
DEFAULT_Z_MM = 15.0
LIFT_Z_MM = 80.0  # 持ち上げ時・移動時の上空高さ [mm]


def smooth_move_rad(follower, sim, target_rad, current_rad, duration=1.2, steps=35):
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
            try:
                sim.update_joints_rad(step_rad)
            except Exception:
                pass
        time.sleep(interval)
    current_rad.update(target_rad)


def pixel_to_robot_polar(u: float, v: float, projector: VisionProjector) -> Tuple[float, float, float, float]:
    """正射影画面上のピクセル (u, v) をロボット直交座標 (X, Y)[mm] および極座標に変換"""
    p0 = projector.marker_phys_xy[0]
    p1 = projector.marker_phys_xy[1]
    p2 = projector.marker_phys_xy[2]
    p3 = projector.marker_phys_xy[3]

    s = (u - MARGIN) / INNER_SPAN_PX
    t = (v - MARGIN) / INNER_SPAN_PX

    top = (1.0 - s) * p0 + s * p1
    bottom = (1.0 - s) * p2 + s * p3
    phys_xy = (1.0 - t) * top + t * bottom

    x_mm = float(phys_xy[0])
    y_mm = float(phys_xy[1])

    r_mm = math.sqrt(x_mm**2 + y_mm**2)
    th_rad = math.atan2(y_mm, x_mm)
    th_deg = math.degrees(th_rad)
    r_cm = r_mm / 10.0

    return x_mm, y_mm, r_cm, th_deg


# クリック管理
click_buffer = []

def on_mouse_click(event, x, y, flags, param):
    global click_buffer
    if event == cv2.EVENT_LBUTTONDOWN:
        click_buffer.append((x, y))


def compute_target_poses(r_cm, th_deg, z_mm, gripper_rad):
    """指定された位置・高さ・グリッパー開閉のIKを計算して返す"""
    tgt_rad, wp_rad, pitch = solve_ik_adaptive_approach(
        r_tcp=(r_cm / 100.0),
        theta_deg=th_deg,
        z_tcp=(z_mm / 1000.0),
        gripper_rad=gripper_rad,
        enable_sag_compensation=True
    )
    return tgt_rad, wp_rad, pitch


def main():
    global click_buffer
    print("==================================================")
    print(" 🤖 ビジョンベース Pick & Place ツール")
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
        try:
            sim.update_joints_rad(current_rad)
        except Exception:
            pass

    # 2. カメラ & プロジェクター初期化
    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("⏳ カメラからマーカーを検出中...")
    for _ in range(30):
        ret, frame = cap.read()
        if ret and projector.update_homography(frame):
            break
        time.sleep(0.05)

    print("🏠 Home 姿勢へ移動します...")
    smooth_move_rad(follower, sim, home_rad, current_rad, duration=1.5)

    cv2.namedWindow("Pick & Place Planner")
    cv2.setMouseCallback("Pick & Place Planner", on_mouse_click)

    pick_pt = None
    place_pt = None

    print("\n👉 【操作手順】")
    print("   1. 正射影ウィンドウ上でまず **[Pick 地点]** をクリック")
    print("   2. 次に **[Place 地点]** をクリック")
    print("   3. シミュレーションプレビュー後、ターミナルで実行承認を行う")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            display_img = warped.copy() if warped is not None else np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

            # グリッド描画
            for p in range(MARGIN, CANVAS_SIZE - MARGIN + 1, 100):
                cv2.line(display_img, (p, MARGIN), (p, CANVAS_SIZE - MARGIN), (255, 200, 0), 1)
                cv2.line(display_img, (MARGIN, p), (CANVAS_SIZE - MARGIN, p), (255, 200, 0), 1)

            # Pick/Place 地点の描画
            if pick_pt is not None:
                cv2.drawMarker(display_img, pick_pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display_img, "PICK", (pick_pt[0]+10, pick_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if place_pt is not None:
                cv2.drawMarker(display_img, place_pt, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display_img, "PLACE", (place_pt[0]+10, place_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # ガイドメッセージ
            msg = "1. Click PICK point" if pick_pt is None else ("2. Click PLACE point" if place_pt is None else "Ready! Check Terminal.")
            cv2.putText(display_img, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Pick & Place Planner", display_img)

            # --- クリック処理 ---
            if len(click_buffer) > 0:
                pt = click_buffer.pop(0)
                if pick_pt is None:
                    pick_pt = pt
                    print(f"📍 Pick 地点決定: Pixel={pick_pt}")
                elif place_pt is None:
                    place_pt = pt
                    print(f"📍 Place 地点決定: Pixel={place_pt}")
                    time.sleep(0.5)
                    # --- 両方の地点が揃ったのでシーケンスを生成・プレビュー・実行 ---
                    print("\n--- 軌道生成・シミュレーション検証を開始 ---")
                    
                    # 座標変換
                    _, _, pick_r, pick_th = pixel_to_robot_polar(pick_pt[0], pick_pt[1], projector)
                    _, _, place_r, place_th = pixel_to_robot_polar(place_pt[0], place_pt[1], projector)

                    # 1. Pick姿勢の計算
                    _, pick_wp, _ = compute_target_poses(pick_r, pick_th, LIFT_Z_MM, GRIPPER_OPEN_RAD)
                    pick_target, _, _ = compute_target_poses(pick_r, pick_th, DEFAULT_Z_MM, GRIPPER_OPEN_RAD)
                    pick_grasp, _, _ = compute_target_poses(pick_r, pick_th, DEFAULT_Z_MM, GRIPPER_CLOSE_RAD)
                    _, pick_lift, _ = compute_target_poses(pick_r, pick_th, LIFT_Z_MM, GRIPPER_CLOSE_RAD)

                    # 2. Place姿勢の計算
                    _, place_wp, _ = compute_target_poses(place_r, place_th, LIFT_Z_MM, GRIPPER_CLOSE_RAD)
                    place_target, _, _ = compute_target_poses(place_r, place_th, DEFAULT_Z_MM, GRIPPER_CLOSE_RAD)
                    place_release, _, _ = compute_target_poses(place_r, place_th, DEFAULT_Z_MM, GRIPPER_OPEN_RAD)
                    _, place_retreat, _ = compute_target_poses(place_r, place_th, LIFT_Z_MM, GRIPPER_OPEN_RAD)

                    if any(p is None for p in [pick_target, place_target]):
                        print("❌ [IK解なし] 指定されたPickまたはPlace地点は可動範囲外です。リセットします。")
                        pick_pt, place_pt = None, None
                        continue

                    # シミュレーションでプレビュー再生
                    """
                    print("🖥️ MuJoCo シミュレータでプレビューを再生します...")
                    if sim is not None:
                        sim_rad = dict(current_rad)
                        try:
                            steps_sim = 20
                            for target in [pick_wp, pick_target, pick_grasp, pick_lift, place_wp, place_target, place_release, place_retreat]:
                                if target is None:
                                    continue
                                for step in range(1, steps_sim + 1):
                                    ratio = step / steps_sim
                                    temp_rad = {sid: sim_rad.get(sid, 0.0) + ratio * (target.get(sid, 0.0) - sim_rad.get(sid, 0.0)) for sid in SERVO_IDS}
                                    try:
                                        sim.update_joints_rad(temp_rad)
                                    except Exception:
                                        pass
                                    time.sleep(0.01)
                                sim_rad.update(target)
                        except Exception as e:
                            print(f"⚠️ シミュレーションプレビュー中エラー: {e}")
                    """

                    # ユーザー承認
                    ans = input("\n🤔 この動作を実機で実行しますか？ [y/N]: ").strip().lower()
                    if ans == 'y':
                        print("🚀 実機アームでの Pick & Place を実行します...")
                        try:
                            smooth_move_rad(follower, sim, pick_wp, current_rad, duration=1.2)
                            smooth_move_rad(follower, sim, pick_target, current_rad, duration=0.8)
                            smooth_move_rad(follower, sim, pick_grasp, current_rad, duration=0.5)
                            smooth_move_rad(follower, sim, pick_lift, current_rad, duration=1.0)
                            smooth_move_rad(follower, sim, place_wp, current_rad, duration=1.5)
                            smooth_move_rad(follower, sim, place_target, current_rad, duration=0.8)
                            smooth_move_rad(follower, sim, place_release, current_rad, duration=0.5)
                            smooth_move_rad(follower, sim, place_retreat, current_rad, duration=1.0)
                            print("🎉 Pick & Place 動作が正常に完了しました！")
                        except Exception as e:
                            print(f"❌ 実機実行中にエラーが発生しました: {e}")
                    else:
                        print("🛑 実機実行はキャンセルされました。")

                    pick_pt, place_pt = None, None
                    print("\n👉 次の Pick 地点を選択してください（または [Q] で終了）。")

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == 32:
                if projector.update_homography(frame):
                    print("🔄 キャリブレーションを再計算しました。")
            elif key in [ord('h'), ord('H')]:
                print("🏠 Home 姿勢へ退避します...")
                smooth_move_rad(follower, sim, home_rad, current_rad, duration=1.5)
                pick_pt, place_pt = None, None

    except KeyboardInterrupt:
        pass
    finally:
        print("\n🏠 終了処理...")
        try:
            smooth_move_rad(follower, sim, home_rad, current_rad, duration=1.5)
        except Exception:
            pass
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()