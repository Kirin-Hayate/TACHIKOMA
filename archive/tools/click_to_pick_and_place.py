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

# Pick / Place における目標高さ [mm]
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


def move_to_home_and_wait(follower, sim, home_rad, current_rad, timeout=4.0):
    """手首(ID4)を先行して引き起こし、確実に全サーボが Home 姿勢に到達・静止するまで待機"""
    print("🏠 Home 姿勢へ安全復帰中...")

    # ステップ1: ID4 (手首ピッチ) を先行して初期姿勢へ引き上げる (干渉・脱落防止)
    intermediate_rad = dict(current_rad)
    intermediate_rad[4] = home_rad[4]
    smooth_move_rad(follower, sim, intermediate_rad, current_rad, duration=1.0, steps=20)

    # ステップ2: 全軸を Home 姿勢へ補間移動
    smooth_move_rad(follower, sim, home_rad, current_rad, duration=2.0, steps=35)

    # ステップ3: 実機サーボの物理到達を監視
    if follower is not None:
        start_t = time.time()
        while time.time() - start_t < timeout:
            all_reached = True
            for sid in [1, 2, 3, 4]:
                pos = follower.read_position(sid)
                if pos is not None:
                    cur_angle = raw_to_radian(sid, pos)
                    threshold = 0.15 if sid == 4 else 0.08
                    if abs(cur_angle - home_rad[sid]) > threshold:
                        all_reached = False
                        break
            if all_reached:
                break
            time.sleep(0.05)
        time.sleep(0.3)
    print("✅ Home 姿勢への復帰が完了しました。")


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


click_buffer = []

def on_mouse_click(event, x, y, flags, param):
    global click_buffer
    if event == cv2.EVENT_LBUTTONDOWN:
        click_buffer.append((x, y))


def compute_target_poses(r_cm, th_deg, z_mm, gripper_rad):
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

    # 1. 実機・シミュレータ初期化
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
        print(f"⚠️ 3Dシミュレータ初期化スキップ: {e}")

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

    # アームをまず Home 姿勢へ移動させてマーカーの視界をクリアにする
    move_to_home_and_wait(follower, sim, home_rad, current_rad)

    # マーカー認識と初期ホモグラフィ構築
    print("⏳ カメラから4隅マーカーを探索・初期化中...")
    calibrated = False
    for _ in range(40):
        ret, frame = cap.read()
        if ret and projector.update_homography(frame):
            calibrated = True
            print("✅ 4隅マーカーを検出し、ホモグラフィ行列を固定しました。")
            break
        time.sleep(0.05)

    if not calibrated:
        print("⚠️ 起動時に4枚すべてのマーカーを検出できませんでした。")
        print("   画角を確認し、4枚がカメラに見える状態で [SPACE] を押してください。")

    cv2.namedWindow("Pick & Place Planner")
    cv2.setMouseCallback("Pick & Place Planner", on_mouse_click)
    cv2.namedWindow("Raw Camera Preview")

    pick_pt = None
    place_pt = None

    print("\n👉 【操作手順】")
    print("   1. [Pick & Place Planner] 画面でまず **[Pick 地点]** をクリック")
    print("   2. 次に **[Place 地点]** をクリック")
    print("   3. ターミナルで 'y' を入力して実行")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 一度キャリブレーションできていれば、その行列を用いて安定ワーピング
            warped = None
            if projector.homography_mat is not None:
                warped = cv2.warpPerspective(frame, projector.homography_mat, (CANVAS_SIZE, CANVAS_SIZE))

            display_img = warped.copy() if warped is not None else np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

            # ガイドグリッド描画
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

            # 状態メッセージ
            if warped is None:
                msg = "Press [SPACE] to Calibrate 4 Markers"
                status_color = (0, 0, 255)
            elif pick_pt is None:
                msg = "1. Click PICK point"
                status_color = (0, 255, 0)
            elif place_pt is None:
                msg = "2. Click PLACE point"
                status_color = (0, 255, 255)
            else:
                msg = "Ready! Check Terminal."
                status_color = (255, 200, 0)

            cv2.putText(display_img, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
            cv2.imshow("Pick & Place Planner", display_img)

            # 生カメラプレビュー（マーカー認識状態を可視化）
            raw_preview = cv2.resize(frame, (480, 270))
            detected_centers = projector.detect_markers(frame)
            for mid, pt in detected_centers.items():
                cx, cy = int(pt[0] * 480 / 1280), int(pt[1] * 270 / 720)
                cv2.circle(raw_preview, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(raw_preview, f"ID{mid}", (cx - 15, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            det_status = " | ".join([f"ID{i}:{'OK' if i in detected_centers else '--'}" for i in range(4)])
            cv2.putText(raw_preview, det_status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Raw Camera Preview", raw_preview)

            # --- クリック処理 ---
            if len(click_buffer) > 0:
                pt = click_buffer.pop(0)
                if warped is None:
                    print("⚠️ マーカー認識が未完了です。[SPACE] を押してキャリブレーションを行ってください。")
                    continue

                if pick_pt is None:
                    pick_pt = pt
                    print(f"📍 Pick 地点決定: Pixel={pick_pt}")
                elif place_pt is None:
                    place_pt = pt
                    print(f"📍 Place 地点決定: Pixel={place_pt}")
                    time.sleep(0.3)

                    print("\n--- 軌道生成・IK検証を開始 ---")
                    _, _, pick_r, pick_th = pixel_to_robot_polar(pick_pt[0], pick_pt[1], projector)
                    _, _, place_r, place_th = pixel_to_robot_polar(place_pt[0], place_pt[1], projector)

                    _, pick_wp, _ = compute_target_poses(pick_r, pick_th, LIFT_Z_MM, GRIPPER_OPEN_RAD)
                    pick_target, _, _ = compute_target_poses(pick_r, pick_th, DEFAULT_Z_MM, GRIPPER_OPEN_RAD)
                    pick_grasp, _, _ = compute_target_poses(pick_r, pick_th, DEFAULT_Z_MM, GRIPPER_CLOSE_RAD)
                    _, pick_lift, _ = compute_target_poses(pick_r, pick_th, LIFT_Z_MM, GRIPPER_CLOSE_RAD)

                    _, place_wp, _ = compute_target_poses(place_r, place_th, LIFT_Z_MM, GRIPPER_CLOSE_RAD)
                    place_target, _, _ = compute_target_poses(place_r, place_th, DEFAULT_Z_MM, GRIPPER_CLOSE_RAD)
                    place_release, _, _ = compute_target_poses(place_r, place_th, DEFAULT_Z_MM, GRIPPER_OPEN_RAD)
                    _, place_retreat, _ = compute_target_poses(place_r, place_th, LIFT_Z_MM, GRIPPER_OPEN_RAD)

                    if any(p is None for p in [pick_target, place_target]):
                        print("❌ [IK解なし] 指定されたPickまたはPlace地点は可動範囲外です。リセットします。")
                        pick_pt, place_pt = None, None
                        continue

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

                    # Home 姿勢へ退避して次へ
                    move_to_home_and_wait(follower, sim, home_rad, current_rad)
                    pick_pt, place_pt = None, None
                    print("\n👉 次の Pick 地点を選択してください（または [Q] で終了）。")

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == 32:  # SPACE: 強制再キャリブレーション
                if projector.update_homography(frame):
                    print("🔄 キャリブレーションを最新フレームで更新・固定しました。")
                else:
                    print("⚠️ 4枚のマーカーが全て見えていません。画角を確認してください。")
            elif key in [ord('h'), ord('H')]:
                move_to_home_and_wait(follower, sim, home_rad, current_rad)
                pick_pt, place_pt = None, None

    except KeyboardInterrupt:
        pass
    finally:
        try:
            move_to_home_and_wait(follower, sim, home_rad, current_rad)
        except Exception:
            pass

        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
            print("✅ 全サーボのトルクを OFF にし、ポートを閉じました。")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()