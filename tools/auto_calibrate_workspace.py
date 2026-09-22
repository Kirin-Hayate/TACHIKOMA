import sys
import os
import time
import math
import csv
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
    GRIPPER_CLOSE_RAD
)
from core.square_workspace_solver import SquareWorkspaceSolver

# 設定・出力パス
JSON_CONFIG_PATH = os.path.join(BASE_DIR, "config", "markers_config.json")
CSV_CONFIG_PATH = os.path.join(BASE_DIR, "config", "calibrated_markers.csv")

# 操作高度・ステップ設定
CALIB_Z_MM = 10.0         # 校正時の机上固定高さ [mm]
STEP_NORMAL_MM = 5.0      # 通常移動ステップ [mm]
STEP_FINE_MM = 1.0        # 微動ステップ [mm]
MOVE_DURATION = 0.25      # ジョグ移動の補間時間 [s]


def draw_outlined_text(img, text, pos, font_scale=0.55, text_color=(255, 255, 255), thickness=1):
    """背景を遮蔽しない、黒縁取り付きの高視認性透過テキスト描画"""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 黒のアウトライン
    #cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # 前景文字
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def smooth_move_rad(follower, sim, target_rad, current_rad, duration=0.25, steps=10):
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
    """Home 姿勢へ遷移し、特に手首(ID4)の引き上げを完了させてから確実に静止待機する"""
    print("🏠 Home 姿勢へ移動中...")

    # ステップ1: まず ID4 (手首ピッチ) を上向きに逃がしてから全体を Home に戻す
    # (机面との擦れや無理なトルクを防ぎ、ID4の戻り遅れを解消する)
    intermediate_rad = dict(current_rad)
    intermediate_rad[4] = home_rad[4]  # 先に手首ピッチを初期姿勢へ引き上げ
    smooth_move_rad(follower, sim, intermediate_rad, current_rad, duration=1.0, steps=20)

    # ステップ2: 全軸を Home 姿勢へ補間移動 (余裕を持たせて2.0秒)
    smooth_move_rad(follower, sim, home_rad, current_rad, duration=2.0, steps=35)
    
    # ステップ3: サーボの物理到達を監視 (ID4 の許容誤差をやや緩和)
    if follower is not None:
        start_t = time.time()
        while time.time() - start_t < timeout:
            all_reached = True
            for sid in [1, 2, 3, 4]:
                pos = follower.read_position(sid)
                if pos is not None:
                    cur_angle = raw_to_radian(sid, pos)
                    # 自重による負荷がかかる ID4 は許容誤差を 0.15 rad (約8.5度) に設定
                    threshold = 0.15 if sid == 4 else 0.08
                    if abs(cur_angle - home_rad[sid]) > threshold:
                        all_reached = False
                        break
            if all_reached:
                break
            time.sleep(0.05)
        time.sleep(0.3)  # 静止安定化マージン
        
    print("✅ Home 姿勢への復帰が完了しました。")


def save_to_csv(calibrated_result: Dict[int, Dict[str, float]], csv_path: str):
    """キャリブレーション結果を CSV に上書き保存"""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["marker_id", "description", "r_cm", "theta_deg", "x_mm", "y_mm"])
        for mid in sorted(calibrated_result.keys()):
            d = calibrated_result[mid]
            writer.writerow([mid, d["description"], d["r_cm"], d["theta_deg"], d["x_mm"], d["y_mm"]])
    print(f"💾 CSV 設定を上書き保存しました: {csv_path}")


def main():
    print("==================================================")
    print(" 🛠️ TACHIKOMA ワークスペース自動キャリブレーション")
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

    # 2. カメラ初期化
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. キャリブレーションステート管理
    cur_x_mm = 135.0
    cur_y_mm = -220.0
    calib_step = 0  # 0: ID 2 (手前左), 1: ID 3 (手前右)
    recorded_points = {}

    solver = SquareWorkspaceSolver(side_length_mm=400.0)

    cv2.namedWindow("Calibration Jog Controller")

    # 初期位置へアプローチ
    r_init_cm = math.hypot(cur_x_mm, cur_y_mm) / 10.0
    th_init_deg = math.degrees(math.atan2(cur_y_mm, cur_x_mm))
    init_tgt, init_wp, _ = solve_ik_adaptive_approach(
        r_tcp=(r_init_cm / 100.0),
        theta_deg=th_init_deg,
        z_tcp=(CALIB_Z_MM / 1000.0),
        gripper_rad=GRIPPER_CLOSE_RAD,
        enable_sag_compensation=True
    )
    if init_tgt is not None:
        if init_wp is not None:
            smooth_move_rad(follower, sim, init_wp, current_rad, duration=1.2)
        smooth_move_rad(follower, sim, init_tgt, current_rad, duration=0.8)

    print("\n👉 【キャリブレーション手順】")
    print("   [ステップ 1] アームを矢印キーで動かし、【ID 2 (手前左)】の中心に爪先を合わせて [ENTER] または [SPACE]")
    print("   [ステップ 2] アームを矢印キーで動かし、【ID 3 (手前右)】の中心に爪先を合わせて [ENTER] または [SPACE]")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_img = cv2.resize(frame, (960, 540))

            # ガイド情報
            target_name = "ID 2 (Front-Left)" if calib_step == 0 else "ID 3 (Front-Right)"
            color = (0, 255, 255) if calib_step == 0 else (0, 165, 255)

            r_cur_cm = math.hypot(cur_x_mm, cur_y_mm) / 10.0
            th_cur_deg = math.degrees(math.atan2(cur_y_mm, cur_x_mm))

            # 背景を黒で塗りつぶさず、完全透過のアウトライン文字で描画
            draw_outlined_text(display_img, f"Target: Align to [{target_name}]", (20, 30), font_scale=0.65, text_color=color, thickness=2)
            draw_outlined_text(display_img, f"Current Pos: X={cur_x_mm:.1f}mm, Y={cur_y_mm:.1f}mm, Z={CALIB_Z_MM:.1f}mm", (20, 60), font_scale=0.55, text_color=(255, 255, 255), thickness=1)
            draw_outlined_text(display_img, f"Polar: r={r_cur_cm:.2f}cm, theta={th_cur_deg:+.2f} deg", (20, 85), font_scale=0.55, text_color=(220, 220, 220), thickness=1)

            cv2.imshow("Calibration Jog Controller", display_img)

            key = cv2.waitKeyEx(30)

            dx = 0.0
            dy = 0.0
            step_size = STEP_NORMAL_MM

            if key in [2490368, 0x260000, ord('w'), ord('W')]:
                dx = step_size
            elif key in [2621440, 0x280000, ord('s'), ord('S')]:
                dx = -step_size
            elif key in [2424832, 0x250000, ord('a'), ord('A')]:
                dy = -step_size
            elif key in [2555904, 0x270000, ord('d'), ord('D')]:
                dy = step_size
            elif key in [ord('i'), ord('I')]:
                dx = STEP_FINE_MM
            elif key in [ord('k'), ord('K')]:
                dx = -STEP_FINE_MM
            elif key in [ord('j'), ord('J')]:
                dy = -STEP_FINE_MM
            elif key in [ord('l'), ord('L')]:
                dy = STEP_FINE_MM

            if dx != 0.0 or dy != 0.0:
                next_x = cur_x_mm + dx
                next_y = cur_y_mm + dy
                next_r_cm = math.hypot(next_x, next_y) / 10.0
                next_th_deg = math.degrees(math.atan2(next_y, next_x))

                tgt_rad, _, _ = solve_ik_adaptive_approach(
                    r_tcp=(next_r_cm / 100.0),
                    theta_deg=next_th_deg,
                    z_tcp=(CALIB_Z_MM / 1000.0),
                    gripper_rad=GRIPPER_CLOSE_RAD,
                    enable_sag_compensation=True
                )

                if tgt_rad is not None:
                    cur_x_mm = next_x
                    cur_y_mm = next_y
                    smooth_move_rad(follower, sim, tgt_rad, current_rad, duration=MOVE_DURATION, steps=8)
                else:
                    print("⚠️ 可動限界に達したため移動を制限しました。")

            # 確定キー: ENTER または SPACE
            elif key in [13, 32]:
                if calib_step == 0:
                    recorded_points[2] = (cur_x_mm, cur_y_mm)
                    print(f"\n📍 [ID 2 (手前左)] を記録: X={cur_x_mm:.1f}mm, Y={cur_y_mm:.1f}mm")
                    calib_step = 1

                    print("🚀 アームを ID 3 (手前右) の初期位置へ移動中...")
                    cur_x_mm = 125.0
                    cur_y_mm = 220.0
                    next_r_cm = math.hypot(cur_x_mm, cur_y_mm) / 10.0
                    next_th_deg = math.degrees(math.atan2(cur_y_mm, cur_x_mm))
                    tgt_rad, wp_rad, _ = solve_ik_adaptive_approach(
                        r_tcp=(next_r_cm / 100.0),
                        theta_deg=next_th_deg,
                        z_tcp=(CALIB_Z_MM / 1000.0),
                        gripper_rad=GRIPPER_CLOSE_RAD,
                        enable_sag_compensation=True
                    )
                    if wp_rad is not None and tgt_rad is not None:
                        smooth_move_rad(follower, sim, wp_rad, current_rad, duration=1.0)
                        smooth_move_rad(follower, sim, tgt_rad, current_rad, duration=0.8)
                    print("👉 矢印キーで微調整し、【ID 3 (手前右)】の中心に合わせて [ENTER] を押してください。")

                elif calib_step == 1:
                    recorded_points[3] = (cur_x_mm, cur_y_mm)
                    print(f"\n📍 [ID 3 (手前右)] を記録: X={cur_x_mm:.1f}mm, Y={cur_y_mm:.1f}mm")

                    print("\n📐 正方形幾何拘束 (400mm) を用いて全作業空間を計算中...")
                    calibrated_result = solver.solve_from_front_points(
                        p2_xy=recorded_points[2],
                        p3_xy=recorded_points[3],
                        enforce_nominal_length=True
                    )

                    print("\n🎉 === キャリブレーション完了結果 ===")
                    for mid in sorted(calibrated_result.keys()):
                        d = calibrated_result[mid]
                        print(f"  Marker {mid} ({d['description']}): r={d['r_cm']:.2f}cm, θ={d['theta_deg']:+.2f}° | X={d['x_mm']:.1f}mm, Y={d['y_mm']:.1f}mm")

                    solver.export_to_config_json(calibrated_result, JSON_CONFIG_PATH)
                    save_to_csv(calibrated_result, CSV_CONFIG_PATH)
                    print("✅ 設定を保存しました。")
                    break

            elif key in [ord('q'), ord('Q'), 27]:
                print("\n🛑 中断しました。")
                break
            elif key in [ord('h'), ord('H')]:
                move_to_home_and_wait(follower, sim, home_rad, current_rad)

    except KeyboardInterrupt:
        pass
    finally:
        # 必ず Home 姿勢へ完全復帰してから切断
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