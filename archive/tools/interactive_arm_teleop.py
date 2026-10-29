"""
==============================================================================
SO-ARM100 リアルタイム座標確認 & キーボード対話テレオペツール
(tools/interactive_arm_teleop.py)
==============================================================================
【概要】
キーボード操作でアーム手先目標位置を微小移動させ、
「直交座標 (X, Y, Z)」「極座標 (r, theta, z)」「たわみ補正量」「各サーボ指令値」
をコンソールおよび MuJoCo 画面上にリアルタイム表示するティーチング・確認ツールです。
==============================================================================
"""

import sys
import os
import time
import math
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.sim_viewer import MujocoSimViewer
from core.kinematics import (
    get_home_radians,
    solve_ik_wrist_and_pitch,
    calculate_sag_compensation,
    radian_to_raw,
    raw_to_radian,
    L_GRIPPER,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD,
    WRIST_ROLL_HORIZONTAL_RAD
)
from core.trajectory_executor import TrajectoryExecutor

try:
    from core.bus_servo_controller import BusServoController
    HAS_HARDWARE_MODULE = True
except ImportError:
    HAS_HARDWARE_MODULE = False

# 初期目標値 (机上手前中央付近)
CURRENT_X_MM = 220.0
CURRENT_Y_MM = 0.0
CURRENT_Z_MM = 30.0
CURRENT_PITCH_DEG = 60.0
CURRENT_ROLL_RAD = WRIST_ROLL_HORIZONTAL_RAD
CURRENT_GRIPPER_RAD = GRIPPER_OPEN_RAD

REQ_GO_HOME = False
REQ_QUIT = False
DIRTY_TARGET = True


def print_status(x_mm, y_mm, z_mm, r_mm, theta_deg, sag_mm, pitch_deg, roll_rad, rad_targets):
    """コンソールに現在の座標とサーボ指令値を綺麗に出力"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 65)
    print(" 🦾 SO-ARM100 リアルタイム手先座標 & 指令値モニタ")
    print("=" * 65)
    print(f" 📍 直交座標 (机面基準) : X = {x_mm:+6.1f} mm,  Y = {y_mm:+6.1f} mm,  Z = {z_mm:+6.1f} mm")
    print(f" 🌐 極座標 (ベース基準) : r = {r_mm:6.1f} mm,  θ = {theta_deg:+6.1f}°,  z = {z_mm:+6.1f} mm")
    print(f" ⚖️ 自重たわみ補正量    : Δz = {sag_mm:+5.1f} mm (実補正後Z: {z_mm + sag_mm:+6.1f} mm)")
    print(f" 📐 手先姿勢拘束        : ピッチ = {pitch_deg:5.1f}°,  ロール = {math.degrees(roll_rad):+6.1f}°")
    print("-" * 65)
    if rad_targets:
        raw_str = " | ".join([f"ID{sid}: {radian_to_raw(sid, rad_targets[sid]):4d}" for sid in range(1, 7)])
        rad_str = " | ".join([f"ID{sid}:{rad_targets[sid]:+5.2f}" for sid in range(1, 7)])
        print(f" ⚙️ サーボ Raw生値     : {raw_str}")
        print(f" 🔄 関節ラジアン(rad)   : {rad_str}")
    else:
        print(" ❌ 【警告】指定座標はアーム可動域外 (IK解なし) です！")
    print("=" * 65)
    print("【キーボード操作ガイド (OpenCVウィンドウまたはMuJoCoにフォーカス)】")
    print("  [W / S] : 前進 / 後退 (X ±5mm)      [A / D] : 左旋回 / 右旋回 (Y ±5mm)")
    print("  [R / F] : 上昇 / 下降 (Z ±5mm)      [U / J] : ピッチ角 (±5°)")
    print("  [O / K] : 手首ロール (±5°)          [C / V] : グリッパー (開 / 閉)")
    print("  [H]     : Home 姿勢へ復帰           [Q/ESC] : 終了")
    print("-" * 65)


def key_process(key_char: str):
    global CURRENT_X_MM, CURRENT_Y_MM, CURRENT_Z_MM
    global CURRENT_PITCH_DEG, CURRENT_ROLL_RAD, CURRENT_GRIPPER_RAD
    global REQ_GO_HOME, REQ_QUIT, DIRTY_TARGET

    k = key_char.lower()
    step_mm = 5.0
    step_deg = 5.0

    if k == 'w':
        CURRENT_X_MM += step_mm
        DIRTY_TARGET = True
    elif k == 's':
        CURRENT_X_MM -= step_mm
        DIRTY_TARGET = True
    elif k == 'a':
        CURRENT_Y_MM -= step_mm
        DIRTY_TARGET = True
    elif k == 'd':
        CURRENT_Y_MM += step_mm
        DIRTY_TARGET = True
    elif k == 'r':
        CURRENT_Z_MM += step_mm
        DIRTY_TARGET = True
    elif k == 'f':
        CURRENT_Z_MM = max(0.0, CURRENT_Z_MM - step_mm)
        DIRTY_TARGET = True
    elif k == 'u':
        CURRENT_PITCH_DEG = min(85.0, CURRENT_PITCH_DEG + step_deg)
        DIRTY_TARGET = True
    elif k == 'j':
        CURRENT_PITCH_DEG = max(0.0, CURRENT_PITCH_DEG - step_deg)
        DIRTY_TARGET = True
    elif k == 'o':
        CURRENT_ROLL_RAD += math.radians(step_deg)
        DIRTY_TARGET = True
    elif k == 'k':
        CURRENT_ROLL_RAD -= math.radians(step_deg)
        DIRTY_TARGET = True
    elif k == 'c':
        CURRENT_GRIPPER_RAD = GRIPPER_OPEN_RAD
        DIRTY_TARGET = True
    elif k == 'v':
        CURRENT_GRIPPER_RAD = GRIPPER_CLOSE_RAD
        DIRTY_TARGET = True
    elif k == 'h':
        REQ_GO_HOME = True
    elif k in ('q', '\x1b'):
        REQ_QUIT = True


def sim_key_callback(keycode: int):
    try:
        char = chr(keycode)
        key_process(char)
    except Exception:
        if keycode in (81, 113, 256):
            key_process('q')
        elif keycode in (72, 104):
            key_process('h')


def main():
    global CURRENT_X_MM, CURRENT_Y_MM, CURRENT_Z_MM
    global CURRENT_PITCH_DEG, CURRENT_ROLL_RAD, CURRENT_GRIPPER_RAD
    global REQ_GO_HOME, REQ_QUIT, DIRTY_TARGET

    controller = None
    if HAS_HARDWARE_MODULE:
        try:
            controller = BusServoController()
            if controller.connect():
                print("✅ 実機サーボコントローラに接続成功しました。")
            else:
                controller = None
        except Exception:
            controller = None

    sim = MujocoSimViewer()
    home_rad = get_home_radians()
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass

    sim.viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=sim_key_callback
    )

    executor = TrajectoryExecutor(sim=sim, servo_controller=controller)

    if executor.is_real_connected:
        print("🤖 実機を初期 Home 姿勢へ移動中...")
        executor.move_to_home_and_wait(home_rad)  

    # 操作受付・表示用の OpenCV ウィンドウ
    dummy_ui = np.zeros((220, 520, 3), dtype=np.uint8)
    cv2.namedWindow("Keyboard Teleop Controller", cv2.WINDOW_AUTOSIZE)

    prev_rad_targets = dict(home_rad)

    try:
        while sim.is_running() and not REQ_QUIT:
            # [H] Home 復帰
            if REQ_GO_HOME:
                REQ_GO_HOME = False
                print("\n🏠 Home 姿勢へ復帰中...")
                executor.move_to_home_and_wait(home_rad)
                CURRENT_X_MM = 220.0
                CURRENT_Y_MM = 0.0
                CURRENT_Z_MM = 30.0
                DIRTY_TARGET = True

            # 目標値の変更があった場合のみ IK 計算と送信を実行
            if DIRTY_TARGET:
                DIRTY_TARGET = False

                r_m = math.hypot(CURRENT_X_MM, CURRENT_Y_MM) / 1000.0
                theta_rad = math.atan2(CURRENT_Y_MM, CURRENT_X_MM)
                theta_deg = math.degrees(theta_rad)

                # 最大リーチ制限
                if r_m > 0.410:
                    scale = 0.410 / r_m
                    CURRENT_X_MM *= scale
                    CURRENT_Y_MM *= scale
                    r_m = 0.410

                # たわみ補正量の計算
                sag_offset_m = calculate_sag_compensation(r_m, theta_rad)
                effective_z = (CURRENT_Z_MM / 1000.0) + sag_offset_m

                # 手首目標位置
                pitch_rad = math.radians(CURRENT_PITCH_DEG)
                r_wrist = r_m - L_GRIPPER * math.cos(pitch_rad)
                z_wrist = effective_z + L_GRIPPER * math.sin(pitch_rad)

                # IK 解の計算
                rad_targets, reason = solve_ik_wrist_and_pitch(
                    r_wrist=r_wrist,
                    theta_deg=theta_deg,
                    z_wrist=z_wrist,
                    target_pitch_deg=CURRENT_PITCH_DEG,
                    gripper_rad=CURRENT_GRIPPER_RAD,
                    wrist_roll_rad=CURRENT_ROLL_RAD,
                    return_reason=True
                )

                print_status(
                    x_mm=CURRENT_X_MM,
                    y_mm=CURRENT_Y_MM,
                    z_mm=CURRENT_Z_MM,
                    r_mm=r_m * 1000.0,
                    theta_deg=theta_deg,
                    sag_mm=sag_offset_m * 1000.0,
                    pitch_deg=CURRENT_PITCH_DEG,
                    roll_rad=CURRENT_ROLL_RAD,
                    rad_targets=rad_targets
                )

                if rad_targets is not None:
                    # アームを移動 (微小移動なので 0.15秒でクイック追従)
                    executor.move_to_rad(rad_targets, duration_sec=0.15, steps=5, send_to_real=True)
                    prev_rad_targets = rad_targets

                    # MuJoCo 上に手先マーカーを描画
                    if sim.viewer is not None:
                        sim.viewer.user_scn.ngeom = 0
                        ng = sim.viewer.user_scn.ngeom
                        mujoco.mjv_initGeom(
                            sim.viewer.user_scn.geoms[ng],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.008, 0.008, 0.008]),
                            pos=np.array([-CURRENT_Y_MM / 1000.0, -CURRENT_X_MM / 1000.0, CURRENT_Z_MM / 1000.0]),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0.0, 1.0, 0.5, 0.9])
                        )
                        sim.viewer.user_scn.ngeom += 1

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            # OpenCV ウィンドウの描画とキー受付
            ui_disp = dummy_ui.copy()
            cv2.putText(ui_disp, f"X:{CURRENT_X_MM:+.0f} Y:{CURRENT_Y_MM:+.0f} Z:{CURRENT_Z_MM:+.0f} (mm)", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(ui_disp, f"r:{math.hypot(CURRENT_X_MM, CURRENT_Y_MM):.0f}mm  th:{math.degrees(math.atan2(CURRENT_Y_MM, CURRENT_X_MM)):+.1f}deg", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(ui_disp, "W/S: X | A/D: Y | R/F: Z", (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(ui_disp, "U/J: Pitch | O/K: Roll | C/V: Grip", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(ui_disp, "[H] Home | [Q] Quit", (15, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
            cv2.imshow("Keyboard Teleop Controller", ui_disp)

            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key != 255:
                try:
                    key_process(chr(key))
                except Exception:
                    pass

    finally:
        if controller is not None:
            controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()