"""
==============================================================================
手首ロール角度対話型キャリブレーションツール
(tools/calibrate_wrist_roll.py)
==============================================================================
【目的】
1. カメラ画像から机上の物体を検出し、MuJoCo 空間上に実寸直方体として同期。
2. 選択中ターゲット (#0〜#9) の把持位置へアームを自動進入。
3. キー操作で手首ロール (ID 5) の角度を微調整 (±1° / ±5°)。
4. グリッパの挟み込み面が物体とぴったり合致した瞬間に [SPACE] を押すことで、
   物体の角度、台座旋回角、手首ロール目標角 (rad/deg/Raw) をターミナルに記録。
   これにより、手首ロール算出式（オフセット・符号・位相）の確定データを取得。

【操作】
  - [0]〜[9] : 把持対象物体の選択
  - [G]      : 選択対象の上空・把持位置へアームを移動
  - [J] / [L]: 手首ロールを微調整 (-2° / +2°)
  - [U] / [O]: 手首ロールを大きく調整 (-10° / +10°)
  - [SPACE]  : 現在の手首角度と物体の傾き関係をログ出力・保存
  - [H]      : ホーム姿勢に戻す
  - [B]      : 背景差分更新
  - [Q/ESC]  : 終了
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
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.tabletop_detector import TabletopDetector
from core.kinematics import (
    get_home_radians,
    solve_ik_wrist_and_pitch,
    calculate_sag_compensation,
    radian_to_raw,
    raw_to_radian,
    L_GRIPPER,
    GRIPPER_OPEN_RAD,
    WRIST_ROLL_HORIZONTAL_RAD
)

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

SELECTED_TARGET_IDX = 0
CURRENT_WRIST_ROLL_RAD = WRIST_ROLL_HORIZONTAL_RAD

IS_AT_TARGET = False
LOCKED_OBJ_DATA: Optional[dict] = None

REQ_EXEC_APPROACH = False
REQ_GO_HOME = False
REQ_LOG_DATA = False
REQ_ADJUST_ROLL = 0.0  # rad 単位の調整量
REQ_SAVE_BG = False
REQ_RECALIB = False
REQ_QUIT = False


def custom_sim_key_callback(keycode: int):
    global SELECTED_TARGET_IDX, CURRENT_WRIST_ROLL_RAD, REQ_EXEC_APPROACH
    global REQ_GO_HOME, REQ_LOG_DATA, REQ_ADJUST_ROLL, REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT

    # 数字キー [0]〜[9]
    if 48 <= keycode <= 57:
        SELECTED_TARGET_IDX = keycode - 48
        print(f"\n🎯 ターゲット #{SELECTED_TARGET_IDX} を選択しました。([G] でアプローチ)")
    elif keycode in (71, 103):  # G, g
        REQ_EXEC_APPROACH = True
    elif keycode in (72, 104):  # H, h
        REQ_GO_HOME = True
    elif keycode == 32:  # SPACE
        REQ_LOG_DATA = True
    elif keycode in (74, 106, 260):  # J, j, Left Arrow
        REQ_ADJUST_ROLL -= math.radians(2.0)
    elif keycode in (76, 108, 262):  # L, l, Right Arrow
        REQ_ADJUST_ROLL += math.radians(2.0)
    elif keycode in (85, 117):  # U, u
        REQ_ADJUST_ROLL -= math.radians(10.0)
    elif keycode in (79, 111):  # O, o
        REQ_ADJUST_ROLL += math.radians(10.0)
    elif keycode in (66, 98):  # B, b
        REQ_SAVE_BG = True
    elif keycode in (81, 113, 256):  # Q, q, ESC
        REQ_QUIT = True


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def interpolate_motion(sim, target_rad: Dict[int, float], steps: int = 30, delay_sec: float = 0.02):
    start_qpos = np.copy(sim.data.qpos[:6])
    target_qpos = np.copy(start_qpos)
    for sid, rad in target_rad.items():
        if 1 <= sid <= 6:
            target_qpos[sid - 1] = rad

    for s in range(1, steps + 1):
        ratio = s / float(steps)
        t = ratio * ratio * (3 - 2 * ratio)
        current = (1.0 - t) * start_qpos + t * target_qpos
        sim.data.qpos[:6] = current
        mujoco.mj_forward(sim.model, sim.data)
        if sim.viewer is not None:
            sim.viewer.sync()
        time.sleep(delay_sec)


def draw_aruco_markers_in_mujoco(sim, projector, marker_size_m: float = 0.04):
    if sim.viewer is None:
        return
    half_s = marker_size_m / 2.0
    half_th = 0.0002

    for marker_idx in range(4):
        phys_x, phys_y = projector.marker_phys_xy[marker_idx]
        mj_x = -phys_y / 1000.0
        mj_y = -phys_x / 1000.0

        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([half_s, half_s, half_th], dtype=np.float64),
                pos=np.array([mj_x, mj_y, half_th], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.9, 0.9, 0.9, 0.95], dtype=np.float32)
            )
            sim.viewer.user_scn.ngeom += 1


def main():
    global SELECTED_TARGET_IDX, CURRENT_WRIST_ROLL_RAD, IS_AT_TARGET, LOCKED_OBJ_DATA
    global REQ_EXEC_APPROACH, REQ_GO_HOME, REQ_LOG_DATA, REQ_ADJUST_ROLL, REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT

    print("==================================================")
    print(" 🛠️ 手首ロール角度 対話型キャリブレーション")
    print("==================================================")
    print("【操作手順】")
    print("  1. [0]〜[9] で物体を選び、[G] でアームを物体直上へ伸ばす")
    print("  2. [J] / [L] (または [U] / [O]) で手首を回転させ、物体の向きにぴったり合わせる")
    print("  3. 角度が合ったら [SPACE] を押す (ターミナルに解析ログを記録)")
    print("  [H] : ホーム姿勢復帰 | [B] : 背景更新 | [Q/ESC] : 終了")
    print("--------------------------------------------------")

    sim = MujocoSimViewer()
    home_rad = get_home_radians()
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass

    sim.viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=custom_sim_key_callback
    )

    # スロット初期化
    slot_info = []
    for i in range(MAX_SLOTS):
        bname = f"obj_block_{i}"
        gname = f"geom_obj_{i}"
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, bname)
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if bid == -1:
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"jenga_block_{i}")
            if bid != -1:
                gid = sim.model.body_geomadr[bid]

        qpos_adr = None
        if bid != -1:
            jnt_adr = sim.model.body_jntadr[bid]
            if jnt_adr != -1:
                qpos_adr = sim.model.jnt_qposadr[jnt_adr]
        slot_info.append({"bid": bid, "gid": gid, "qpos_adr": qpos_adr})

    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = TabletopDetector(projector)
    cv2.namedWindow("Wrist Calibration Visualizer")

    logged_records = []

    try:
        while sim.is_running() and not REQ_QUIT:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=500, out_h=500)
            if warped is None:
                continue

            if REQ_SAVE_BG:
                detector.update_background(warped)
                REQ_SAVE_BG = False

            # 物体検出
            detected_objs, _ = detector.detect_objects(warped)
            annotated = detector.draw_annotations(warped, detected_objs, target_idx=SELECTED_TARGET_IDX)

            # MuJoCo 直方体同期
            if sim.viewer is not None:
                sim.viewer.user_scn.ngeom = 0

            draw_aruco_markers_in_mujoco(sim, projector)

            for i in range(MAX_SLOTS):
                sinfo = slot_info[i]
                qadr, gid = sinfo["qpos_adr"], sinfo["gid"]
                if qadr is None or gid == -1:
                    continue

                if i < len(detected_objs):
                    obj = detected_objs[i]
                    x_mm, y_mm = obj["phys_xy"]
                    major_mm, minor_mm = obj["size_mm"]

                    mj_x = -y_mm / 1000.0
                    mj_y = -x_mm / 1000.0
                    yaw_rad = math.radians(-obj["angle_deg"])

                    sim.data.qpos[qadr:qadr + 3] = [mj_x, mj_y, HALF_Z]
                    sim.data.qpos[qadr + 3:qadr + 7] = euler_yaw_to_quat(yaw_rad)

                    half_x = max(0.005, (major_mm / 1000.0) / 2.0)
                    half_y = max(0.005, (minor_mm / 1000.0) / 2.0)
                    sim.model.geom_size[gid] = [half_x, half_y, HALF_Z]

                    # 3D ラベル
                    is_target = (i == SELECTED_TARGET_IDX)
                    label_text = f"#{i} TARGET" if is_target else f"#{i}"
                    rgba = np.array([1.0, 0.2, 0.2, 1.0]) if is_target else np.array([1.0, 1.0, 0.2, 1.0])

                    if sim.viewer is not None and sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
                        ng = sim.viewer.user_scn.ngeom
                        label_pos = np.array([mj_x, mj_y, HALF_Z + 0.035], dtype=np.float64)
                        mujoco.mjv_initGeom(
                            sim.viewer.user_scn.geoms[ng],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.zeros(3),
                            pos=label_pos,
                            mat=np.eye(3).flatten(),
                            rgba=rgba
                        )
                        sim.viewer.user_scn.geoms[ng].label = label_text.encode("utf-8")
                        sim.viewer.user_scn.ngeom += 1
                else:
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]

            # [H] ホーム復帰
            if REQ_GO_HOME:
                REQ_GO_HOME = False
                IS_AT_TARGET = False
                LOCKED_OBJ_DATA = None
                print("🏠 ホーム姿勢へ戻ります...")
                interpolate_motion(sim, home_rad, steps=25)

            # [G] 対象物体上空へアプローチ進入
            if REQ_EXEC_APPROACH:
                REQ_EXEC_APPROACH = False
                if SELECTED_TARGET_IDX < len(detected_objs):
                    LOCKED_OBJ_DATA = dict(detected_objs[SELECTED_TARGET_IDX])
                    x_mm, y_mm = LOCKED_OBJ_DATA["phys_xy"]
                    angle_deg = LOCKED_OBJ_DATA["angle_deg"]

                    r_tcp = math.hypot(x_mm, y_mm) / 1000.0
                    theta_deg = math.degrees(math.atan2(y_mm, x_mm))
                    z_target = 0.020 + calculate_sag_compensation(r_tcp, math.radians(theta_deg))  # 把持高さ 20mm

                    # ピッチ 60° 進入で位置を解く
                    pitch_rad = math.radians(60.0)
                    r_wrist = r_tcp - L_GRIPPER * math.cos(pitch_rad)
                    z_wrist = z_target + L_GRIPPER * math.sin(pitch_rad)

                    # 現在の手首ロール角度で姿勢を算出
                    rad_targets = solve_ik_wrist_and_pitch(
                        r_wrist=r_wrist,
                        theta_deg=theta_deg,
                        z_wrist=z_wrist,
                        target_pitch_deg=60.0,
                        gripper_rad=GRIPPER_OPEN_RAD,
                        wrist_roll_rad=CURRENT_WRIST_ROLL_RAD
                    )

                    if rad_targets is not None:
                        interpolate_motion(sim, rad_targets, steps=30)
                        IS_AT_TARGET = True
                        print(f"\n📍 ターゲット #{SELECTED_TARGET_IDX} 上空へ到達しました。")
                        print("👉 [J]/[L] で手首を回転させ、向きが合ったら [SPACE] を押してください。")
                    else:
                        print("❌ アプローチ位置の IK 解が算出できませんでした。")
                else:
                    print(f"⚠️ 指定された #{SELECTED_TARGET_IDX} の物体が見つかりません。")

            # 手首ロールのインタラクティブ微調整
            if REQ_ADJUST_ROLL != 0.0:
                CURRENT_WRIST_ROLL_RAD += REQ_ADJUST_ROLL
                REQ_ADJUST_ROLL = 0.0
                # アーム先端の手首ロール（ID 5: qpos[4]）を即時反映
                sim.data.qpos[4] = CURRENT_WRIST_ROLL_RAD
                mujoco.mj_forward(sim.model, sim.data)
                if sim.viewer is not None:
                    sim.viewer.sync()

                deg_disp = math.degrees(CURRENT_WRIST_ROLL_RAD)
                raw_disp = radian_to_raw(5, CURRENT_WRIST_ROLL_RAD)
                sys.stdout.write(f"\r   🔄 現在の手首ロール: {deg_disp:+6.1f}° (Raw={raw_disp:4d})    ")
                sys.stdout.flush()

            # [SPACE] 角度データの記録・解析
            if REQ_LOG_DATA:
                REQ_LOG_DATA = False
                if LOCKED_OBJ_DATA is not None:
                    x_mm, y_mm = LOCKED_OBJ_DATA["phys_xy"]
                    obj_angle = LOCKED_OBJ_DATA["angle_deg"]
                    base_theta = math.degrees(math.atan2(y_mm, x_mm))
                    wrist_deg = math.degrees(CURRENT_WRIST_ROLL_RAD)
                    wrist_raw = radian_to_raw(5, CURRENT_WRIST_ROLL_RAD)

                    # 角度関係の分析
                    # 期待関係式: wrist_deg = ± (obj_angle - base_theta) + Phase_Offset
                    diff_deg = obj_angle - base_theta
                    phase_offset = wrist_deg - diff_deg

                    print("\n\n" + "=" * 65)
                    print(f" 📝 [キャリブレーション記録 #{len(logged_records) + 1}]")
                    print("=" * 65)
                    print(f"  ・物体検出位置     : X={x_mm:.1f} mm, Y={y_mm:.1f} mm")
                    print(f"  ・物体検出角度     : {obj_angle:+.1f}°")
                    print(f"  ・台座旋回角 (Base): {base_theta:+.1f}°")
                    print(f"  ・相対角度 (Diff)  : {diff_deg:+.1f}°  (= 物体角度 - 台座角度)")
                    print(f"  -------------------------------------------------------------")
                    print(f"  🎯 合わせた手首角度: {wrist_deg:+.1f}° (Raw={wrist_raw})")
                    print(f"  💡 推定位相差 (Offset): {phase_offset:+.1f}°")
                    print(f"     => wrist_roll_deg ≈ (obj_angle - base_theta) + ({phase_offset:+.1f}°)")
                    print("=" * 65 + "\n")

                    logged_records.append({
                        "obj_angle": obj_angle,
                        "base_theta": base_theta,
                        "diff_deg": diff_deg,
                        "wrist_deg": wrist_deg,
                        "phase_offset": phase_offset
                    })
                else:
                    print("\n⚠️ まず [G] キーを押して物体上空へアームを移動させてください。")

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            deg_now = math.degrees(CURRENT_WRIST_ROLL_RAD)
            status_text = f"Target: #{SELECTED_TARGET_IDX} | Roll: {deg_now:+.1f}deg | [J]/[L]: Turn | [SPACE]: Record"
            cv2.putText(annotated, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Wrist Calibration Visualizer", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif 48 <= key <= 57:
                SELECTED_TARGET_IDX = key - 48
            elif key in [ord('g'), ord('G')]:
                REQ_EXEC_APPROACH = True
            elif key in [ord('h'), ord('H')]:
                REQ_GO_HOME = True
            elif key in [ord('j'), ord('J')]:
                REQ_ADJUST_ROLL -= math.radians(2.0)
            elif key in [ord('l'), ord('L')]:
                REQ_ADJUST_ROLL += math.radians(2.0)
            elif key in [ord('u'), ord('U')]:
                REQ_ADJUST_ROLL -= math.radians(10.0)
            elif key in [ord('o'), ord('O')]:
                REQ_ADJUST_ROLL += math.radians(10.0)
            elif key == 32:
                REQ_LOG_DATA = True

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logged_records:
            print("\n📋 === 収集データサマリー ===")
            offsets = [r["phase_offset"] for r in logged_records]
            mean_offset = np.mean(offsets)
            print(f"記録サンプル数: {len(logged_records)}")
            print(f"平均位相オフセット: {mean_offset:+.1f}°")
            print(f"推奨補正式: wrist_roll_rad = math.radians((angle_deg - base_theta_deg) + ({mean_offset:.1f}))")


if __name__ == "__main__":
    main()