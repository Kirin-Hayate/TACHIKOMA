"""
==============================================================================
自律連続 Pick & Place 自動化システム (MuJoCo リアルタイム可視化対応版)
(tools/run_auto_pick_and_place.py)
==============================================================================
【概要】
カメラ認識した机上の複数物体を自動選定し、人間の介入なしに
次々と指定エリア (Place Area) へ連続で仕分け・搬送する自律スクリプト。
MuJoCo 画面上にアーム・ArUcoマーカー・検出物体・把持マーカーをリアルタイム同期します。
==============================================================================
"""

import sys
import os
import time
import math
import random
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.tabletop_detector import TabletopDetector
from core.kinematics import (
    get_home_radians,
    solve_ik_tabletop_grasp,
    solve_ik_tabletop_place,
    calculate_sag_compensation,
    raw_to_radian,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)
from core.trajectory_executor import TrajectoryExecutor

try:
    from core.bus_servo_controller import BusServoController
    HAS_HARDWARE_MODULE = True
except ImportError:
    HAS_HARDWARE_MODULE = False

# --------------------------------------------------------------------------
# 把持・配置・動作パラメータ設定
# --------------------------------------------------------------------------
MANUAL_OFFSET_MAJOR_MM = -15.0   # 長い辺オフセット
MANUAL_OFFSET_MINOR_MM = -50.0   # 短い辺オフセット

PLACE_X_MM = 180.0               # 配置目標 X (mm)
PLACE_Y_MM = 160.0               # 配置目標 Y (mm)
PLACE_Z_MM = 20.0                # 配置目標 Z (机面 +20mm)
PLACE_ANGLE_DEG = 0.0            # 配置姿勢角

PICK_Z_MM = 2.0                  # 把持高度 (机面 +2mm)

MAX_RETRIES_PER_OBJECT = 6       # 同一物体への最大リトライ回数 (超えたらスキップ)
OBJECT_FAIL_HISTORY: Dict[str, int] = {}

# 爪完全閉止時の実測値
GRIPPER_CLOSED_RAW = 1889
EMPTY_GRASP_TOLERANCE_RAW = 30

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

REQ_QUIT = False
REQ_PAUSE = False
REQ_SAVE_BG = False

CURRENT_GRASP_TCP_MARKERS: Optional[Dict[str, np.ndarray]] = None


def sim_key_callback(keycode: int):
    global REQ_QUIT, REQ_PAUSE, REQ_SAVE_BG
    if keycode in (81, 113, 256):  # Q, ESC
        REQ_QUIT = True
    elif keycode in (32,):         # Space: 一時停止/再開
        REQ_PAUSE = not REQ_PAUSE
        state_str = "一時停止" if REQ_PAUSE else "自動再開"
        print(f"\n⏸️ 動作切り替え: {state_str}")
    elif keycode in (66, 98):      # B: 背景更新
        REQ_SAVE_BG = True


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def draw_markers(sim, projector, markers_dict):
    """MuJoCo の user_scn に ArUco マーカーと把持目標点を描画"""
    if sim.viewer is None:
        return
    sim.viewer.user_scn.ngeom = 0

    # 1. ArUco マーカーの四角形描画
    half_s, half_th = 0.02, 0.0002
    for m_idx in range(4):
        px, py = projector.marker_phys_xy[m_idx]
        mx, my = -py / 1000.0, -px / 1000.0
        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([half_s, half_s, half_th]),
                pos=np.array([mx, my, half_th]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.9, 0.9, 0.9, 0.9])
            )
            sim.viewer.user_scn.ngeom += 1

    # 2. 把持目標点 (シアン) & たわみ補正点 (赤) の球体マーカー
    if markers_dict:
        t_pos = markers_dict.get("target_tcp")
        s_pos = markers_dict.get("sag_tcp")
        if t_pos is not None and sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.007, 0.007, 0.007]),
                pos=t_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.1, 0.8, 1.0, 0.9])
            )
            sim.viewer.user_scn.ngeom += 1
        if s_pos is not None and sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.007, 0.007, 0.007]),
                pos=s_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.2, 0.2, 0.95])
            )
            sim.viewer.user_scn.ngeom += 1


def select_best_target(detected_objs: List[Dict]) -> Optional[int]:
    """検出物体から最近傍の対象を選定。配置済みエリアや連続失敗上限の物体は除外"""
    if not detected_objs:
        return None

    best_idx = None
    min_dist_to_base = float('inf')

    for i, obj in enumerate(detected_objs):
        x, y = obj["phys_xy"]
        dist_to_place = math.hypot(x - PLACE_X_MM, y - PLACE_Y_MM)
        if dist_to_place < 40.0:
            continue

        obj_key = f"{int(round(x / 30.0))}_{int(round(y / 30.0))}"
        if OBJECT_FAIL_HISTORY.get(obj_key, 0) >= MAX_RETRIES_PER_OBJECT:
            continue

        dist_to_base = math.hypot(x, y)
        if dist_to_base < min_dist_to_base:
            min_dist_to_base = dist_to_base
            best_idx = i

    return best_idx


def main():
    global REQ_QUIT, REQ_PAUSE, REQ_SAVE_BG, CURRENT_GRASP_TCP_MARKERS

    print("==================================================")
    print(" 🤖 SO-ARM100 完全自律 Pick & Place システム")
    print("==================================================")
    print("【操作】")
    print("  [SPACE]  : 自動実行の一時停止 / 再開")
    print("  [B]      : 背景差分更新")
    print("  [Q/ESC]  : 安全停止して終了")
    print("--------------------------------------------------")

    controller = None
    if HAS_HARDWARE_MODULE:
        try:
            controller = BusServoController()
            if controller.connect():
                print("✅ 実機サーボコントローラに接続成功しました。")
            else:
                controller = None
        except Exception as e:
            print(f"⚠️ 実機接続スキップ: {e}")
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
        print("🤖 実機をホーム姿勢へ初期化中...")
        executor.move_to_home_and_wait(home_rad)

    # MuJoCo の物体スロット初期化
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
        qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[bid]] if bid != -1 else None
        slot_info.append({"bid": bid, "gid": gid, "qpos_adr": qadr})

    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(5):
        cap.read()
        time.sleep(0.04)

    detector = TabletopDetector(projector)
    cv2.namedWindow("Autonomous Pick & Place", cv2.WINDOW_AUTOSIZE)

    state = "IDLE"
    target_obj_cache = None
    stable_detect_count = 0

    try:
        while sim.is_running() and not REQ_QUIT:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=500, out_h=500)
            if warped is None:
                fallback_disp = cv2.resize(frame, (500, 500))
                cv2.putText(fallback_disp, "Searching ArUco Markers...", (30, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Autonomous Pick & Place", fallback_disp)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                continue

            if REQ_SAVE_BG:
                detector.update_background(warped)
                REQ_SAVE_BG = False

            detected_objs, _ = detector.detect_objects(warped)
            annotated = detector.draw_annotations(warped, detected_objs, target_idx=None)

            # ------------------------------------------------------------------
            # 🖼️ MuJoCo リアルタイム描画同期 (ArUco マーカー & 検出物体)
            # ------------------------------------------------------------------
            draw_markers(sim, projector, CURRENT_GRASP_TCP_MARKERS)

            for i in range(MAX_SLOTS):
                sinfo = slot_info[i]
                qadr, gid = sinfo["qpos_adr"], sinfo["gid"]
                if qadr is None or gid == -1:
                    continue

                if i < len(detected_objs):
                    obj = detected_objs[i]
                    x_mm, y_mm = obj["phys_xy"]
                    major_mm, minor_mm = obj["size_mm"]

                    # MuJoCo 座標系 [-Y, -X, Z] へのマッピング
                    sim.data.qpos[qadr:qadr + 3] = [-y_mm / 1000.0, -x_mm / 1000.0, HALF_Z]
                    sim.data.qpos[qadr + 3:qadr + 7] = euler_yaw_to_quat(math.radians(-obj["angle_deg"]))
                    sim.model.geom_size[gid] = [max(0.005, major_mm / 2000.0), max(0.005, minor_mm / 2000.0), HALF_Z]
                else:
                    # 検出されていないスロットは机の下へ隠す
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            # ------------------------------------------------------------------
            # 自律ステートマシン
            # ------------------------------------------------------------------
            if not REQ_PAUSE:
                if state == "IDLE":
                    CURRENT_GRASP_TCP_MARKERS = None
                    best_target_idx = select_best_target(detected_objs)
                    if best_target_idx is not None:
                        stable_detect_count += 1
                        if stable_detect_count >= 3:
                            target_obj_cache = detected_objs[best_target_idx]
                            print(f"\n🎯 把持対象 #{best_target_idx} を自動選定: ({target_obj_cache['phys_xy'][0]:.1f}, {target_obj_cache['phys_xy'][1]:.1f})")
                            state = "PLAN_AND_EXECUTE"
                            stable_detect_count = 0
                    else:
                        stable_detect_count = 0

                elif state == "PLAN_AND_EXECUTE":
                    x_mm, y_mm = target_obj_cache["phys_xy"]
                    major_mm, minor_mm = target_obj_cache["size_mm"]
                    angle_deg = target_obj_cache["angle_deg"]
                    obj_key = f"{int(round(x_mm / 30.0))}_{int(round(y_mm / 30.0))}"
                    fail_count = OBJECT_FAIL_HISTORY.get(obj_key, 0)

                    # 6段階リトライ戦略
                    cur_major = MANUAL_OFFSET_MAJOR_MM
                    cur_minor = MANUAL_OFFSET_MINOR_MM
                    cur_z = PICK_Z_MM

                    if fail_count == 1:
                        cur_z = max(0.0, cur_z - 2.0)
                        print(f"   🔄 [リトライ 1/5] 深掘りアプローチ (Z: {cur_z:.1f}mm)")
                    elif fail_count == 2:
                        jitter_major = random.uniform(-4.0, 4.0)
                        cur_major += 15.0 + jitter_major
                        cur_z = max(0.0, cur_z - 1.5)
                        print(f"   🔄 [リトライ 2/5] 長手(+)シフト (Major: {cur_major:+.1f}mm, Z: {cur_z:.1f}mm)")
                    elif fail_count == 3:
                        jitter_major = random.uniform(-4.0, 4.0)
                        cur_major -= 15.0 + jitter_major
                        cur_z = max(0.0, cur_z - 1.5)
                        print(f"   🔄 [リトライ 3/5] 長手(-)シフト (Major: {cur_major:+.1f}mm, Z: {cur_z:.1f}mm)")
                    elif fail_count == 4:
                        cur_minor += 8.0
                        cur_z = max(0.0, cur_z - 2.0)
                        print(f"   🔄 [リトライ 4/5] 短手引き込み深掘り (Minor: {cur_minor:+.1f}mm, Z: {cur_z:.1f}mm)")
                    elif fail_count >= 5:
                        jitter_major = random.uniform(-10.0, 10.0)
                        jitter_minor = random.uniform(-6.0, 6.0)
                        cur_major += jitter_major
                        cur_minor += jitter_minor
                        cur_z = max(0.0, cur_z - 1.5)
                        print(f"   🔄 [リトライ 5/5] 広角ジッター探索 (Major: {cur_major:+.1f}mm, Minor: {cur_minor:+.1f}mm)")

                    # 把持マーカーを MuJoCo 上に表示更新
                    r_phys_m = math.hypot(x_mm, y_mm) / 1000.0
                    theta_phys_rad = math.atan2(y_mm, x_mm)
                    sag_offset_m = calculate_sag_compensation(r_phys_m, theta_phys_rad)
                    CURRENT_GRASP_TCP_MARKERS = {
                        "target_tcp": np.array([-y_mm / 1000.0, -x_mm / 1000.0, cur_z / 1000.0]),
                        "sag_tcp": np.array([-y_mm / 1000.0, -x_mm / 1000.0, (cur_z / 1000.0) - sag_offset_m])
                    }

                    # 1. Pick 側 IK
                    ik_grasp, ik_pick_wp, _ = solve_ik_tabletop_grasp(
                        x_phys_mm=x_mm,
                        y_phys_mm=y_mm,
                        z_phys_mm=cur_z,
                        angle_deg=angle_deg,
                        obj_thickness_mm=minor_mm,
                        gripper_open_rad=GRIPPER_OPEN_RAD,
                        enable_sag_compensation=True,
                        offset_major_mm=cur_major,
                        offset_minor_mm=cur_minor,
                        verbose=False
                    )

                    # 2. Place 側 IK
                    ik_place_target, ik_place_wp, _ = solve_ik_tabletop_place(
                        x_phys_mm=PLACE_X_MM,
                        y_phys_mm=PLACE_Y_MM,
                        z_phys_mm=PLACE_Z_MM,
                        place_angle_deg=PLACE_ANGLE_DEG,
                        enable_sag_compensation=True,
                        verbose=False
                    )

                    if ik_grasp is None or ik_pick_wp is None or ik_place_target is None or ik_place_wp is None:
                        print("⚠️ IK 解が見つかりません。カウントを増やして別物体へ切り替えます。")
                        OBJECT_FAIL_HISTORY[obj_key] = fail_count + 1
                        state = "IDLE"
                    else:
                        print("🚀 自律 Pick & Place シーケンスを開始...")
                        pick_wp_open     = dict(ik_pick_wp);      pick_wp_open[6]     = GRIPPER_OPEN_RAD
                        pick_grasp_open  = dict(ik_grasp);        pick_grasp_open[6]  = GRIPPER_OPEN_RAD
                        pick_grasp_close = dict(ik_grasp);        pick_grasp_close[6] = GRIPPER_CLOSE_RAD
                        pick_wp_close    = dict(ik_pick_wp);      pick_wp_close[6]    = GRIPPER_CLOSE_RAD

                        place_wp_close   = dict(ik_place_wp);     place_wp_close[6]   = GRIPPER_CLOSE_RAD
                        place_land_close = dict(ik_place_target); place_land_close[6] = GRIPPER_CLOSE_RAD
                        place_land_open  = dict(ik_place_target); place_land_open[6]  = GRIPPER_OPEN_RAD
                        place_wp_open    = dict(ik_place_wp);     place_wp_open[6]    = GRIPPER_OPEN_RAD

                        # Pick 動作
                        executor.move_to_rad(pick_wp_open, duration_sec=1.2, send_to_real=True)
                        executor.move_to_rad(pick_grasp_open, duration_sec=0.8, send_to_real=True)
                        executor.move_to_rad(pick_grasp_close, duration_sec=0.5, send_to_real=True)
                        executor.move_to_rad(pick_wp_close, duration_sec=0.8, send_to_real=True)

                        # 把持判定
                        time.sleep(0.2)
                        is_grasped = True
                        if executor.is_real_connected and hasattr(controller, 'driver') and controller.driver:
                            grip_raw = None
                            for _ in range(4):
                                grip_raw = controller.driver.read_position(6)
                                if grip_raw is not None:
                                    break
                                time.sleep(0.05)

                            if grip_raw is not None:
                                raw_diff = abs(grip_raw - GRIPPER_CLOSED_RAW)
                                print(f"   🔍 [把持判定] 現在爪 Raw: {grip_raw} (完全閉止値 1889 との差: {raw_diff} count)")
                                if raw_diff <= EMPTY_GRASP_TOLERANCE_RAW:
                                    is_grasped = False
                            else:
                                print("   ⚠️ サーボ ID 6 の位置読み取りに失敗しました。")
                                is_grasped = False

                        if is_grasped:
                            OBJECT_FAIL_HISTORY.pop(obj_key, None)
                            executor.move_to_rad(place_wp_close, duration_sec=1.5, send_to_real=True)
                            executor.move_to_rad(place_land_close, duration_sec=0.8, send_to_real=True)
                            executor.move_to_rad(place_land_open, duration_sec=0.5, send_to_real=True)
                            executor.move_to_rad(place_wp_open, duration_sec=0.8, send_to_real=True)
                            print("✨ 配置完了！")

                            CURRENT_GRASP_TCP_MARKERS = None
                            executor.move_to_home_and_wait(home_rad)
                            time.sleep(0.3)
                            state = "IDLE"

                        else:
                            new_fail_count = fail_count + 1
                            OBJECT_FAIL_HISTORY[obj_key] = new_fail_count
                            print(f"⚠️ 把持空振りを検知 (試行 {new_fail_count}/{MAX_RETRIES_PER_OBJECT})")

                            executor.move_to_rad(pick_wp_open, duration_sec=0.4, send_to_real=True)

                            if new_fail_count < MAX_RETRIES_PER_OBJECT:
                                print("⚡ Home を経由せず、上空から即座にオフセット摂動をかけて再試行します...")
                                time.sleep(0.2)
                                state = "PLAN_AND_EXECUTE"
                            else:
                                print("🛑 連続失敗上限に達しました。Home へ戻り、別の物体へ切り替えます。")
                                CURRENT_GRASP_TCP_MARKERS = None
                                executor.move_to_home_and_wait(home_rad)
                                time.sleep(0.5)
                                state = "IDLE"

            # 画面ステータス表示
            status_text = f"State: {state} | Objs: {len(detected_objs)} | [SPACE] Pause/Resume"
            if REQ_PAUSE:
                status_text = "PAUSED (Press [SPACE] to Resume)"
            cv2.putText(annotated, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Autonomous Pick & Place", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key == ord(' '):
                REQ_PAUSE = not REQ_PAUSE
            elif key in [ord('b'), ord('B')]:
                REQ_SAVE_BG = True

    finally:
        if controller is not None:
            controller.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()