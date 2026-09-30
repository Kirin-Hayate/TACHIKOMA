"""
==============================================================================
自律連続 Pick & Place 自動化システム
(tools/run_auto_pick_and_place.py)
==============================================================================
【概要】
カメラ認識した机上の複数物体を自動選定し、人間の介入なしに
次々と指定エリア (Place Area) へ連続で仕分け・搬送する自律スクリプトです。

【主な機能】
1. 自律タスクスケジューラ:
   - 検出された物体群から、ロボット中心に近い安全な物体を自動選定してキューイング。
2. グリッパ把持成否フィードバック:
   - 爪を閉じた際、サーボ ID 6 の物理現在値を読み取り、空振り（把持失敗）を自動検知。
3. 2段階安全 Home 復帰 & S字加減速軌道:
   - 確立済みの安全ルーチンで連続稼働時のサーボ脱調・過負荷を防止。
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

PICK_Z_MM = 2.0                  # 把持高度 (机面 +2mm)

# 👉 【追加】リトライ制御およびブラックリスト設定
MAX_RETRIES_PER_OBJECT = 3       # 同一物体への最大リトライ回数 (超えたらスキップ)
OBJECT_FAIL_HISTORY: Dict[str, int] = {}  # 物体座標キーごとの失敗カウント記録

# 空振り判定閾値 (ID 6 の角度が完全に閉じた状態に近い場合は空振りと判定)
# 👉 【修正】Raw 生値による空振り判定閾値設定
# 爪完全閉止時の実測値: 1889
GRIPPER_CLOSED_RAW = 1889
# 閉止位置からこのカウント幅以内なら「空振り」と判定 (約 50〜70 カウント)
EMPTY_GRASP_TOLERANCE_RAW = 30

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

REQ_QUIT = False
REQ_PAUSE = False
REQ_SAVE_BG = False


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


def select_best_target(detected_objs: List[Dict]) -> Optional[int]:
    """
    検出物体から最近傍の対象を選定。
    配置済みエリア内、および連続失敗上限に達した物体は除外する。
    """
    if not detected_objs:
        return None

    best_idx = None
    min_dist_to_base = float('inf')

    for i, obj in enumerate(detected_objs):
        x, y = obj["phys_xy"]
        dist_to_place = math.hypot(x - PLACE_X_MM, y - PLACE_Y_MM)
        if dist_to_place < 40.0:
            continue

        # 👉 過去に失敗回数上限に達した物体はスキップ
        obj_key = f"{int(round(x / 30.0))}_{int(round(y / 30.0))}"
        if OBJECT_FAIL_HISTORY.get(obj_key, 0) >= MAX_RETRIES_PER_OBJECT:
            continue

        dist_to_base = math.hypot(x, y)
        if dist_to_base < min_dist_to_base:
            min_dist_to_base = dist_to_base
            best_idx = i

    return best_idx


def main():
    global REQ_QUIT, REQ_PAUSE, REQ_SAVE_BG

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

    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # カメラ露光安定化ウォームアップ
    for _ in range(5):
        cap.read()
        time.sleep(0.04)

    detector = TabletopDetector(projector)
    cv2.namedWindow("Autonomous Pick & Place", cv2.WINDOW_AUTOSIZE)

    # 状態管理
    state = "IDLE"  # IDLE -> DETECT -> EXECUTE -> VERIFY
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
            # 自律ステートマシン
            # ------------------------------------------------------------------
            if not REQ_PAUSE:
                if state == "IDLE":
                    best_target_idx = select_best_target(detected_objs)
                    if best_target_idx is not None:
                        stable_detect_count += 1
                        # 3フレーム連続で同一候補が捉えられたら動作開始（チャタリング防止）
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

                    # 👉 【戦略的リトライ ＆ ランダムジッターの算出】
                    cur_major = MANUAL_OFFSET_MAJOR_MM
                    cur_minor = MANUAL_OFFSET_MINOR_MM
                    cur_z = PICK_Z_MM

                    if fail_count == 1:
                        # リトライ1回目: 把持高度を深くし、物体側に少し寄せる
                        cur_z = max(0.0, cur_z - 2.0)
                        cur_major -= 5.0
                        print(f"   🔄 [リトライ 1] 深掘りアプローチ (Z: {cur_z:.1f}mm, Minor: {cur_minor:+.1f}mm)")
                    elif fail_count >= 2:
                        # リトライ2回目以降: 長手シフト + ランダム摂動 (ジッター)
                        jitter_major = random.uniform(-6.0, 6.0)
                        jitter_minor = random.uniform(-4.0, 4.0)
                        cur_major += (15.0 if fail_count % 2 == 0 else -15.0) + jitter_major
                        cur_minor += jitter_minor
                        cur_z = max(0.0, cur_z - 1.5)
                        print(f"   🔄 [リトライ {fail_count}] 摂動アプローチ (Major: {cur_major:+.1f}mm, Minor: {cur_minor:+.1f}mm, Z: {cur_z:.1f}mm)")

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

                        # ------------------------------------------------------
                        # 👉 把持判定 (安定化待機 ＆ 複数回ポーリング)
                        # ------------------------------------------------------
                        time.sleep(0.2)  # 把持後の振動安定化
                        is_grasped = True
                        
                        if executor.is_real_connected and hasattr(controller, 'driver') and controller.driver:
                            grip_raw = None
                            # 最大 4 回ポーリングして確実に生値を取得
                            for _ in range(4):
                                grip_raw = controller.driver.read_position(6)
                                if grip_raw is not None:
                                    break
                                time.sleep(0.05)

                            if grip_raw is not None:
                                raw_diff = abs(grip_raw - GRIPPER_CLOSED_RAW)
                                print(f"   🔍 [把持判定] 現在爪 Raw: {grip_raw} (完全閉止値 1889 との差: {raw_diff} count)")

                                # 完全閉止近傍 (±60 count 以内) なら空振りと判定
                                if raw_diff <= EMPTY_GRASP_TOLERANCE_RAW:
                                    is_grasped = False
                            else:
                                print("   ⚠️ サーボ ID 6 の位置読み取りに失敗しました (タイムアウト)。")
                                is_grasped = False  # 安全のため読み取り失敗時も空振りとみなしてリトライへ

                        if is_grasped:
                            # 👉 成功時: 失敗履歴を削除
                            OBJECT_FAIL_HISTORY.pop(obj_key, None)
                            executor.move_to_rad(place_wp_close, duration_sec=1.5, send_to_real=True)
                            executor.move_to_rad(place_land_close, duration_sec=0.8, send_to_real=True)
                            executor.move_to_rad(place_land_open, duration_sec=0.5, send_to_real=True)
                            executor.move_to_rad(place_wp_open, duration_sec=0.8, send_to_real=True)
                            print("✨ 配置完了！")
                        else:
                            # 👉 失敗時: 失敗カウントをインクリメント
                            OBJECT_FAIL_HISTORY[obj_key] = fail_count + 1
                            print(f"⚠️ 把持空振りを検知 (失敗回数: {OBJECT_FAIL_HISTORY[obj_key]}/{MAX_RETRIES_PER_OBJECT})")
                            if OBJECT_FAIL_HISTORY[obj_key] >= MAX_RETRIES_PER_OBJECT:
                                print("🛑 上限に達したため、この物体を一時スキップして次を優先します。")

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