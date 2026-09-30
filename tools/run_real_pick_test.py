"""
==============================================================================
実機把持テスト ＆ プレビュー実行ツール
(tools/run_real_pick_test.py)
==============================================================================
【操作フロー】
1. カメラから物体を認識し、MuJoCo 上に実寸直方体・ArUco・把持マーカーを同期。
2. [0]〜[9] で把持対象を選択。
3. [P] キーを押すと、MuJoCo 上で一連の動作 (上空 ➔ 把持 ➔ 持ち上げ) をプレビュー。
4. プレビュー後、ターミナルで `y` を入力すると、実機サーボが同一の軌道で物体を持ち上げます。
5. [H] でホーム姿勢へ復帰。
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

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.tabletop_detector import TabletopDetector
from core.kinematics import (
    get_home_radians,
    solve_ik_tabletop_grasp,
    calculate_sag_compensation,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)
from core.trajectory_executor import TrajectoryExecutor

try:
    from core.bus_servo_controller import BusServoController
    HAS_HARDWARE_MODULE = True
except ImportError:
    HAS_HARDWARE_MODULE = False

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

SELECTED_TARGET_IDX = 0
REQ_PREVIEW = False
REQ_GO_HOME = False
REQ_SAVE_BG = False
REQ_QUIT = False

CURRENT_GRASP_TCP_MARKERS: Optional[Dict[str, np.ndarray]] = None


def sim_key_callback(keycode: int):
    global SELECTED_TARGET_IDX, REQ_PREVIEW, REQ_GO_HOME, REQ_SAVE_BG, REQ_QUIT
    if 48 <= keycode <= 57:
        SELECTED_TARGET_IDX = keycode - 48
        print(f"\n🎯 ターゲット #{SELECTED_TARGET_IDX} を選択しました。")
    elif keycode in (80, 112):  # P, p (Preview)
        REQ_PREVIEW = True
    elif keycode in (72, 104):  # H, h (Home)
        REQ_GO_HOME = True
    elif keycode in (66, 98):  # B, b (Background)
        REQ_SAVE_BG = True
    elif keycode in (81, 113, 256):  # Q, ESC
        REQ_QUIT = True


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def draw_markers(sim, projector, markers_dict):
    if sim.viewer is None:
        return
    sim.viewer.user_scn.ngeom = 0

    # ArUco マーカー
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

    # TCP マーカー
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


def main():
    global SELECTED_TARGET_IDX, REQ_PREVIEW, REQ_GO_HOME, REQ_SAVE_BG, REQ_QUIT
    global CURRENT_GRASP_TCP_MARKERS

    print("==================================================")
    print(" 🦾 SO-ARM100 実機把持 (Pick) 統合テスト")
    print("==================================================")
    print("【操作】")
    print("  [0]〜[9] : 把持対象物体の選択")
    print("  [P]      : 選択対象への把持シーケンスをプレビュー")
    print("  [H]      : ホーム姿勢に戻す")
    print("  [B]      : 背景差分更新")
    print("  [Q/ESC]  : 終了")
    print("--------------------------------------------------")

    # 1. 実機コントローラ初期化 (接続を試行)
    controller = None
    if HAS_HARDWARE_MODULE:
        try:
            controller = BusServoController()
            if controller.connect():
                print("✅ 実機サーボコントローラに接続成功しました。")
            else:
                print("⚠️ 実機シリアルポートが見つかりません。シミュレーション単体モードで起動します。")
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

    # 実機接続時は実機をまずホーム姿勢へ
    if executor.is_real_connected:
        print("🤖 実機をホーム姿勢へ初期化中...")
        executor.move_to_home_and_wait(home_rad)  # 👉 move_to_home_and_wait に変更

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
        qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[bid]] if bid != -1 else None
        slot_info.append({"bid": bid, "gid": gid, "qpos_adr": qadr})

    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = TabletopDetector(projector)
    cv2.namedWindow("Real Pick Vision Tracker")

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

            detected_objs, _ = detector.detect_objects(warped)
            annotated = detector.draw_annotations(warped, detected_objs, target_idx=SELECTED_TARGET_IDX)

            # MuJoCo 描画更新
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

                    sim.data.qpos[qadr:qadr + 3] = [-y_mm / 1000.0, -x_mm / 1000.0, HALF_Z]
                    sim.data.qpos[qadr + 3:qadr + 7] = euler_yaw_to_quat(math.radians(-obj["angle_deg"]))
                    sim.model.geom_size[gid] = [max(0.005, major_mm / 2000.0), max(0.005, minor_mm / 2000.0), HALF_Z]
                else:
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]

            # [H] ホーム復帰
            if REQ_GO_HOME:
                REQ_GO_HOME = False
                CURRENT_GRASP_TCP_MARKERS = None
                executor.move_to_home_and_wait(home_rad)  # 👉 統一

            # [P] 実機シーケンス終了後のホーム復帰
                if confirm == 'y':
                    print("🦾 実機把持シーケンスを開始します...")
                    executor.execute_waypoints(pick_sequence, send_to_real=True)
                    time.sleep(0.8)
                    # 👉 到達監視＆ID4先行引き上げ付きで Home 復帰
                    executor.move_to_home_and_wait(home_rad)
                    print("✨ 実機把持テストが完了しました！")

            # [P] 把持プレビュー ＆ 実機実行確認
            if REQ_PREVIEW:
                REQ_PREVIEW = False
                if SELECTED_TARGET_IDX < len(detected_objs):
                    target_obj = detected_objs[SELECTED_TARGET_IDX]
                    x_mm, y_mm = target_obj["phys_xy"]
                    major_mm, minor_mm = target_obj["size_mm"]
                    angle_deg = target_obj["angle_deg"]

                    print("\n" + "=" * 55)
                    print(f"🎯 ターゲット #{SELECTED_TARGET_IDX} の把持計画を計算:")
                    print(f"   位置: X={x_mm:.1f}mm, Y={y_mm:.1f}mm | 傾き: {angle_deg:+.1f}°")

                    z_grasp_mm = 2.0

                    r_phys_m = math.hypot(x_mm, y_mm) / 1000.0
                    theta_phys_rad = math.atan2(y_mm, x_mm)
                    sag_offset_m = calculate_sag_compensation(r_phys_m, theta_phys_rad)

                    CURRENT_GRASP_TCP_MARKERS = {
                        "target_tcp": np.array([-y_mm / 1000.0, -x_mm / 1000.0, z_grasp_mm / 1000.0]),
                        "sag_tcp": np.array([-y_mm / 1000.0, -x_mm / 1000.0, (z_grasp_mm / 1000.0) - sag_offset_m])
                    }

                    ik_grasp, ik_wp, adopted_pitch = solve_ik_tabletop_grasp(
                        x_phys_mm=x_mm,
                        y_phys_mm=y_mm,
                        z_phys_mm=z_grasp_mm,
                        angle_deg=angle_deg,
                        obj_thickness_mm=minor_mm,
                        gripper_open_rad=GRIPPER_OPEN_RAD,
                        enable_sag_compensation=True
                    )

                    if ik_grasp is None or ik_wp is None:
                        print("❌ IK 解の算出に失敗しました。")
                    else:
                        print(f"✅ IK 解決成功 (進入ピッチ: {adopted_pitch:.1f}°, 手首ロール: {math.degrees(ik_grasp[5]):.1f}°)")

                        # ウェイポイントシーケンス構築
                        ik_wp_open = dict(ik_wp); ik_wp_open[6] = GRIPPER_OPEN_RAD
                        ik_grasp_open = dict(ik_grasp); ik_grasp_open[6] = GRIPPER_OPEN_RAD
                        ik_grasp_close = dict(ik_grasp); ik_grasp_close[6] = GRIPPER_CLOSE_RAD
                        ik_wp_close = dict(ik_wp); ik_wp_close[6] = GRIPPER_CLOSE_RAD

                        pick_sequence = [
                            (ik_wp_open, 1.2, "上空アプローチ"),
                            (ik_grasp_open, 0.8, "把持点へ降下"),
                            (ik_grasp_close, 0.5, "爪を閉じて把持"),
                            (ik_wp_close, 0.8, "物体を持ち上げ退避")
                        ]

                        # 👉 【変更点 2】 プレビュー時は send_to_real=False で MuJoCo だけ動かす
                        print("\n🎬 [シミュレーション] プレビュー再生中...")
                        executor.execute_waypoints(pick_sequence, send_to_real=False)

                        # 実機実行ゲート
                        if executor.is_real_connected:
                            print("\n" + "!" * 55)
                            confirm = input("⚠️ 実機サーボでこの動作を実行しますか？ (y/N): ").strip().lower()
                            if confirm == 'y':
                                print("🦾 実機把持シーケンスを開始します...")
                                # 👉 本番実行時のみ send_to_real=True で実機を動かす
                                executor.execute_waypoints(pick_sequence, send_to_real=True)
                                time.sleep(1.0)
                                print("🏠 実機をホーム姿勢へ戻します...")
                                executor.move_to_rad(home_rad, duration_sec=5.0, send_to_real=True)
                                print("✨ 実機把持テストが完了しました！")
                            else:
                                print("🛡️ 実機実行をキャンセルしました。")
                        else:
                            print("💡 (実機未接続のためシミュレーションプレビューのみ実行しました)")
                else:
                    print(f"⚠️ 指定された #{SELECTED_TARGET_IDX} の物体が見つかりません。")

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            status = f"Target: #{SELECTED_TARGET_IDX} | Press [P] to Preview/Execute, [H] to Home"
            cv2.putText(annotated, status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Real Pick Vision Tracker", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif 48 <= key <= 57:
                SELECTED_TARGET_IDX = key - 48
            elif key in [ord('p'), ord('P')]:
                REQ_PREVIEW = True
            elif key in [ord('h'), ord('H')]:
                REQ_GO_HOME = True
            elif key in [ord('b'), ord('B')]:
                REQ_SAVE_BG = True

    finally:
        if controller is not None:
            controller.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()