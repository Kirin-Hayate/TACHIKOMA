"""
==============================================================================
机上物体把持プレビュー＆デジタルツイン同期テスト
(tools/test_grasp_preview.py)
==============================================================================
【機能】
1. カメラ画像から机上の物体を検出。
2. MuJoCo 空間上に検出物体を「直方体 (寸法・向き反映)」としてリアルタイム同期。
3. 選択中ターゲット (#0〜#9) の直上にマーカーとターゲット表示を重畳。
4. [G] キーで把持シーケンスを実行し、爪の挟み込み角度や位置アライメントを目視検証。

【操作】
  - [0]〜[9] : 把持対象物体の選択
  - [G]      : 選択対象への把持シーケンスを実行
  - [H]      : ホーム姿勢に戻す
  - [B]      : 背景画像を記憶 (差分更新)
  - [SPACE]  : マーカー正射影の再計算
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
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.tabletop_detector import TabletopDetector
from core.kinematics import (
    get_home_radians,
    solve_ik_tabletop_grasp,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

SELECTED_TARGET_IDX = 0
REQ_EXEC_GRASP = False
REQ_GO_HOME = False
REQ_SAVE_BG = False
REQ_RECALIB = False
REQ_QUIT = False


def custom_sim_key_callback(keycode: int):
    global SELECTED_TARGET_IDX, REQ_EXEC_GRASP, REQ_GO_HOME, REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT
    if 48 <= keycode <= 57:  # 0〜9
        SELECTED_TARGET_IDX = keycode - 48
        print(f"\n🎯 把持ターゲットを #{SELECTED_TARGET_IDX} に切り替えました。")
    elif keycode in (71, 103):  # G, g
        REQ_EXEC_GRASP = True
    elif keycode in (72, 104):  # H, h
        REQ_GO_HOME = True
    elif keycode in (66, 98):  # B, b
        REQ_SAVE_BG = True
    elif keycode == 32:  # Space
        REQ_RECALIB = True
    elif keycode in (81, 113, 256):  # Q, q, ESC
        REQ_QUIT = True


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def interpolate_motion(sim, target_rad: Dict[int, float], steps: int = 35, delay_sec: float = 0.02):
    """現在の関節角度から目標姿勢へスムーズに補間アニメーション"""
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

        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            lbl_pos = np.array([mj_x, mj_y, 0.025], dtype=np.float64)
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=np.zeros(3),
                pos=lbl_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.3, 0.8, 1.0, 1.0], dtype=np.float32)
            )
            sim.viewer.user_scn.geoms[ng].label = f"ArUco #{marker_idx}".encode("utf-8")
            sim.viewer.user_scn.ngeom += 1


def main():
    global SELECTED_TARGET_IDX, REQ_EXEC_GRASP, REQ_GO_HOME, REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT

    print("==================================================")
    print(" 🤖 把持プレビュー ＆ 物体 3D デジタルツイン同期")
    print("==================================================")
    print("【操作】")
    print("  [0]〜[9] : 把持対象物体の選択")
    print("  [G]      : 選択対象への把持シーケンスを実行")
    print("  [H]      : ホーム姿勢に戻す")
    print("  [B]      : 背景画像を記憶 (差分更新)")
    print("  [SPACE]  : マーカー正射影の再計算")
    print("  [Q/ESC]  : 終了")
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
    cv2.namedWindow("Grasp Preview Vision Tracker")

    try:
        while sim.is_running() and not REQ_QUIT:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=500, out_h=500)
            if warped is None:
                cv2.imshow("Grasp Preview Vision Tracker", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                    break
                continue

            if REQ_SAVE_BG:
                detector.update_background(warped)
                REQ_SAVE_BG = False

            if REQ_RECALIB:
                projector.update_homography(frame)
                REQ_RECALIB = False
                print("🔄 キャリブレーションを更新しました。")

            # 1. 物体検出
            detected_objs, _ = detector.detect_objects(warped)

            # 2. OpenCV 画面描画
            annotated = detector.draw_annotations(warped, detected_objs, target_idx=SELECTED_TARGET_IDX)

            # 3. MuJoCo 空間への動的反映 (直方体とマーカー)
            if sim.viewer is not None:
                sim.viewer.user_scn.ngeom = 0

            draw_aruco_markers_in_mujoco(sim, projector, marker_size_m=0.04)

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

                    # 位置と向きの更新
                    sim.data.qpos[qadr:qadr + 3] = [mj_x, mj_y, HALF_Z]
                    sim.data.qpos[qadr + 3:qadr + 7] = euler_yaw_to_quat(yaw_rad)

                    # 直方体サイズの動的反映 (half-size)
                    half_x = max(0.005, (major_mm / 1000.0) / 2.0)
                    half_y = max(0.005, (minor_mm / 1000.0) / 2.0)
                    sim.model.geom_size[gid] = [half_x, half_y, HALF_Z]

                    # 選択ターゲット強調ラベル
                    is_target = (i == SELECTED_TARGET_IDX)
                    label_text = f"#{i} TARGET" if is_target else f"#{i}"
                    rgba = np.array([1.0, 0.3, 0.3, 1.0]) if is_target else np.array([1.0, 1.0, 0.2, 1.0])

                    if sim.viewer is not None and sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
                        ngeom = sim.viewer.user_scn.ngeom
                        label_pos = np.array([mj_x, mj_y, HALF_Z + 0.035], dtype=np.float64)
                        mujoco.mjv_initGeom(
                            sim.viewer.user_scn.geoms[ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.zeros(3),
                            pos=label_pos,
                            mat=np.eye(3).flatten(),
                            rgba=rgba
                        )
                        sim.viewer.user_scn.geoms[ngeom].label = label_text.encode("utf-8")
                        sim.viewer.user_scn.ngeom += 1
                else:
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]
                    sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

            # 4. ホーム復帰
            if REQ_GO_HOME:
                REQ_GO_HOME = False
                print("🏠 ホーム姿勢へ戻ります...")
                interpolate_motion(sim, home_rad, steps=25)

            # 5. 把持プレビュー実行
            if REQ_EXEC_GRASP:
                REQ_EXEC_GRASP = False
                if SELECTED_TARGET_IDX < len(detected_objs):
                    target_obj = detected_objs[SELECTED_TARGET_IDX]
                    x_mm, y_mm = target_obj["phys_xy"]
                    major_mm, minor_mm = target_obj["size_mm"]
                    angle_deg = target_obj["angle_deg"]

                    print("\n==================================================")
                    print(f"🎯 ターゲット #{SELECTED_TARGET_IDX} の把持計画を計算:")
                    print(f"   位置: X={x_mm:.1f}mm, Y={y_mm:.1f}mm")
                    print(f"   寸法: {major_mm:.1f} x {minor_mm:.1f} mm")
                    print(f"   傾き: {angle_deg:+.1f}°")

                    z_grasp_mm = 14.0

                    ik_grasp, ik_wp, adopted_pitch = solve_ik_tabletop_grasp(
                        x_phys_mm=x_mm,
                        y_phys_mm=y_mm,
                        z_phys_mm=z_grasp_mm,
                        angle_deg=angle_deg,
                        obj_thickness_mm=minor_mm,
                        gripper_open_rad=GRIPPER_OPEN_RAD,
                        enable_sag_compensation=True,
                        verbose=True
                    )

                    if ik_grasp is None or ik_wp is None:
                        print("❌ 警告: IK 解が算出できませんでした。")
                    else:
                        print(f"✅ IK 解決成功! (進入ピッチ角: {adopted_pitch:.1f}°, 手首ロール ID5: {math.degrees(ik_grasp[5]):.1f}°)")

                        # 1. 上空アプローチ
                        print("▶️ [1/3] 上空アプローチ中...")
                        ik_wp[6] = GRIPPER_OPEN_RAD
                        interpolate_motion(sim, ik_wp, steps=35)
                        time.sleep(0.4)

                        # 2. 把持位置へ降下
                        print("▶️ [2/3] 把持位置へ降下中...")
                        ik_grasp[6] = GRIPPER_OPEN_RAD
                        interpolate_motion(sim, ik_grasp, steps=25)
                        time.sleep(0.5)

                        # 3. 把持 (爪を閉じる)
                        print("▶️ [3/3] 爪を閉じて把持中...")
                        ik_grasp_closed = dict(ik_grasp)
                        ik_grasp_closed[6] = GRIPPER_CLOSE_RAD
                        interpolate_motion(sim, ik_grasp_closed, steps=15)
                        time.sleep(0.6)

                        # 4. 持ち上げ退避
                        print("▶️ 持ち上げ退避中...")
                        ik_wp_closed = dict(ik_wp)
                        ik_wp_closed[6] = GRIPPER_CLOSE_RAD
                        interpolate_motion(sim, ik_wp_closed, steps=25)
                        time.sleep(0.5)

                        print("✨ 把持シーケンス完了。[H] キーでホームに戻せます。")
                else:
                    print(f"⚠️ 指定されたインデックス #{SELECTED_TARGET_IDX} の物体が見つかりません。")

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            status = f"Target: #{SELECTED_TARGET_IDX} | Press [G] to Grasp, [H] to Home"
            cv2.putText(annotated, status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Grasp Preview Vision Tracker", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif 48 <= key <= 57:
                SELECTED_TARGET_IDX = key - 48
            elif key in [ord('g'), ord('G')]:
                REQ_EXEC_GRASP = True
            elif key in [ord('h'), ord('H')]:
                REQ_GO_HOME = True
            elif key in [ord('b'), ord('B')]:
                REQ_SAVE_BG = True
            elif key == 32:
                REQ_RECALIB = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()