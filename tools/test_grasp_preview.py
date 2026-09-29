"""
==============================================================================
机上物体把持プレビューテストツール (tools/test_grasp_preview.py)
【モジュール統合リファクタリング版】
==============================================================================
【目的】
1. core/tabletop_detector.py により机上の全物体（位置・傾き・ミリ寸法）を検出。
2. 数字キー [0]〜[9] で選択したターゲットに対し、
   core/kinematics.py の solve_ik_tabletop_grasp を実行。
3. 手首ロール角アライメントと非対称爪（固定爪干渉回避）オフセットが反映された
   把持姿勢を MuJoCo 空間内でアニメーションプレビュー（Home -> Waypoint -> Grasp）。

【操作】
  - [0]〜[9] : 把持対象物体の選択
  - [G]      : 選択中ターゲットへの把持シーケンスを実行
  - [H]      : ホーム姿勢に戻す
  - [B]      : 机面背景の記憶 (背景差分更新)
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
from typing import Dict

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


def interpolate_motion(sim, target_rad: Dict[int, float], steps: int = 40, delay_sec: float = 0.02):
    """現在の関節角度から目標姿勢へスムーズに補間アニメーション"""
    start_qpos = np.copy(sim.data.qpos[:6])
    target_qpos = np.copy(start_qpos)

    for sid, rad in target_rad.items():
        if 1 <= sid <= 6:
            target_qpos[sid - 1] = rad

    for s in range(1, steps + 1):
        ratio = s / float(steps)
        t = ratio * ratio * (3 - 2 * ratio)  # Smooth-step イージング
        current = (1.0 - t) * start_qpos + t * target_qpos
        sim.data.qpos[:6] = current

        mujoco.mj_forward(sim.model, sim.data)
        if sim.viewer is not None:
            sim.viewer.sync()
        time.sleep(delay_sec)


def main():
    global SELECTED_TARGET_IDX, REQ_EXEC_GRASP, REQ_GO_HOME, REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT

    print("==================================================")
    print(" 🤖 非対称爪・手首ロール把持プレビュー (モジュール版)")
    print("==================================================")
    print("【操作】")
    print("  [0]〜[9] : 把持対象物体の選択")
    print("  [G]      : 選択対象への把持シーケンスを実行")
    print("  [H]      : ホーム姿勢に戻す")
    print("  [B]      : 背景画像を記憶 (差分更新)")
    print("  [SPACE]  : マーカー正射影の再計算")
    print("  [Q/ESC]  : 終了")
    print("--------------------------------------------------")

    # 1. MuJoCo ビューア初期化
    sim = MujocoSimViewer()
    home_rad = get_home_radians()
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass

    sim.viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=custom_sim_key_callback
    )

    # 2. カメラ & 検出モジュール初期化
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

            # 背景更新要求
            if REQ_SAVE_BG:
                detector.update_background(warped)
                REQ_SAVE_BG = False

            # キャリブレーション更新要求
            if REQ_RECALIB:
                projector.update_homography(frame)
                REQ_RECALIB = False
                print("🔄 キャリブレーションを更新しました。")

            # 1. 物体検出 (幾何計測)
            detected_objs, _ = detector.detect_objects(warped)

            # 2. アノテーション描画
            annotated = detector.draw_annotations(warped, detected_objs, target_idx=SELECTED_TARGET_IDX)

            # 3. ホーム姿勢復帰
            if REQ_GO_HOME:
                REQ_GO_HOME = False
                print("🏠 ホーム姿勢へ戻ります...")
                interpolate_motion(sim, home_rad, steps=30)

            # 4. 把持プレビュー実行
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

                    # 机面から物体厚みの半分（約8mm）を把持点とする
                    z_grasp_mm = 8.0

                    # 逆運動学の解決（手首ロール・非対称爪オフセット自動計算）
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
                        print("❌ 警告: 到達不能または特異点のため IK 解が算出できませんでした。")
                    else:
                        print(f"✅ IK 解決成功! (採用進入ピッチ角: {adopted_pitch:.1f}°)")
                        print(f"   目標手首ロール角 (ID5): {math.degrees(ik_grasp[5]):.1f}°")

                        # [1] 上空アプローチ点 (Waypoint) へ移動
                        print("▶️ [1/3] 上空アプローチ点へ進入中...")
                        ik_wp[6] = GRIPPER_OPEN_RAD
                        interpolate_motion(sim, ik_wp, steps=35)
                        time.sleep(0.3)

                        # [2] 垂直下降 (Grasp 位置へ到達)
                        print("▶️ [2/3] 把持位置へ下降中...")
                        ik_grasp[6] = GRIPPER_OPEN_RAD
                        interpolate_motion(sim, ik_grasp, steps=25)
                        time.sleep(0.3)

                        # [3] 爪を閉じて把持
                        print("▶️ [3/3] 爪を閉じて把持中...")
                        ik_grasp_closed = dict(ik_grasp)
                        ik_grasp_closed[6] = GRIPPER_CLOSE_RAD
                        interpolate_motion(sim, ik_grasp_closed, steps=15)
                        time.sleep(0.6)

                        # [4] 上空退避 (持ち上げ)
                        print("▶️ 持ち上げ退避中...")
                        ik_wp_closed = dict(ik_wp)
                        ik_wp_closed[6] = GRIPPER_CLOSE_RAD
                        interpolate_motion(sim, ik_wp_closed, steps=25)
                        time.sleep(0.5)

                        print("✨ 把持シーケンス完了。[H] キーでホームに戻せます。")
                else:
                    print(f"⚠️ 指定されたインデックス #{SELECTED_TARGET_IDX} の物体が見つかりません。")

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