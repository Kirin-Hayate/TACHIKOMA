"""
==============================================================================
アプローチ手法 比較検証テスト (tests/test_ik_approach_comparison.py)
==============================================================================
【検証対象】
  1. ヌルスペース投影 (Null-space Projection):
     - 直接目標点へ向かうが、IKの余剰自由度で肘上げ・手先垂直姿勢へ引き込む。
  2. 経由姿勢アプローチ (Waypoint / Pre-Grasp Approach):
     - 目標点上空 (z + 50mm) の待機姿勢を経由してから、真下へ垂直降下する。

【操作方法】
  - [M] キー : モード切り替え (ヌルスペース ↔ 経由姿勢)
  - [Space]  : 一時停止 / 再開
  - [1]〜[9]  : 再生速度倍率の変更
==============================================================================
"""

import sys
import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

# プロジェクトルートのパス解決
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and os.path.basename(current_dir) != "tachikoma":
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config.joint_config import SERVO_IDS, JOINT_CONFIG
from core.kinematics import solve_ik_polar, raw_to_radian


def interpolate_positions(start_pos: dict, end_pos: dict, ratio: float) -> dict:
    """コサイン S 字補間による姿勢計算"""
    s_ratio = (1.0 - math.cos(ratio * math.pi)) / 2.0
    current = {}
    for sid in SERVO_IDS:
        s_val = start_pos.get(sid, JOINT_CONFIG[sid]["init"])
        e_val = end_pos.get(sid, JOINT_CONFIG[sid]["init"])
        current[sid] = int(s_val + s_ratio * (e_val - s_val))
    return current


def execute_trajectory(viewer, model, data, start_pos, target_pos, target_xyz, steps=35, speed=2.0, paused_flag_getter=None):
    """2点間の滑らかな補間移動とターゲット球の描画"""
    for step in range(1, steps + 1):
        if not viewer.is_running():
            break
        while paused_flag_getter and paused_flag_getter() and viewer.is_running():
            time.sleep(0.05)

        ratio = step / steps
        interp_pos = interpolate_positions(start_pos, target_pos, ratio)
        
        for i, sid in enumerate(SERVO_IDS):
            if i < model.nq:
                data.qpos[i] = raw_to_radian(sid, interp_pos[sid])
        mujoco.mj_forward(model, data)

        # 赤いターゲット球の描画
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=target_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.1, 0.1, 0.75]
        )
        viewer.user_scn.ngeom = 1
        viewer.sync()

        time.sleep(0.015 / speed)


def main():
    xml_path = os.path.join(current_dir, "assets", "so100_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    paused = False
    playback_speed = 2.0
    mode = "WAYPOINT"  # "NULLSPACE" または "WAYPOINT"

    def key_callback(keycode):
        nonlocal paused, playback_speed, mode
        if keycode == 32:  # Space
            paused = not paused
            print("\n[一時停止]" if paused else "\n[再開]")
        elif keycode in (77, 109):  # M, m
            mode = "WAYPOINT" if mode == "NULLSPACE" else "NULLSPACE"
            print(f"\n🔄 動作モード切替: 【{mode}】")
        elif 49 <= keycode <= 57:  # 1〜9
            playback_speed = float(keycode - 48)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")
        elif 321 <= keycode <= 329:  # テンキー 1〜9
            playback_speed = float(keycode - 320)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")

    # 把持テスト用のグリッド点 (床面 z=15mm を中心とした代表点)
    r_list = [0.20, 0.25, 0.29]
    theta_list = [-45.0, 0.0, 45.0]
    z_target = 0.015  # 把持高さ: 15mm

    test_points = [(r, th, z_target) for r in r_list for th in theta_list]
    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("=======================================================================")
    print(" 🚀 アプローチ手法 比較検証 (ヌルスペース vs 経由姿勢)")
    print(" 💡 [M] キーでいつでも手法を切り替えられます。")
    print("    - 現在のモード: " + mode)
    print("    - [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        idx = 0
        while viewer.is_running():
            r, theta, z = test_points[idx % len(test_points)]
            idx += 1

            th_rad = math.radians(-theta)
            target_xyz = np.array([r * math.sin(th_rad), -r * math.cos(th_rad), z])

            # 最終目標の IK 計算
            ik_target = solve_ik_polar(r=r, theta_deg=theta, z=z, prevent_penetration=True)

            if ik_target is None:
                print(f"[{mode}] r={r*1000:.0f}mm, θ={theta:+3.0f}° ➔ 到達不能 [--]")
                continue

            print(f"[{mode}] 目標: r={r*1000:.0f}mm, θ={theta:+3.0f}°, z={z*1000:.0f}mm へ進入開始")

            if mode == "NULLSPACE":
                # --- 手法1: ヌルスペース直接アプローチ ---
                # Home ➔ 直接目標点
                execute_trajectory(viewer, model, data, home_pos, ik_target, target_xyz, 
                                   steps=40, speed=playback_speed, paused_flag_getter=lambda: paused)
                time.sleep(0.3 / playback_speed)
                # 目標点 ➔ Home
                execute_trajectory(viewer, model, data, ik_target, home_pos, target_xyz, 
                                   steps=35, speed=playback_speed, paused_flag_getter=lambda: paused)

            else:
                # --- 手法2: 経由姿勢 (Pre-Grasp) アプローチ ---
                # 目標上空 (z + 50mm) の経由姿勢を計算
                ik_pre = solve_ik_polar(r=r, theta_deg=theta, z=z + 0.050, prevent_penetration=True)
                if ik_pre is None:
                    ik_pre = ik_target

                # 1. Home ➔ 上空待機点 (Pre-Grasp)
                execute_trajectory(viewer, model, data, home_pos, ik_pre, target_xyz, 
                                   steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)
                time.sleep(0.1 / playback_speed)
                
                # 2. 上空待機点 ➔ 真下へ垂直降下 (Pick)
                execute_trajectory(viewer, model, data, ik_pre, ik_target, target_xyz, 
                                   steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)
                time.sleep(0.3 / playback_speed)
                
                # 3. 把持点 ➔ 再び上空へ垂直退避
                execute_trajectory(viewer, model, data, ik_target, ik_pre, target_xyz, 
                                   steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)
                
                # 4. 上空待機点 ➔ Home 復帰
                execute_trajectory(viewer, model, data, ik_pre, home_pos, target_xyz, 
                                   steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)

            time.sleep(0.2 / playback_speed)


if __name__ == "__main__":
    main()