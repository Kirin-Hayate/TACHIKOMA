"""
==============================================================================
広域グリッド点 3D 視覚的到達テスト (tests/test_ik_grid_visualizer.py)
==============================================================================
【役割】
指定したワークスペースのグリッド点 (r, theta, z) に「赤いターゲット球」を表示し、
Home 姿勢から各目標点へ滑らかにアプローチして爪先が重なるかを検証します。
キーボード操作:
  - [Space] : 一時停止 / 再開
  - [1]〜[9] : 再生速度倍率の変更 (1倍〜9倍速)
==============================================================================
"""

import sys
import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(base_dir, "assets", "so100_scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 状態変数
    paused = False
    playback_speed = 2.0  # サクサク確認できるようにデフォルト 2.0 倍速

    def key_callback(keycode):
        nonlocal paused, playback_speed
        if keycode == 32:  # Space
            paused = not paused
            print("\n[一時停止]" if paused else "\n[再開]")
        elif 49 <= keycode <= 57:  # 1〜9
            playback_speed = float(keycode - 48)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")
        elif 321 <= keycode <= 329:  # テンキー 1〜9
            playback_speed = float(keycode - 320)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")

    # テスト対象グリッド（計 50 点）
    r_list = [0.18, 0.22, 0.26, 0.30]             # 距離 4 段階
    theta_list = [-60.0, -30.0, 0.0, 30.0, 60.0]  # 旋回 5 段階
    z_list = [0.050, 0.015]                        # 高さ 2 段階 (上空 50mm -> 机面 15mm)

    grid_points = []
    for z in z_list:
        for r in r_list:
            for th in theta_list:
                grid_points.append((r, th, z))

    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("=======================================================================")
    print(f" 🚀 SO-ARM100 グリッド 3D 視覚検証 (全 {len(grid_points)} 点)")
    print(" 🔴 赤いターゲット球に向けてアームが接近します。")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    current_target_xyz = np.array([0.0, 0.0, -1.0])

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        for idx, (r, theta, z) in enumerate(grid_points, 1):
            if not viewer.is_running():
                break

            # 目標点のワールド直交座標を算出
            th_rad = math.radians(-theta)
            target_x = r * math.sin(th_rad)
            target_y = -r * math.cos(th_rad)
            target_z = z
            current_target_xyz = np.array([target_x, target_y, target_z])

            # IK 解の算出
            ik_targets = solve_ik_polar(r=r, theta_deg=theta, z=z, prevent_penetration=True)

            status_str = "到達可能 [OK]" if ik_targets is not None else "到達不能/めり込み除外 [--]"
            print(f"[{idx:02d}/{len(grid_points):02d}] 目標: r={r*1000:3.0f}mm, θ={theta:+3.0f}°, z={z*1000:2.0f}mm ➔ {status_str}")

            if ik_targets is None:
                continue

            # --- 1. Home ➔ 目標点へのアプローチ補間 ---
            steps = 40
            for step in range(1, steps + 1):
                if not viewer.is_running():
                    break
                while paused and viewer.is_running():
                    time.sleep(0.05)

                ratio = step / steps
                interp_pos = interpolate_positions(home_pos, ik_targets, ratio)
                
                # 関節角の更新
                for i, sid in enumerate(SERVO_IDS):
                    if i < model.nq:
                        data.qpos[i] = raw_to_radian(sid, interp_pos[sid])
                mujoco.mj_forward(model, data)

                # 目標点に「赤い半透明球」を描画
                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.012, 0, 0],  # 直径 24mm の球
                    pos=current_target_xyz,
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.1, 0.1, 0.75]  # 赤色半透明
                )
                viewer.user_scn.ngeom = 1
                viewer.sync()

                time.sleep((0.015 / playback_speed))

            time.sleep(0.2 / playback_speed)  # 目標点での静止確認

            # --- 2. 目標点 ➔ Home への復帰補間 ---
            for step in range(1, steps + 1):
                if not viewer.is_running():
                    break
                while paused and viewer.is_running():
                    time.sleep(0.05)

                ratio = step / steps
                interp_pos = interpolate_positions(ik_targets, home_pos, ratio)
                
                for i, sid in enumerate(SERVO_IDS):
                    if i < model.nq:
                        data.qpos[i] = raw_to_radian(sid, interp_pos[sid])
                mujoco.mj_forward(model, data)

                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.012, 0, 0],
                    pos=current_target_xyz,
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.1, 0.1, 0.75]
                )
                viewer.user_scn.ngeom = 1
                viewer.sync()

                time.sleep((0.012 / playback_speed))

    print("\n✅ 全グリッドの視覚検証が完了しました。")


if __name__ == "__main__":
    main()