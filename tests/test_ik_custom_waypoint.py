"""
==============================================================================
カスタム Waypoint (経由点) パラメータ検証テスト (tests/test_ik_custom_waypoint.py)
==============================================================================
【役割】
目標点 (r, theta, z) に対する経由点 (r_wp, theta, z_wp) のオフセットを自在に調整し、
アプローチ角度の改善効果を視覚的に検証します。

【表示要素】
  - 🔴 赤い球 : 最終目標点 (Pick / Grasp)
  - 🟡 黄色い球 : 経由点 (Waypoint / Pre-Grasp)

【操作方法】
  - [Space] : 一時停止 / 再開
  - [1]〜[9]: 再生速度倍率変更
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


# ==============================================================================
# 🛠️ Waypoint 調整パラメータ (ここを自由に変更してテスト)
# ==============================================================================
# 目標点に対する経由点 (Waypoint) の相対オフセット
DELTA_R_WP = +0.350   # 半径オフセット [m] (例: +0.060 で目標の 6cm 奥を経由)
DELTA_Z_WP = +0.350   # 高さオフセット [m] (例: +0.050 で目標の 5cm 上空を経由)

# アームの物理限界に基づくクランプ設定 [m]
R_MIN_LIMIT = 0.160
R_MAX_LIMIT = 0.310
Z_MIN_LIMIT = 0.015
Z_MAX_LIMIT = 0.150
# ==============================================================================


def calculate_waypoint(r: float, theta_deg: float, z: float):
    """目標値 (r, theta, z) から安全な経由点 (r_wp, theta, z_wp) を計算"""
    r_wp = max(R_MIN_LIMIT, min(R_MAX_LIMIT, r + DELTA_R_WP))
    z_wp = max(Z_MIN_LIMIT, min(Z_MAX_LIMIT, z + DELTA_Z_WP))
    return r_wp, theta_deg, z_wp


def interpolate_positions(start_pos: dict, end_pos: dict, ratio: float) -> dict:
    """コサイン S 字補間"""
    s_ratio = (1.0 - math.cos(ratio * math.pi)) / 2.0
    current = {}
    for sid in SERVO_IDS:
        s_val = start_pos.get(sid, JOINT_CONFIG[sid]["init"])
        e_val = end_pos.get(sid, JOINT_CONFIG[sid]["init"])
        current[sid] = int(s_val + s_ratio * (e_val - s_val))
    return current


def execute_trajectory(viewer, model, data, start_pos, target_pos, target_xyz, wp_xyz, steps=30, speed=2.0, paused_flag_getter=None):
    """補間移動および目標球・経由点球の描画"""
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

        # ターゲット球 (赤) と 経由点球 (黄) を描画
        viewer.user_scn.ngeom = 0
        
        # 1. 最終目標点: 赤球
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=target_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.1, 0.1, 0.75]
        )
        # 2. 経由点: 黄色球
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.009, 0, 0],
            pos=wp_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.9, 0.1, 0.65]
        )
        viewer.user_scn.ngeom = 2
        viewer.sync()

        time.sleep(0.015 / speed)


def main():
    xml_path = os.path.join(current_dir, "assets", "so100_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    paused = False
    playback_speed = 2.0

    def key_callback(keycode):
        nonlocal paused, playback_speed
        if keycode == 32:  # Space
            paused = not paused
            print("\n[一時停止]" if paused else "\n[再開]")
        elif 49 <= keycode <= 57:  # 1〜9
            playback_speed = float(keycode - 48)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")
        elif 321 <= keycode <= 329:  # テンキー
            playback_speed = float(keycode - 320)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")

    # 検証用代表点 (近傍・中間・遠方 × 左・正面・右)
    r_list = [0.19, 0.24, 0.28]
    #theta_list = [-40.0, 0.0, 40.0]
    theta_list = [0.0, 0.0, 0.0]
    z_target = 0.015

    test_points = [(r, th, z_target) for r in r_list for th in theta_list]
    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("=======================================================================")
    print(" 🚀 カスタム Waypoint アプローチ検証")
    print(f" ⚙️ 設定オフセット: Δr = {DELTA_R_WP*1000:+3.0f}mm, Δz = {DELTA_Z_WP*1000:+3.0f}mm")
    print(" 🔴 赤球: 最終目標点 / 🟡 黄球: 経由点")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        idx = 0
        while viewer.is_running():
            r, theta, z = test_points[idx % len(test_points)]
            idx += 1

            # 経由点座標を算出
            r_wp, theta_wp, z_wp = calculate_waypoint(r, theta, z)

            # 直交座標変換
            th_rad = math.radians(-theta)
            target_xyz = np.array([r * math.sin(th_rad), -r * math.cos(th_rad), z])
            wp_xyz = np.array([r_wp * math.sin(th_rad), -r_wp * math.cos(th_rad), z_wp])

            # IK 計算
            ik_target = solve_ik_polar(r=r, theta_deg=theta, z=z, prevent_penetration=True)
            ik_wp = solve_ik_polar(r=r_wp, theta_deg=theta_wp, z=z_wp, prevent_penetration=True)

            if ik_target is None:
                print(f"⚠️ 目標点到達不能: r={r*1000:.0f}mm, θ={theta:+3.0f}°")
                continue
            if ik_wp is None:
                ik_wp = ik_target  # 経由点が範囲外なら直接アプローチ

            print(f"[{idx:02d}] Target(🔴): r={r*1000:3.0f}mm ➔ Waypoint(🟡): r={r_wp*1000:3.0f}mm, z={z_wp*1000:2.0f}mm")

            # 1. Home ➔ 経由点 (🟡)
            execute_trajectory(viewer, model, data, home_pos, ik_wp, target_xyz, wp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.1 / playback_speed)

            # 2. 経由点 (🟡) ➔ 最終目標 (🔴)
            execute_trajectory(viewer, model, data, ik_wp, ik_target, target_xyz, wp_xyz,
                               steps=25, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.4 / playback_speed)

            # 3. 最終目標 (🔴) ➔ 経由点 (🟡) へ退避
            execute_trajectory(viewer, model, data, ik_target, ik_wp, target_xyz, wp_xyz,
                               steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)

            # 4. 経由点 (🟡) ➔ Home 復帰
            execute_trajectory(viewer, model, data, ik_wp, home_pos, target_xyz, wp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)

            time.sleep(0.2 / playback_speed)


if __name__ == "__main__":
    main()