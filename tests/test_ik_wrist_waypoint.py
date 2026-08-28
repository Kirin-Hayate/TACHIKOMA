"""
==============================================================================
手首角度拘束付き Waypoint アプローチ検証 (tests/test_ik_wrist_waypoint.py)
==============================================================================
【役割】
経由点において、空間座標 (r_wp, theta, z_wp) だけでなく
「ID 4 (手首) の目標角度（曲げ角）」を直接拘束した経由姿勢を生成し、
アプローチ角度の改善と安定性を検証します。

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
from core.kinematics import solve_ik_polar, raw_to_radian, radian_to_raw, _IK_MODEL, _IK_DATA, _JAW_BODY_ID, JAW_LOCAL_TCP_OFFSET, check_ground_penetration 


# ==============================================================================
# 🛠️ 経由パラメータ設定 (ここを調整して挙動をテスト)
# ==============================================================================
DELTA_R_WP = +0.040        # 経由点の半径オフセット [m] (+0.040 = 4cm 奥)
DELTA_Z_WP = +0.060        # 経由点の高さオフセット [m] (+0.060 = 6cm 上空)

# 経由点における ID 4 (手首) の強制目標角度 (ラジアン)
# 0.0: まっすぐ, 負の値: 下向きに深く曲げる (例: -0.9 〜 -1.2 rad で手首を真下に立てる)
WRIST_WP_TARGET_RAD = -1.55  # 約 -60度 手首を折り曲げる

# ワークスペース限界 [m]
R_MIN_LIMIT = 0.160
R_MAX_LIMIT = 0.310
Z_MIN_LIMIT = 0.015
Z_MAX_LIMIT = 0.150
# ==============================================================================


def solve_ik_with_fixed_wrist(r: float, theta_deg: float, z: float, wrist_rad: float) -> dict | None:
    """
    ID 4 (手首ピッチ) を特定角度に固定した状態で、ID 1〜3 (旋回・肩・肘) を解く
    """
    if _IK_MODEL is None or _IK_DATA is None or _JAW_BODY_ID == -1:
        return None

    theta_rad = math.radians(-theta_deg)
    target_pos = np.array([r * math.sin(theta_rad), -r * math.cos(theta_rad), z])

    # 初期姿勢: 手首を指定角度に固定
    init_qpos = np.array([theta_rad, 1.2, -1.8, wrist_rad, 0.0, 0.0])
    _IK_DATA.qpos[:6] = init_qpos
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    jacp = np.zeros((3, _IK_MODEL.nv))

    for _ in range(40):
        rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
        current_tip_pos = _IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET
        error = target_pos - current_tip_pos

        if np.linalg.norm(error) < 2e-3:
            break

        # ID 1〜3 (旋回・肩・肘) の3軸のみ動かすヤコビアン 
        mujoco.mj_jac(_IK_MODEL, _IK_DATA, jacp, None, current_tip_pos, _JAW_BODY_ID)
        J = jacp[:, :3]  # 3x3 行列

        # DLS 逆行列
        J_inv = J.T @ np.linalg.inv(J @ J.T + (0.015**2) * np.eye(3))
        delta_q = np.clip(J_inv @ error, -0.25, 0.25)

        _IK_DATA.qpos[:3] += delta_q
        _IK_DATA.qpos[3] = wrist_rad  # 手首角度を固定維持
        mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
    final_tip_pos = _IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET
    if np.linalg.norm(target_pos - final_tip_pos) > 0.025:
        return None

    if check_ground_penetration(min_z_threshold=0.002):
        return None

    raw_targets = {
        1: radian_to_raw(1, _IK_DATA.qpos[0]),
        2: radian_to_raw(2, _IK_DATA.qpos[1]),
        3: radian_to_raw(3, _IK_DATA.qpos[2]),
        4: radian_to_raw(4, _IK_DATA.qpos[3]),
        5: JOINT_CONFIG[5]["init"],
        6: JOINT_CONFIG[6]["init"],
    }

    for sid in [1, 2, 3, 4]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            if not (cfg["f_min"] <= raw_targets[sid] <= cfg["f_max"]):
                return None

    return raw_targets


def interpolate_positions(start_pos: dict, end_pos: dict, ratio: float) -> dict:
    s_ratio = (1.0 - math.cos(ratio * math.pi)) / 2.0
    return {
        sid: int(start_pos[sid] + s_ratio * (end_pos[sid] - start_pos[sid]))
        for sid in SERVO_IDS
    }


def execute_trajectory(viewer, model, data, start_pos, target_pos, target_xyz, wp_xyz, steps=30, speed=2.0, paused_flag_getter=None):
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

        # 赤球 (目標) & 黄球 (経由点) 描画
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=target_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.1, 0.1, 0.75]
        )
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
        if keycode == 32:
            paused = not paused
            print("\n[一時停止]" if paused else "\n[再開]")
        elif 49 <= keycode <= 57:
            playback_speed = float(keycode - 48)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")
        elif 321 <= keycode <= 329:
            playback_speed = float(keycode - 320)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")

    r_list = [0.19, 0.24, 0.28]
    theta_list = [-40.0, 0.0, 40.0]
    z_target = 0.015

    test_points = [(r, th, z_target) for r in r_list for th in theta_list]
    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("=======================================================================")
    print(" 🚀 手首角度拘束 Waypoint アプローチ検証")
    print(f" ⚙️ 経由設定: Δr={DELTA_R_WP*1000:+3.0f}mm, Δz={DELTA_Z_WP*1000:+3.0f}mm, Wrist={math.degrees(WRIST_WP_TARGET_RAD):.1f}°")
    print(" 🔴 赤球: 目標点 / 🟡 黄球: 経由点")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        idx = 0
        while viewer.is_running():
            r, theta, z = test_points[idx % len(test_points)]
            idx += 1

            r_wp = max(R_MIN_LIMIT, min(R_MAX_LIMIT, r + DELTA_R_WP))
            z_wp = max(Z_MIN_LIMIT, min(Z_MAX_LIMIT, z + DELTA_Z_WP))

            th_rad = math.radians(-theta)
            target_xyz = np.array([r * math.sin(th_rad), -r * math.cos(th_rad), z])
            wp_xyz = np.array([r_wp * math.sin(th_rad), -r_wp * math.cos(th_rad), z_wp])

            # 最終目標: 通常IK
            ik_target = solve_ik_polar(r=r, theta_deg=theta, z=z, prevent_penetration=True)
            # 経由点: 手首角度を強制指定したIK
            ik_wp = solve_ik_with_fixed_wrist(r=r_wp, theta_deg=theta, z=z_wp, wrist_rad=WRIST_WP_TARGET_RAD)
            if ik_wp is None:
                ik_wp = solve_ik_polar(r=r_wp, theta_deg=theta, z=z_wp, prevent_penetration=True)

            if ik_target is None or ik_wp is None:
                print(f"⚠️ 到達不能: r={r*1000:.0f}mm, θ={theta:+3.0f}°")
                continue

            print(f"[{idx:02d}] Target(🔴): r={r*1000:3.0f}mm ➔ Waypoint(🟡): 手首固定で進入")

            # 1. Home ➔ 経由点 (手首曲げ固定)
            execute_trajectory(viewer, model, data, home_pos, ik_wp, target_xyz, wp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.1 / playback_speed)

            # 2. 経由点 ➔ 最終目標 (垂直アプローチ)
            execute_trajectory(viewer, model, data, ik_wp, ik_target, target_xyz, wp_xyz,
                               steps=25, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.4 / playback_speed)

            # 3. 最終目標 ➔ 経由点
            execute_trajectory(viewer, model, data, ik_target, ik_wp, target_xyz, wp_xyz,
                               steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)

            # 4. 経由点 ➔ Home
            execute_trajectory(viewer, model, data, ik_wp, home_pos, target_xyz, wp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)

            time.sleep(0.2 / playback_speed)


if __name__ == "__main__":
    main()