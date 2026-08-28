"""
==============================================================================
手首位置指定 ＆ 進入角適応フォールバック検証 (tests/test_ik_wrist_adaptive.py)
==============================================================================
"""

import sys
import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and os.path.basename(current_dir) != "tachikoma":
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config.joint_config import SERVO_IDS, JOINT_CONFIG
from core.kinematics import (
    raw_to_radian, 
    radian_to_raw, 
    _IK_MODEL, 
    _IK_DATA, 
    check_ground_penetration
)

# 手首関節ボディ ID の検出
_WRIST_BODY_ID = -1
if _IK_MODEL is not None:
    for name in ["wrist_pitch_link", "wrist_pitch", "wrist", "gripper"]:
        bid = mujoco.mj_name2id(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            _WRIST_BODY_ID = bid
            break

# ==============================================================================
# 🛠️ 幾何・経由パラメータ
# ==============================================================================
L_GRIPPER = 0.105         # 手首関節から爪先端までの実効長 [m] (10.5cm)
DELTA_Z_WRIST_WP = 0.150  # 手首目標に対する手首経由点の上空オフセット [m] (5cm)
DEFAULT_APPROACH_DEG = 90.0  # デフォルト進入角度 (度)
MIN_APPROACH_DEG = 30.0      # フォールバック最小許容角度 (度)
# ==============================================================================


def solve_ik_wrist_and_pitch(
    r_wrist: float, 
    theta_deg: float, 
    z_wrist: float, 
    target_pitch_deg: float
) -> dict | None:
    """
    手首関節(ID 4)を (r_wrist, theta, z_wrist) に配置し、
    手先が地面に対して下向きに折れ曲がるよう全4軸を確定する
    """
    if _IK_MODEL is None or _IK_DATA is None or _WRIST_BODY_ID == -1:
        return None

    theta_rad = math.radians(-theta_deg)
    target_wrist_pos = np.array([
        r_wrist * math.sin(theta_rad),
        -r_wrist * math.cos(theta_rad),
        z_wrist
    ])

    init_qpos = np.array([theta_rad, 1.2, -1.8, -1.5, 0.0, 0.0])
    _IK_DATA.qpos[:6] = init_qpos
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    jacp = np.zeros((3, _IK_MODEL.nv))

    # ステップ 1: ID 1〜3 (旋回・肩・肘) で手首関節を目標座標へ誘導
    for _ in range(40):
        current_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
        error = target_wrist_pos - current_wrist_pos

        if np.linalg.norm(error) < 1.5e-3:
            break

        mujoco.mj_jacBody(_IK_MODEL, _IK_DATA, jacp, None, _WRIST_BODY_ID)
        J = jacp[:, :3]

        J_inv = J.T @ np.linalg.inv(J @ J.T + (0.015**2) * np.eye(3))
        delta_q = np.clip(J_inv @ error, -0.25, 0.25)

        _IK_DATA.qpos[:3] += delta_q
        mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    final_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
    if np.linalg.norm(target_wrist_pos - final_wrist_pos) > 0.020:
        return None

    # ステップ 2: 逆方向へシフトして真下に折り曲げる
    q2 = _IK_DATA.qpos[1]
    q3 = _IK_DATA.qpos[2]
    pitch_rad = math.radians(target_pitch_deg)
    q4 = -(pitch_rad + q2 + q3 - (math.pi)*0.9)
    _IK_DATA.qpos[3] = q4
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

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

    # サーボ可動限界チェック
    for sid in [1, 2, 3, 4]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            if not (cfg["f_min"] <= raw_targets[sid] <= cfg["f_max"]):
                return None

    return raw_targets


def solve_ik_adaptive_approach(
    r_tcp: float, 
    theta_deg: float, 
    z_tcp: float, 
    init_pitch_deg: float = DEFAULT_APPROACH_DEG
):
    """
    進入角度 x を 90° から段階的に緩和しながら手首目標と手首経由点を算出する
    """
    for pitch_deg in np.arange(init_pitch_deg, MIN_APPROACH_DEG - 1.0, -5.0):
        pitch_rad = math.radians(pitch_deg)

        # 1. 爪先と進入角から手首目標位置を幾何逆算
        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = z_tcp + L_GRIPPER * math.sin(pitch_rad)

        # 2. 手首経由点 (手首目標の指定cm真上)
        r_wrist_wp = r_wrist_target
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP

        ik_target = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target, theta_deg=theta_deg, z_wrist=z_wrist_target, target_pitch_deg=pitch_deg
        )
        ik_wp = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp, theta_deg=theta_deg, z_wrist=z_wrist_wp, target_pitch_deg=pitch_deg
        )

        if ik_target is not None and ik_wp is not None:
            return ik_target, ik_wp, pitch_deg, (r_wrist_target, z_wrist_target), (r_wrist_wp, z_wrist_wp)

    return None, None, None, None, None


def interpolate_positions(start_pos: dict, end_pos: dict, ratio: float) -> dict:
    s_ratio = (1.0 - math.cos(ratio * math.pi)) / 2.0
    return {
        sid: int(start_pos[sid] + s_ratio * (end_pos[sid] - start_pos[sid]))
        for sid in SERVO_IDS
    }


def execute_trajectory(viewer, model, data, start_pos, target_pos, tcp_xyz, wrist_xyz, wp_xyz, steps=30, speed=2.0, paused_flag_getter=None):
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

        viewer.user_scn.ngeom = 0
        # 1. 爪先目標 (赤球)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=tcp_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.1, 0.1, 0.75]
        )
        # 2. 手首目標位置 (青球)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.009, 0, 0],
            pos=wrist_xyz,
            mat=np.eye(3).flatten(),
            rgba=[0.1, 0.4, 1.0, 0.75]
        )
        # 3. 手首経由点 (黄球)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.008, 0, 0],
            pos=wp_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.9, 0.1, 0.65]
        )
        viewer.user_scn.ngeom = 3
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

    # テスト対象グリッド
    r_list = [0.18, 0.22, 0.26, 0.29]
    theta_list = [-40.0, 0.0, 40.0]
    z_target = 0.015

    test_points = [(r, th, z_target) for r in r_list for th in theta_list]
    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("=======================================================================")
    print(" 🚀 手首位置指定 ＆ 適応型アプローチ検証")
    print(f" ⚙️ 手首経由点オフセット: Δz_wp = +{DELTA_Z_WRIST_WP*1000:.0f}mm")
    print(" 🔴 赤: 爪先目標 / 🔵 青: 手首目標 / 🟡 黄: 手首経由点")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        idx = 0
        while viewer.is_running():
            r, theta, z = test_points[idx % len(test_points)]
            idx += 1

            th_rad = math.radians(-theta)
            tcp_xyz = np.array([r * math.sin(th_rad), -r * math.cos(th_rad), z])

            # 適応型進入角 IK
            ik_target, ik_wp, achieved_pitch, wrist_pt, wp_pt = solve_ik_adaptive_approach(
                r_tcp=r, theta_deg=theta, z_tcp=z
            )

            if ik_target is None:
                print(f"⚠️ 到達不能 (30°まで緩和しても不可): r={r*1000:.0f}mm, θ={theta:+3.0f}°")
                continue

            r_w, z_w = wrist_pt
            r_wp, z_wp = wp_pt
            wrist_xyz = np.array([r_w * math.sin(th_rad), -r_w * math.cos(th_rad), z_w])
            wp_xyz = np.array([r_wp * math.sin(th_rad), -r_wp * math.cos(th_rad), z_wp])

            print(f"[{idx:02d}] r={r*1000:3.0f}mm, θ={theta:+3.0f}° ➔ 進入角: {achieved_pitch:4.1f}° (手首: r={r_w*1000:3.0f}mm, z={z_w*1000:3.0f}mm)")

            # 1. Home ➔ 手首経由点 (🟡)
            execute_trajectory(viewer, model, data, home_pos, ik_wp, tcp_xyz, wrist_xyz, wp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.1 / playback_speed)

            # 2. 手首経由点 (🟡) ➔ 手首目標点 (🔵) / 爪先 (🔴) 降下
            execute_trajectory(viewer, model, data, ik_wp, ik_target, tcp_xyz, wrist_xyz, wp_xyz,
                               steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.4 / playback_speed)

            # 3. 把持点 ➔ 手首経由点 (🟡) 退避
            execute_trajectory(viewer, model, data, ik_target, ik_wp, tcp_xyz, wrist_xyz, wp_xyz,
                               steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)

            # 4. 手首経由点 (🟡) ➔ Home
            execute_trajectory(viewer, model, data, ik_wp, home_pos, tcp_xyz, wrist_xyz, wp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)

            time.sleep(0.2 / playback_speed)


if __name__ == "__main__":
    main()