"""
==============================================================================
手首角度ダイレクト指定・爪先追従テスト (tests/test_ik_wrist_parametric.py)
==============================================================================
【役割】
手首 (ID 4) の曲げ角を「Raw値 / 角度」で直接固定した状態で、
爪先 (TCP) が赤いボールにピタリと重なるように肩・肘 (ID 1〜3) を数値IKで誘導します。
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
    _JAW_BODY_ID,
    JAW_LOCAL_TCP_OFFSET,
    check_ground_penetration
) 


# ==============================================================================
# 🛠️ 手首 (ID 4) の曲げ角度設定
# ==============================================================================
# WRIST_ANGLE_DEG: 
#   0.0  : まっすぐ水平
#  -45.0 : 斜め下 45度
#  -70.0 : 深く折り曲げて真下に近い角度
WRIST_ANGLE_DEG = -60.0
# ==============================================================================


def solve_ik_with_wrist_angle(r: float, theta_deg: float, z: float, wrist_deg: float) -> dict | None:
    """
    ID 4 (手首) を指定角度に固定した状態で、爪先 (TCP) を (r, theta, z) に誘導する
    """
    if _IK_MODEL is None or _IK_DATA is None or _JAW_BODY_ID == -1:
        return None

    theta_rad = math.radians(-theta_deg)
    target_pos = np.array([r * math.sin(theta_rad), -r * math.cos(theta_rad), z])
    wrist_rad = math.radians(wrist_deg)

    # 初期姿勢: 手首を指定角度に固定した状態でスタート
    init_qpos = np.array([theta_rad, 1.2, -1.8, wrist_rad, 0.0, 0.0])
    _IK_DATA.qpos[:6] = init_qpos
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    jacp = np.zeros((3, _IK_MODEL.nv))

    # ID 1〜3 (旋回・肩・肘) の3軸のみを更新して爪先を合わせる
    for _ in range(40):
        rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
        current_tip_pos = _IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET
        error = target_pos - current_tip_pos

        if np.linalg.norm(error) < 2e-3:
            break

        mujoco.mj_jac(_IK_MODEL, _IK_DATA, jacp, None, current_tip_pos, _JAW_BODY_ID)
        J = jacp[:, :3]  # 3x3 位置ヤコビアン

        J_inv = J.T @ np.linalg.inv(J @ J.T + (0.015**2) * np.eye(3))
        delta_q = np.clip(J_inv @ error, -0.25, 0.25)

        _IK_DATA.qpos[:3] += delta_q
        _IK_DATA.qpos[3] = wrist_rad  # 手首角度を維持
        mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    # 収束判定 (爪先が 20mm 以内に到達しているか)
    rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
    final_tip_pos = _IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET
    if np.linalg.norm(target_pos - final_tip_pos) > 0.020:
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

    # サーボ物理可動限界チェック
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


def execute_trajectory(viewer, model, data, start_pos, target_pos, target_tcp, steps=30, speed=2.0, paused_flag_getter=None):
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
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=target_tcp,
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
    r_list = [0.18, 0.22, 0.26]
    theta_list = [-40.0, 0.0, 40.0]
    z_target = 0.015

    test_points = [(r, th, z_target) for r in r_list for th in theta_list]
    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("=======================================================================")
    print(" 🚀 手首角度ダイレクト指定・爪先追従テスト")
    print(f" ⚙️ 手首固定角度 (ID 4): {WRIST_ANGLE_DEG:.1f}°")
    print(" 🔴 赤球: 爪先目標位置 (z=15mm)")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        idx = 0
        while viewer.is_running():
            r, theta, z = test_points[idx % len(test_points)]
            idx += 1

            th_rad = math.radians(-theta)
            target_tcp_xyz = np.array([r * math.sin(th_rad), -r * math.cos(th_rad), z])

            # 目標点と上空退避点のIK
            ik_target = solve_ik_with_wrist_angle(r=r, theta_deg=theta, z=z, wrist_deg=WRIST_ANGLE_DEG)
            ik_pre = solve_ik_with_wrist_angle(r=r, theta_deg=theta, z=z + 0.050, wrist_deg=WRIST_ANGLE_DEG)

            if ik_target is None or ik_pre is None:
                print(f"⚠️ 到達限界外: r={r*1000:.0f}mm, θ={theta:+3.0f}°")
                continue

            print(f"[{idx:02d}] 進入テスト: r={r*1000:3.0f}mm, θ={theta:+3.0f}° [OK]")

            # 1. Home ➔ 上空退避点
            execute_trajectory(viewer, model, data, home_pos, ik_pre, target_tcp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.1 / playback_speed)

            # 2. 上空退避点 ➔ 目標点降下 (爪先が赤い球に届く)
            execute_trajectory(viewer, model, data, ik_pre, ik_target, target_tcp_xyz,
                               steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)
            time.sleep(0.4 / playback_speed)

            # 3. 目標点 ➔ 上空退避点
            execute_trajectory(viewer, model, data, ik_target, ik_pre, target_tcp_xyz,
                               steps=20, speed=playback_speed, paused_flag_getter=lambda: paused)

            # 4. 上空退避点 ➔ Home
            execute_trajectory(viewer, model, data, ik_pre, home_pos, target_tcp_xyz,
                               steps=30, speed=playback_speed, paused_flag_getter=lambda: paused)

            time.sleep(0.2 / playback_speed)


if __name__ == "__main__":
    main()