"""
==============================================================================
到達可能レンジ スイープ検証テスト (tests/test_ik_range_sweep.py)
==============================================================================
【役割】
  - ID 1 (旋回) を 0°(正面), +60°(最左), -60°(最右) に固定
  - r を 0mm から 10mm 刻みで走査し、到達成否・進入角度・手首位置を判定
  - 成功した点について MuJoCo で 3D アニメーション（目標球 ＆ 動作）を描画

【操作方法】
  - [Z] キー : 角度を【正面 0°】に切り替え
  - [X] キー : 角度を【最左 +60°】に切り替え
  - [C] キー : 角度を【最右 -60°】に切り替え
  - [Space]  : 一時停止 / 再開
  - [1]〜[9]  : 再生速度変更
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
# 🛠️ スイープ・幾何パラメータ
# ==============================================================================
L_GRIPPER = 0.160            # 手首関節から爪先端までの実効長 [m] (16cm)  
DELTA_Z_WRIST_WP = 0.050     # 手首目標に対する手首経由点の上空オフセット [m] (5cm)  
DEFAULT_APPROACH_DEG = 90.0  # デフォルト進入角度 (度)  
MIN_APPROACH_DEG = 30.0      # フォールバック最小許容角度 (度)  

# 走査範囲 [m]
R_START = 0.000
R_END = 1.000
R_STEP = 0.010               # 10mm 刻み
Z_TARGET = 0.015             # 把持高さ 15mm

# 初期旋回角設定 (0.0, 60.0, -60.0)
INITIAL_THETA_DEG = -60.0
# ==============================================================================


def solve_ik_wrist_and_pitch(
    r_wrist: float, 
    theta_deg: float, 
    z_wrist: float, 
    target_pitch_deg: float
) -> dict | None:
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

    # ステップ 1: ID 1〜3 で手首関節を目標座標へ誘導  
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

    # ステップ 2: 90°基準から前方へ外向きに倒す  
    q2 = _IK_DATA.qpos[1]  
    q3 = _IK_DATA.qpos[2]  
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)  
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad  
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
    for pitch_deg in np.arange(init_pitch_deg, MIN_APPROACH_DEG - 1.0, -5.0):  
        pitch_rad = math.radians(pitch_deg)  

        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)  
        z_wrist_target = z_tcp + L_GRIPPER * math.sin(pitch_rad)  

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


def execute_trajectory(viewer, model, data, start_pos, target_pos, tcp_xyz, wrist_xyz, wp_xyz, steps=25, speed=2.0, paused_flag_getter=None):
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
    playback_speed = 2.5
    current_theta = INITIAL_THETA_DEG

    def key_callback(keycode):
        nonlocal paused, playback_speed, current_theta
        if keycode == 32:  # Space
            paused = not paused  
            print("\n[一時停止]" if paused else "\n[再開]")  
        elif keycode in (90, 122):  # Z, z (正面 0°)
            current_theta = 0.0
            print(f"\n🔄 旋回角切替: 【正面 (θ = 0.0°)】")
        elif keycode in (88, 120):  # X, x (最左 +60°)
            current_theta = 60.0
            print(f"\n🔄 旋回角切替: 【最左 (θ = +60.0°)】")
        elif keycode in (67, 99):   # C, c (最右 -60°)
            current_theta = -60.0
            print(f"\n🔄 旋回角切替: 【最右 (θ = -60.0°)】")
        elif 49 <= keycode <= 57:  # 1〜9  
            playback_speed = float(keycode - 48)  
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")  
        elif 321 <= keycode <= 329:  # テンキー  
            playback_speed = float(keycode - 320)  
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")  

    # r を 0mm から 360mm まで 10mm 刻みで生成
    r_sweep_list = [round(r, 3) for r in np.arange(R_START, R_END + 0.001, R_STEP)]
    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}  

    print("=======================================================================")
    print(" 🚀 到達可能レンジ スイープ検証テスト (r: 0mm ➔ 360mm / 10mm刻み)")
    print(f" ⚙️ 現在の旋回角: θ = {current_theta:+.1f}°")
    print(" 💡 [Z]: 正面 (0°) / [X]: 最左 (+60°) / [C]: 最右 (-60°)")
    print(" 🔴 赤: 爪先目標 / 🔵 青: 手首目標 / 🟡 黄: 手首経由点")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:  
        r_idx = 0
        while viewer.is_running():
            r = r_sweep_list[r_idx % len(r_sweep_list)]
            r_idx += 1

            th_rad = math.radians(-current_theta)  
            tcp_xyz = np.array([r * math.sin(th_rad), -r * math.cos(th_rad), Z_TARGET])  

            ik_target, ik_wp, achieved_pitch, wrist_pt, wp_pt = solve_ik_adaptive_approach(  
                r_tcp=r, theta_deg=current_theta, z_tcp=Z_TARGET  
            )

            if ik_target is None:
                print(f"[{r*1000:3.0f}mm | θ={current_theta:+3.0f}°]  ❌ 到達不可 / 限界外")
                time.sleep(0.02)
                continue

            r_w, z_w = wrist_pt  
            r_wp, z_wp = wp_pt  
            wrist_xyz = np.array([r_w * math.sin(th_rad), -r_w * math.cos(th_rad), z_w])  
            wp_xyz = np.array([r_wp * math.sin(th_rad), -r_wp * math.cos(th_rad), z_wp])  

            print(f"[{r*1000:3.0f}mm | θ={current_theta:+3.0f}°]  ✅ 成功 ➔ 進入角: {achieved_pitch:4.1f}° (手首: r={r_w*1000:3.0f}mm, z={z_w*1000:3.0f}mm)")

            # 1. Home ➔ 手首経由点 (🟡)  
            execute_trajectory(viewer, model, data, home_pos, ik_wp, tcp_xyz, wrist_xyz, wp_xyz,  
                               steps=25, speed=playback_speed, paused_flag_getter=lambda: paused)  
            time.sleep(0.05 / playback_speed)

            # 2. 手首経由点 (🟡) ➔ 目標点 (🔵/🔴)  
            execute_trajectory(viewer, model, data, ik_wp, ik_target, tcp_xyz, wrist_xyz, wp_xyz,  
                               steps=18, speed=playback_speed, paused_flag_getter=lambda: paused)  
            time.sleep(0.3 / playback_speed)

            # 3. 目標点 ➔ 手首経由点 (🟡) 退避  
            execute_trajectory(viewer, model, data, ik_target, ik_wp, tcp_xyz, wrist_xyz, wp_xyz,  
                               steps=18, speed=playback_speed, paused_flag_getter=lambda: paused)  

            # 4. 手首経由点 (🟡) ➔ Home  
            execute_trajectory(viewer, model, data, ik_wp, home_pos, tcp_xyz, wrist_xyz, wp_xyz,  
                               steps=25, speed=playback_speed, paused_flag_getter=lambda: paused)  

            time.sleep(0.1 / playback_speed)


if __name__ == "__main__":
    main()