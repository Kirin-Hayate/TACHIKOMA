"""
==============================================================================
作業領域グリッドスイープ (255点 ➔ 正面30cm) Pick & Place 検証テスト
(tests/test_grid_sweep_to_center.py)
==============================================================================
【動作仕様】
  1. 角度: -70° 〜 +70° (10°刻み, 15分割)
     距離: 0cm 〜 65cm (5cm刻み, 14分割)
     の計 210通り (※ 0cm〜65cm の 5cm 刻みは 14 行) について、
     Place 地点「正面 (0°), 30cm, z=15mm」への Pick & Place 軌道成立を判定。
  2. ターミナルへ判定結果マトリクス (◯ / ✕) を表形式で出力。
  3. 成功した経路のみを抽出し、MuJoCo ビューアー上で順次連続再生。

【操作方法】
  - [Space] : 一時停止 / 再開
  - [1]〜[9]: 再生速度変更
  - [Esc]   : 終了
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
# 🛠️ 幾何・把持パラメータ 
# ==============================================================================
L_GRIPPER = 0.160            # 手首関節から爪先端までの実効長 [m] 
DELTA_Z_WRIST_WP = 0.050     # 手首目標に対する上空待機オフセット [m] 
DEFAULT_APPROACH_DEG = 90.0  # 進入角度 
MIN_APPROACH_DEG = 30.0 

# ID 5 (手首ロール): 水平横挟み姿勢 (+90° オフセット) 
WRIST_ROLL_HORIZONTAL_RAW = JOINT_CONFIG[5]["init"] + int(1024) 

# ID 6 (グリッパー): 開閉設定 
GRIPPER_OPEN_RAW = JOINT_CONFIG[6].get("f_max", 2600)    # 全開 
GRIPPER_CLOSE_RAW = JOINT_CONFIG[6].get("f_min", 1400)   # 全閉 
# ==============================================================================


def solve_ik_wrist_and_pitch(
    r_wrist: float, 
    theta_deg: float, 
    z_wrist: float, 
    target_pitch_deg: float,
    gripper_raw: int
) -> dict | None:
    """手首位置・進入角・ID 5(横挟み90°)・ID 6(指定開閉状態) を統合したIK"""
    if _IK_MODEL is None or _IK_DATA is None or _WRIST_BODY_ID == -1: 
        return None 

    theta_rad = math.radians(-theta_deg) 
    target_wrist_pos = np.array([
        r_wrist * math.sin(theta_rad),
        -r_wrist * math.cos(theta_rad),
        z_wrist
    ]) 

    init_qpos = np.array([theta_rad, 1.2, -1.8, -1.5, 1.5708, 0.0]) 
    _IK_DATA.qpos[:6] = init_qpos 
    mujoco.mj_forward(_IK_MODEL, _IK_DATA) 

    jacp = np.zeros((3, _IK_MODEL.nv)) 

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

    q2 = _IK_DATA.qpos[1] 
    q3 = _IK_DATA.qpos[2] 
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg) 
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad 
    _IK_DATA.qpos[3] = q4 

    _IK_DATA.qpos[4] = raw_to_radian(5, WRIST_ROLL_HORIZONTAL_RAW) 
    _IK_DATA.qpos[5] = raw_to_radian(6, gripper_raw) 
    mujoco.mj_forward(_IK_MODEL, _IK_DATA) 

    if check_ground_penetration(min_z_threshold=0.002): 
        return None 

    raw_targets = {
        1: radian_to_raw(1, _IK_DATA.qpos[0]), 
        2: radian_to_raw(2, _IK_DATA.qpos[1]), 
        3: radian_to_raw(3, _IK_DATA.qpos[2]), 
        4: radian_to_raw(4, _IK_DATA.qpos[3]), 
        5: WRIST_ROLL_HORIZONTAL_RAW, 
        6: gripper_raw, 
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
    gripper_raw: int,
    init_pitch_deg: float = DEFAULT_APPROACH_DEG 
):
    """適応型進入角度で目標姿勢と上空経由姿勢を計算"""
    for pitch_deg in np.arange(init_pitch_deg, MIN_APPROACH_DEG - 1.0, -5.0): 
        pitch_rad = math.radians(pitch_deg) 

        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad) 
        z_wrist_target = z_tcp + L_GRIPPER * math.sin(pitch_rad) 

        r_wrist_wp = r_wrist_target 
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP 

        ik_target = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target, theta_deg=theta_deg, z_wrist=z_wrist_target, 
            target_pitch_deg=pitch_deg, gripper_raw=gripper_raw
        ) 
        ik_wp = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp, theta_deg=theta_deg, z_wrist=z_wrist_wp, 
            target_pitch_deg=pitch_deg, gripper_raw=gripper_raw
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


def execute_trajectory(viewer, model, data, start_pos, end_pos, pick_xyz, place_xyz, steps=15, speed=3.0, paused_flag_getter=None):
    """滑らかな姿勢遷移と Pick/Place 両マーカー球の描画"""
    for step in range(1, steps + 1): 
        if not viewer.is_running(): 
            break
        while paused_flag_getter and paused_flag_getter() and viewer.is_running(): 
            time.sleep(0.05) 

        ratio = step / steps 
        interp_pos = interpolate_positions(start_pos, end_pos, ratio) 
        
        for i, sid in enumerate(SERVO_IDS): 
            if i < model.nq: 
                data.qpos[i] = raw_to_radian(sid, interp_pos[sid]) 
        mujoco.mj_forward(model, data) 

        viewer.user_scn.ngeom = 0 
        # Pick 球 (赤) 
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0], 
            type=mujoco.mjtGeom.mjGEOM_SPHERE, 
            size=[0.012, 0, 0], 
            pos=pick_xyz, 
            mat=np.eye(3).flatten(), 
            rgba=[1.0, 0.15, 0.15, 0.85]
        )
        # Place 球 (緑) 
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1], 
            type=mujoco.mjtGeom.mjGEOM_SPHERE, 
            size=[0.012, 0, 0], 
            pos=place_xyz, 
            mat=np.eye(3).flatten(), 
            rgba=[0.15, 1.0, 0.25, 0.85]
        )
        viewer.user_scn.ngeom = 2 
        viewer.sync() 

        time.sleep(0.012 / speed)


def run_pick_and_place(viewer, model, data, pick_ik, place_ik, pick_xyz, place_xyz, home_pos, speed_getter, paused_getter):
    """計算済み IK を用いて Pick & Place を再生"""
    ik_pk_target_open, ik_pk_wp_open = pick_ik
    ik_pl_target_open, ik_pl_wp_open = place_ik

    ik_pk_target_closed = ik_pk_target_open.copy() 
    ik_pk_target_closed[6] = GRIPPER_CLOSE_RAW 
    ik_pk_wp_closed = ik_pk_wp_open.copy() 
    ik_pk_wp_closed[6] = GRIPPER_CLOSE_RAW 

    ik_pl_target_closed = ik_pl_target_open.copy() 
    ik_pl_target_closed[6] = GRIPPER_CLOSE_RAW 
    ik_pl_wp_closed = ik_pl_wp_open.copy() 
    ik_pl_wp_closed[6] = GRIPPER_CLOSE_RAW 

    # シーケンス再生 (ステップ数を軽量化)
    execute_trajectory(viewer, model, data, home_pos, ik_pk_wp_open, pick_xyz, place_xyz, steps=18, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pk_wp_open, ik_pk_target_open, pick_xyz, place_xyz, steps=12, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pk_target_open, ik_pk_target_closed, pick_xyz, place_xyz, steps=8, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pk_target_closed, ik_pk_wp_closed, pick_xyz, place_xyz, steps=12, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pk_wp_closed, ik_pl_wp_closed, pick_xyz, place_xyz, steps=24, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pl_wp_closed, ik_pl_target_closed, pick_xyz, place_xyz, steps=12, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pl_target_closed, ik_pl_target_open, pick_xyz, place_xyz, steps=8, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pl_target_open, ik_pl_wp_open, pick_xyz, place_xyz, steps=12, speed=speed_getter(), paused_flag_getter=paused_getter) 
    execute_trajectory(viewer, model, data, ik_pl_wp_open, home_pos, pick_xyz, place_xyz, steps=18, speed=speed_getter(), paused_flag_getter=paused_getter) 


def main():
    angles = list(range(-70, 71, 10))       # -70° 〜 +70° (15 点)
    distances_cm = list(range(0, 66, 5))    # 0cm 〜 65cm (14 点)
    
    # Place 地点: 0° (正面), 30cm, z=15mm
    place_r = 0.30
    place_theta = 0.0
    place_z = 0.015

    print("=======================================================================")
    print(" 🔍 作業領域グリッドスイープ (Pick ➔ 正面30cm Place) を事前計算中...")
    print(f" 📐 角度: {angles[0]}° 〜 {angles[-1]}° (10°刻み, {len(angles)}列)")
    print(f" 📏 距離: {distances_cm[0]}cm 〜 {distances_cm[-1]}cm (5cm刻み, {len(distances_cm)}行)")
    print(f" 🎯 Place 目標地点: r={place_r*100:.0f}cm, θ={place_theta:.0f}°, z={place_z*1000:.0f}mm")
    print("=======================================================================\n")

    # 1. Place 地点の IK 計算
    ik_pl_tgt, ik_pl_wp, pitch_pl, _, _ = solve_ik_adaptive_approach(
        r_tcp=place_r, theta_deg=place_theta, z_tcp=place_z, gripper_raw=GRIPPER_OPEN_RAW
    )
    if ik_pl_tgt is None:
        print("❌ エラー: 固定 Place 地点 (正面 30cm) の IK が解けませんでした。")
        return

    # 2. 全 210 通りのグリッド事前計算
    results_grid = {}          # (d_cm, a_deg) -> True / False
    success_tasks = []         # 成功したタスク情報

    for d_cm in distances_cm:
        r_m = d_cm / 100.0
        for a_deg in angles:
            ik_pk_tgt, ik_pk_wp, pitch_pk, _, _ = solve_ik_adaptive_approach(
                r_tcp=r_m, theta_deg=float(a_deg), z_tcp=place_z, gripper_raw=GRIPPER_OPEN_RAW
            )
            is_ok = (ik_pk_tgt is not None)
            results_grid[(d_cm, a_deg)] = is_ok

            if is_ok:
                th_pk_rad = math.radians(-a_deg)
                pick_xyz = np.array([r_m * math.sin(th_pk_rad), -r_m * math.cos(th_pk_rad), place_z])
                th_pl_rad = math.radians(-place_theta)
                place_xyz = np.array([place_r * math.sin(th_pl_rad), -place_r * math.cos(th_pl_rad), place_z])
                
                success_tasks.append({
                    "dist_cm": d_cm,
                    "angle_deg": a_deg,
                    "pick_ik": (ik_pk_tgt, ik_pk_wp),
                    "place_ik": (ik_pl_tgt, ik_pl_wp),
                    "pick_xyz": pick_xyz,
                    "place_xyz": place_xyz,
                    "pick_pitch": pitch_pk
                })

    # 3. ターミナル一覧マトリクスの整形表示
    header = "角度 [°]|" + "".join([f"{a:>5}" for a in angles])
    separator = "--------+" + "-----" * len(angles)
    
    print(header)
    print(separator)
    for d_cm in distances_cm:
        row_str = f"{d_cm:>3}cm   |"
        for a_deg in angles:
            mark = "  ◯  " if results_grid[(d_cm, a_deg)] else "  ✕  "
            row_str += mark
        print(row_str)

    total_pts = len(distances_cm) * len(angles)
    success_pts = len(success_tasks)
    print(separator)
    print(f"📊 判定結果: {success_pts} / {total_pts} 点が到達可能 (成功率: {success_pts / total_pts * 100:.1f}%)\n")

    # 4. 成功した経路を MuJoCo 上で順次再生
    xml_path = os.path.join(current_dir, "assets", "so100_scene.xml") 
    model = mujoco.MjModel.from_xml_path(xml_path) 
    data = mujoco.MjData(model) 

    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS} 
    home_pos[5] = WRIST_ROLL_HORIZONTAL_RAW 
    home_pos[6] = GRIPPER_OPEN_RAW 

    paused = False 
    playback_speed = 3.5

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

    print("=======================================================================")
    print(f" 🚀 成功した {success_pts} 件の動作を MuJoCo 上で連続再生します")
    print(" 🔴 赤マーカー: Pick 地点 / 🟢 緑マーカー: Place 地点 (正面30cm)")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer: 
        task_idx = 0
        while viewer.is_running():
            task = success_tasks[task_idx % len(success_tasks)]
            print(f"[{task_idx + 1:03d}/{success_pts:03d}] 実行: Pick(r={task['dist_cm']}cm, θ={task['angle_deg']}°, 進入角={task['pick_pitch']:.0f}°) ➔ Place(正面30cm)")
            
            run_pick_and_place(
                viewer=viewer,
                model=model,
                data=data,
                pick_ik=task["pick_ik"],
                place_ik=task["place_ik"],
                pick_xyz=task["pick_xyz"],
                place_xyz=task["place_xyz"],
                home_pos=home_pos,
                speed_getter=lambda: playback_speed,
                paused_getter=lambda: paused
            )

            task_idx += 1
            time.sleep(0.15 / playback_speed)


if __name__ == "__main__":
    main()