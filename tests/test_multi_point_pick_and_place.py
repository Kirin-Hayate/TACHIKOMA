"""
==============================================================================
多地点 Pick & Place 動作確認テスト
(tests/test_multi_point_pick_and_place.py)
==============================================================================
【動作仕様】
  - test_gripper_and_roll.py の動作実績がある IK ロジックを完全踏襲。
  - 事前に定義した複数の (Pick地点, Place地点) ペアを巡回実行。
  - Pick地点 (赤球) と Place地点 (緑球) を同時に画面上に可視化。

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
# 🛠️ 幾何・把持パラメータ (正常動作品と完全一致)
# ==============================================================================
L_GRIPPER = 0.160            # 手首関節から爪先端までの実効長 [m]
DELTA_Z_WRIST_WP = 0.050     # 手首目標に対する上空待機オフセット [m]
DEFAULT_APPROACH_DEG = 90.0  # 進入角度
MIN_APPROACH_DEG = 30.0

# ID 5 (手首ロール): 初期位置から +90° 回転させた Raw カウント値
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

    # ステップ 2: 手首 (ID 4) のピッチ角代入
    q2 = _IK_DATA.qpos[1]
    q3 = _IK_DATA.qpos[2]
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad
    _IK_DATA.qpos[3] = q4

    # ステップ 3: ID 5 (手首ロール90°回転) と ID 6 (グリッパー) を反映
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


def execute_trajectory(viewer, model, data, start_pos, end_pos, pick_xyz, place_xyz, steps=25, speed=2.0, paused_flag_getter=None):
    """滑らかな姿勢遷移と Pick/Place 両マーカー球の常時表示"""
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
        
        # 🔴 Pick 地点マーカー球 (赤)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=pick_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.15, 0.15, 0.8]
        )
        # 🟢 Place 地点マーカー球 (緑)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=place_xyz,
            mat=np.eye(3).flatten(),
            rgba=[0.15, 1.0, 0.25, 0.8]
        )
        viewer.user_scn.ngeom = 2
        viewer.sync()

        time.sleep(0.015 / speed)


def run_pick_and_place_task(viewer, model, data, pick_target, place_target, home_pos, speed_getter, paused_getter):
    """指定された Pick/Place 座標ペアに対する一連の動作を実行"""
    r_pk, th_pk, z_pk = pick_target
    r_pl, th_pl, z_pl = place_target

    th_pk_rad = math.radians(-th_pk)
    pick_xyz = np.array([r_pk * math.sin(th_pk_rad), -r_pk * math.cos(th_pk_rad), z_pk])
    
    th_pl_rad = math.radians(-th_pl)
    place_xyz = np.array([r_pl * math.sin(th_pl_rad), -r_pl * math.cos(th_pl_rad), z_pl])

    # 1. 姿勢計算 (Pick 側)
    ik_pk_target_open, ik_pk_wp_open, pitch_pk, _, _ = solve_ik_adaptive_approach(
        r_tcp=r_pk, theta_deg=th_pk, z_tcp=z_pk, gripper_raw=GRIPPER_OPEN_RAW
    )
    if ik_pk_target_open is None:
        print(f"❌ [到達不可] Pick 座標: r={r_pk*1000:.0f}mm, θ={th_pk:.1f}°")
        return False

    # 2. 姿勢計算 (Place 側)
    ik_pl_target_open, ik_pl_wp_open, pitch_pl, _, _ = solve_ik_adaptive_approach(
        r_tcp=r_pl, theta_deg=th_pl, z_tcp=z_pl, gripper_raw=GRIPPER_OPEN_RAW
    )
    if ik_pl_target_open is None:
        print(f"❌ [到達不可] Place 座標: r={r_pl*1000:.0f}mm, θ={th_pl:.1f}°")
        return False

    print(f"✅ 姿勢解計算成功 | Pick角: {pitch_pk:.1f}° ➔ Place角: {pitch_pl:.1f}°")

    ik_pk_target_closed = ik_pk_target_open.copy()
    ik_pk_target_closed[6] = GRIPPER_CLOSE_RAW
    ik_pk_wp_closed = ik_pk_wp_open.copy()
    ik_pk_wp_closed[6] = GRIPPER_CLOSE_RAW

    ik_pl_target_closed = ik_pl_target_open.copy()
    ik_pl_target_closed[6] = GRIPPER_CLOSE_RAW
    ik_pl_wp_closed = ik_pl_wp_open.copy()
    ik_pl_wp_closed[6] = GRIPPER_CLOSE_RAW

    # --- 実行シーケンス ---
    # [1] Pick 上空へアプローチ (開)
    execute_trajectory(viewer, model, data, home_pos, ik_pk_wp_open, pick_xyz, place_xyz, speed=speed_getter(), paused_flag_getter=paused_getter)
    
    # [2] 把持点へ降下 (開)
    execute_trajectory(viewer, model, data, ik_pk_wp_open, ik_pk_target_open, pick_xyz, place_xyz, steps=18, speed=speed_getter(), paused_flag_getter=paused_getter)
    time.sleep(0.15 / speed_getter())

    # [3] 把持 (閉)
    execute_trajectory(viewer, model, data, ik_pk_target_open, ik_pk_target_closed, pick_xyz, place_xyz, steps=12, speed=speed_getter(), paused_flag_getter=paused_getter)
    time.sleep(0.2 / speed_getter())

    # [4] 上空へ持ち上げ (閉)
    execute_trajectory(viewer, model, data, ik_pk_target_closed, ik_pk_wp_closed, pick_xyz, place_xyz, steps=18, speed=speed_getter(), paused_flag_getter=paused_getter)
    time.sleep(0.15 / speed_getter())

    # [5] Place 上空へ旋回移動 (閉)
    execute_trajectory(viewer, model, data, ik_pk_wp_closed, ik_pl_wp_closed, pick_xyz, place_xyz, steps=35, speed=speed_getter(), paused_flag_getter=paused_getter)

    # [6] 配置点へ降下 (閉)
    execute_trajectory(viewer, model, data, ik_pl_wp_closed, ik_pl_target_closed, pick_xyz, place_xyz, steps=18, speed=speed_getter(), paused_flag_getter=paused_getter)
    time.sleep(0.15 / speed_getter())

    # [7] 開放 (開)
    execute_trajectory(viewer, model, data, ik_pl_target_closed, ik_pl_target_open, pick_xyz, place_xyz, steps=12, speed=speed_getter(), paused_flag_getter=paused_getter)
    time.sleep(0.2 / speed_getter())

    # [8] 上空退避 ➔ Home 復帰 (開)
    execute_trajectory(viewer, model, data, ik_pl_target_open, ik_pl_wp_open, pick_xyz, place_xyz, steps=18, speed=speed_getter(), paused_flag_getter=paused_getter)
    execute_trajectory(viewer, model, data, ik_pl_wp_open, home_pos, pick_xyz, place_xyz, speed=speed_getter(), paused_flag_getter=paused_getter)
    time.sleep(0.4 / speed_getter())

    return True


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
        elif 321 <= keycode <= 329:  # テンキー 1〜9
            playback_speed = float(keycode - 320)
            print(f"\n⏩ 再生速度: {playback_speed:.1f}倍速")

    # ==========================================================================
    # 📍 テストする座標ペアリスト: (r [m], theta [deg], z [m])
    # ==========================================================================
    """
    test_tasks = [
        # パターン 1: 近距離・対称旋回 (右前 ➔ 左前)
        {
            "name": "タスク 1 (近距離 対称旋回: -30° ➔ +30°)",
            "pick":  (0.200, -30.0, 0.015),
            "place": (0.200,  30.0, 0.015),
        },
        # パターン 2: 正面から左サイドへの長距離移載 (正面 ➔ 左側深め)
        {
            "name": "タスク 2 (正面 ➔ 左前深め: 0° ➔ +45°)",
            "pick":  (0.240,   0.0, 0.015),
            "place": (0.180,  45.0, 0.015),
        },
        # パターン 3: 遠方からの引き込み把持 (遠方アプローチ 80° ➔ 手前)
        {
            "name": "タスク 3 (遠方適応アプローチ ➔ 手前右: -15° ➔ -45°)",
            "pick":  (0.320, -15.0, 0.015),
            "place": (0.180, -45.0, 0.015),
        },
        # パターン 4: 広角大移動 (右奥 ➔ 左奥)
        {
            "name": "タスク 4 (広角大移動: -40° ➔ +40°)",
            "pick":  (0.260, -40.0, 0.015),
            "place": (0.260,  40.0, 0.015),
        },
    ]
    """
    test_tasks = [
        
        {
            "name": "Test",
            "pick":  (0.450, -30.0, 0.015),
            "place": (0.200,  30.0, 0.015),
        }
    ]

    home_pos = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}
    home_pos[5] = WRIST_ROLL_HORIZONTAL_RAW
    home_pos[6] = GRIPPER_OPEN_RAW

    print("=======================================================================")
    print(" 🚀 多地点 Pick & Place 連続動作検証テスト")
    print(" 🔴 赤マーカー: Pick 地点 / 🟢 緑マーカー: Place 地点")
    print(" [Space]: 一時停止 / [1]〜[9]: 速度変更 / [Esc]: 終了")
    print("=======================================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        task_idx = 0
        while viewer.is_running():
            current_task = test_tasks[task_idx % len(test_tasks)]
            print(f"\n▶ 実行中: {current_task['name']}")

            success = run_pick_and_place_task(
                viewer=viewer,
                model=model,
                data=data,
                pick_target=current_task["pick"],
                place_target=current_task["place"],
                home_pos=home_pos,
                speed_getter=lambda: playback_speed,
                paused_getter=lambda: paused
            )

            task_idx += 1
            time.sleep(0.5 / playback_speed)


if __name__ == "__main__":
    main()