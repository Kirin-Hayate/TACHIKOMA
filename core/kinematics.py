"""
==============================================================================
角度変換・運動学計算モジュール (core/kinematics.py)
==============================================================================
【役割】
1. ラジアン ⇄ フォロワー目標 Raw 値の双方向変換 (実機通信レイヤーとの境界)
2. リーダー生値 ➔ フォロワー物理ラジアンの直接変換 (calculate_target_rad)
3. MuJoCo 内蔵ヤコビアン数値 IK: 目標極座標から各関節の目標ラジアン辞書の算出
4. 机面・地面へのめり込み防止判定 (check_ground_penetration)
==============================================================================
"""

import os
import math
from typing import Dict, Optional, Tuple
import numpy as np
import mujoco

from config.joint_config import (
    SERVO_IDS,
    JOINT_CONFIG,
    DIRECTION,
    SIM_OFFSETS,
    SIM_DIRECTIONS
)

# STS3215 サーボの分解能定数 (4096カウント / 360度)
COUNTS_PER_RAD = 4096.0 / (2.0 * math.pi)

# ==============================================================================
# MuJoCo IK 専用内部モデルの読み込み
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(BASE_DIR, "assets", "so100_scene.xml")

_IK_MODEL = None
_IK_DATA = None
_WRIST_BODY_ID = -1
_JAW_BODY_ID = -1

if os.path.exists(XML_PATH):
    _IK_MODEL = mujoco.MjModel.from_xml_path(XML_PATH)
    _IK_DATA = mujoco.MjData(_IK_MODEL)
    for name in ["wrist_pitch_link", "wrist_pitch", "wrist", "gripper"]:
        bid = mujoco.mj_name2id(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            _WRIST_BODY_ID = bid
            break
    _JAW_BODY_ID = mujoco.mj_name2id(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, "jaw")

# 爪先端 (TCP) のローカルオフセット [m]
JAW_LOCAL_TCP_OFFSET = np.array([0.0, -0.045, 0.0])

# 幾何パラメータ [m]
L_GRIPPER = 0.160            # 手首関節から爪先端までの実効長
DELTA_Z_WRIST_WP = 0.050     # 手首目標に対する上空待機オフセット
DEFAULT_APPROACH_DEG = 90.0  # 進入角度
MIN_APPROACH_DEG = 30.0

# ------------------------------------------------------------------------------
# 1. 基本単位変換 (Raw ⇄ Radian)
# ------------------------------------------------------------------------------
def raw_to_radian(sid: int, target_raw: int) -> float:
    """フォロワー Raw 値 (0-4095) を MuJoCo 物理ジョイント角度 (rad) に変換"""
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    diff = (target_raw - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def radian_to_raw(sid: int, angle_rad: float) -> int:
    """MuJoCo 物理ジョイント角度 (rad) をフォロワー Raw 値 (0-4095) に変換"""
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    offset = SIM_OFFSETS.get(sid, 2048)
    raw_val = int(round(offset + (angle_rad / direction) * COUNTS_PER_RAD))
    return int(max(0, min(4095, raw_val)))


# 標準姿勢のラジアン定数定義
WRIST_ROLL_HORIZONTAL_RAD = raw_to_radian(5, 2000)  # 約 -1.61 rad (-92.3°) # 手首ロール横挟み (90°)
GRIPPER_OPEN_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_max", 2600))
GRIPPER_CLOSE_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_min", 1400))


def get_home_radians() -> Dict[int, float]:
    """joint_config.py の init 値に基づく Home 姿勢のラジアン辞書を取得"""
    home_rad = {}
    for sid in SERVO_IDS:
        init_raw = JOINT_CONFIG[sid]["init"]
        home_rad[sid] = raw_to_radian(sid, init_raw)
    home_rad[5] = WRIST_ROLL_HORIZONTAL_RAD
    home_rad[6] = GRIPPER_OPEN_RAD
    return home_rad


# ------------------------------------------------------------------------------
# 2. リーダー生値 ➔ フォロワー目標ラジアン変換
# ------------------------------------------------------------------------------
def calculate_target_rad(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> float:
    """リーダー生値からフォロワーの目標物理角度 (rad) を直接算出"""
    config = JOINT_CONFIG[sid]
    direction = DIRECTION.get(sid, 1)

    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        r_cross = config["r_cross"]
        f_min = config["f_min"]
        f_max = config["f_max"]
        f_cross = config["f_cross"]

        raw_l = raw_leader
        if r_cross and raw_l < (r_max - 4096):
            raw_l += 4096

        if r_max == r_min:
            ratio = 0.0
        else:
            ratio = (raw_l - r_min) / (r_max - r_min)
        ratio = max(0.0, min(1.0, ratio))

        f_max_linear = f_max
        if f_cross and f_max < f_min:
            f_max_linear += 4096

        target_linear = f_min + ratio * (f_max_linear - f_min)
        if direction == -1:
            target_linear = f_min + (f_max_linear - target_linear)

        target_raw = int(max(0, min(4095, target_linear)))
        return raw_to_radian(sid, target_raw)

    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw
        # 0/4095 境界跨ぎの最短経路判定
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        # 制限をかけない仮想累積カウント
        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)

        # キャッシュ更新用として辞書に仮想カウントを直接保存
        follower_current_cache[sid] = new_target

        return raw_to_radian(sid, int(new_target))


def calculate_target(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> int:
    """既存コード互換用 (Raw値を返すラッパー)"""
    target_rad = calculate_target_rad(sid, raw_leader, prev_raw_cache, follower_current_cache)
    return radian_to_raw(sid, target_rad)


# ------------------------------------------------------------------------------
# 3. 机面めり込み検知ガード
# ------------------------------------------------------------------------------
def check_ground_penetration(min_z_threshold: float = 0.002) -> bool:
    """アームの可動リンク (body 2 以降) および爪先の床面接触を検知"""
    if _IK_MODEL is None or _IK_DATA is None:
        return False
    for i in range(2, _IK_MODEL.nbody):
        if _IK_DATA.xpos[i][2] < min_z_threshold:
            return True
    if _JAW_BODY_ID != -1:
        rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
        tcp_z = (_IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET)[2]
        if tcp_z < min_z_threshold:
            return True
    return False


# ------------------------------------------------------------------------------
# 4. 逆運動学 (IK) コアロジック (ラジアン出力)
# ------------------------------------------------------------------------------
def solve_ik_wrist_and_pitch(
    r_wrist: float, 
    theta_deg: float, 
    z_wrist: float, 
    target_pitch_deg: float,
    gripper_rad: float
) -> Optional[Dict[int, float]]:
    """手首位置・進入角・横挟み・グリッパー開閉角を指定して各関節の目標ラジアンを算出"""
    if _IK_MODEL is None or _IK_DATA is None or _WRIST_BODY_ID == -1:
        return None

    theta_rad = math.radians(-theta_deg)
    target_wrist_pos = np.array([
        r_wrist * math.sin(theta_rad),
        -r_wrist * math.cos(theta_rad),
        z_wrist
    ])

    init_qpos = np.array([theta_rad, 1.2, -1.8, -1.5, WRIST_ROLL_HORIZONTAL_RAD, 0.0])
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

    # ステップ 2: ID 4 (手首ピッチ)
    q2 = _IK_DATA.qpos[1]
    q3 = _IK_DATA.qpos[2]
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad
    _IK_DATA.qpos[3] = q4

    # ステップ 3: ID 5 & 6
    _IK_DATA.qpos[4] = WRIST_ROLL_HORIZONTAL_RAD
    _IK_DATA.qpos[5] = gripper_rad
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    if check_ground_penetration(min_z_threshold=0.002):
        return None

    rad_targets = {
        1: float(_IK_DATA.qpos[0]),
        2: float(_IK_DATA.qpos[1]),
        3: float(_IK_DATA.qpos[2]),
        4: float(_IK_DATA.qpos[3]),
        5: WRIST_ROLL_HORIZONTAL_RAD,
        6: gripper_rad,
    }

    # 実機のサーボ可動限界 (f_min, f_max) チェック
    for sid in [1, 2, 3, 4]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            raw_val = radian_to_raw(sid, rad_targets[sid])
            if not (cfg["f_min"] <= raw_val <= cfg["f_max"]):
                return None

    return rad_targets


def solve_ik_adaptive_approach(
    r_tcp: float, 
    theta_deg: float, 
    z_tcp: float, 
    gripper_rad: float,
    init_pitch_deg: float = DEFAULT_APPROACH_DEG
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[float]]:
    """適応型進入角度で目標姿勢と上空経由姿勢の目標ラジアンを計算"""
    for pitch_deg in np.arange(init_pitch_deg, MIN_APPROACH_DEG - 1.0, -5.0):
        pitch_rad = math.radians(pitch_deg)

        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = z_tcp + L_GRIPPER * math.sin(pitch_rad)

        r_wrist_wp = r_wrist_target
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP

        ik_target = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target, theta_deg=theta_deg, z_wrist=z_wrist_target, 
            target_pitch_deg=pitch_deg, gripper_rad=gripper_rad
        )
        ik_wp = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp, theta_deg=theta_deg, z_wrist=z_wrist_wp, 
            target_pitch_deg=pitch_deg, gripper_rad=gripper_rad
        )

        if ik_target is not None and ik_wp is not None:
            return ik_target, ik_wp, float(pitch_deg)

    return None, None, None