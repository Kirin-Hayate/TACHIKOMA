"""
==============================================================================
角度変換・運動学計算モジュール (core/kinematics.py)
==============================================================================
【役割】
1. リーダーアームの生値 (Raw: 0〜4095) をフォロワーアームの目標値 (Target: 0〜4095) へ変換
2. フォロワー目標値 (0〜4095) と MuJoCo ラジアン角 (-π〜+π) の双方向変換
3. MuJoCo 内蔵ヤコビアン数値IK (位置＋真下向き姿勢拘束):
   目標極座標 (r, theta, z) から各関節目標 Raw 値の算出
==============================================================================
"""

import os
import math
from typing import Dict, Optional
import numpy as np
import mujoco

from config.joint_config import JOINT_CONFIG, DIRECTION, SIM_OFFSETS, SIM_DIRECTIONS

# STS3215 サーボの分解能定数 (4096カウント / 360度)
COUNTS_PER_RAD = 4096.0 / (2.0 * math.pi)

# ==============================================================================
# MuJoCo IK 専用内部モデルの読み込み
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(BASE_DIR, "assets", "so100_scene.xml")

_IK_MODEL = None
_IK_DATA = None
_JAW_BODY_ID = -1

if os.path.exists(XML_PATH):
    _IK_MODEL = mujoco.MjModel.from_xml_path(XML_PATH)
    _IK_DATA = mujoco.MjData(_IK_MODEL)
    # 最先端の爪パーツ (jaw) のボディIDを取得
    _JAW_BODY_ID = mujoco.mj_name2id(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, "jaw")


def calculate_target(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> int:
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

        return int(max(0, min(4095, target_linear)))

    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw

        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)

        return int(new_target)


def raw_to_radian(sid: int, target_val: int) -> float:
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    diff = (target_val - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def radian_to_raw(sid: int, angle_rad: float) -> int:
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    offset = SIM_OFFSETS.get(sid, 2048)
    raw_val = int(offset + (angle_rad / direction) * COUNTS_PER_RAD)
    return max(0, min(4095, raw_val))


# 爪パーツ (jaw) のローカル座標系における先端 (TCP) オフセット [m]
JAW_LOCAL_TCP_OFFSET = np.array([0.0, -0.045, 0.0])


def solve_ik_polar(
    r: float, 
    theta_deg: float, 
    z: float, 
    max_steps: int = 40, 
    tol_pos: float = 2e-3, 
    damping: float = 0.015
) -> Optional[Dict[int, int]]:
    """
    極座標 (r, theta, z) から SO-ARM100 の「爪先端」を目標位置へ誘導する (ヤコビアンDLS法)[cite: 2]
    """
    if _IK_MODEL is None or _IK_DATA is None or _JAW_BODY_ID == -1:
        return None

    # 極座標 -> 直交座標
    theta_rad = math.radians(-theta_deg)
    target_x = r * math.sin(theta_rad)
    target_y = -r * math.cos(theta_rad)
    target_z = z
    target_pos = np.array([target_x, target_y, target_z])

    # 安定した初期姿勢 (肘上げ・前傾)[cite: 2]
    init_qpos = np.array([theta_rad, 1.4, -2.0, -0.4, 0.0, 0.0])
    _IK_DATA.qpos[:6] = init_qpos
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    jacp = np.zeros((3, _IK_MODEL.nv))

    for step in range(max_steps):
        # 爪先端 (TCP) のグローバル座標を計算 (jaw 原点 + 回転オフセット)[cite: 2]
        rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
        current_tip_pos = _IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET
        
        error = target_pos - current_tip_pos

        if np.linalg.norm(error) < tol_pos:
            break

        # 爪先端グローバル位置に対するヤコビアンを計算[cite: 2]
        mujoco.mj_jac(_IK_MODEL, _IK_DATA, jacp, None, current_tip_pos, _JAW_BODY_ID)
        J = jacp[:, :4]  # アーム4軸 (ID 1〜4)

        # DLS 逆行列計算
        J_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))
        delta_q = J_inv @ error

        delta_q = np.clip(delta_q, -0.25, 0.25)
        _IK_DATA.qpos[:4] += delta_q
        mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    # 爪先端の最終位置誤差チェック
    rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
    final_tip_pos = _IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET
    if np.linalg.norm(target_pos - final_tip_pos) > 0.020:
        return None

    raw_targets = {
        1: radian_to_raw(1, _IK_DATA.qpos[0]),
        2: radian_to_raw(2, _IK_DATA.qpos[1]),
        3: radian_to_raw(3, _IK_DATA.qpos[2]),
        4: radian_to_raw(4, _IK_DATA.qpos[3]),
        5: SIM_OFFSETS.get(5, 3050),
        6: JOINT_CONFIG[6]["init"],
    }

    # 可動限界チェック (Bounded 軸)[cite: 4]
    for sid in [1, 2, 3, 4]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            if not (cfg["f_min"] <= raw_targets[sid] <= cfg["f_max"]):
                return None

    return raw_targets