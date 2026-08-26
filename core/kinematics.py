"""
==============================================================================
角度変換・運動学計算モジュール (core/kinematics.py)
==============================================================================
【役割】
1. リーダーアームの生値 (Raw: 0〜4095) をフォロワーアームの目標値 (Target: 0〜4095) へ変換
2. フォロワー目標値 (0〜4095) と MuJoCo ラジアン角 (-π〜+π) の双方向変換
3. 解析的逆運動学 (IK): 極座標 (r, theta, z) から各関節目標 Raw 値の算出
==============================================================================
"""

import math
from typing import Dict, Optional
from config.joint_config import JOINT_CONFIG, DIRECTION, SIM_OFFSETS, SIM_DIRECTIONS


# ==============================================================================
# SO-ARM100 幾何学リンク定数 (URDFより抽出: 単位はメートル)
# ==============================================================================
L0_BASE_HEIGHT = 0.1025       # 台座基準面から肩関節(Joint 2)までの垂直オフセット
L1_UPPER_ARM   = 0.1160       # 肩(Joint 2)から肘(Joint 3)までのリンク長
L2_LOWER_ARM   = 0.1350       # 肘(Joint 3)から手首(Joint 4)までのリンク長
L3_HAND_TCP    = 0.1400       # 手首(Joint 4)からグリッパー把持中心(TCP)までの実効長

# STS3215 サーボの分解能定数 (4096カウント / 360度)
COUNTS_PER_RAD = 4096.0 / (2.0 * math.pi)


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


def solve_ik_polar(
    r: float, 
    theta_deg: float, 
    z: float, 
    pitch_deg: float = -60.0
) -> Optional[Dict[int, int]]:
    """
    極座標 (r, theta, z) から SO-ARM100 の各サーボ目標 Raw 値を解析的に逆算する
    """
    # 旋回軸: 右が正 (+), 左が負 (-)[cite: 9]
    theta_rad = math.radians(-theta_deg)
    pitch_rad = math.radians(pitch_deg)

    # 1. 手首ピッチ軸 (Joint 4) の目標位置を逆算[cite: 13]
    # 肩関節(Joint 2)の幾何中心: 高さ 0.119m, 前方オフセット 0.0m[cite: 13]
    Z_SHOULDER = 0.1190
    r_w = r - L3_HAND_TCP * math.cos(pitch_rad)
    z_w = z - Z_SHOULDER - L3_HAND_TCP * math.sin(pitch_rad)

    d_sq = r_w**2 + z_w**2
    d = math.sqrt(d_sq)

    # 幾何到達判定[cite: 13]
    max_reach = L1_UPPER_ARM + L2_LOWER_ARM
    min_reach = abs(L1_UPPER_ARM - L2_LOWER_ARM)
    if d > max_reach or d < min_reach:
        return None

    # 2. 余弦定理 (Upper Arm と Lower Arm の成す三角形)[cite: 13]
    cos_alpha = (L1_UPPER_ARM**2 + d_sq - L2_LOWER_ARM**2) / (2.0 * L1_UPPER_ARM * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)

    cos_beta = (L1_UPPER_ARM**2 + L2_LOWER_ARM**2 - d_sq) / (2.0 * L1_UPPER_ARM * L2_LOWER_ARM)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)

    # 3. MuJoCo 各ジョイント qpos への直接解法[cite: 2]
    # Joint 1: 台座旋回
    qpos_1 = theta_rad

    # Joint 2: 肩ピッチ (Elbow Up: 仰角 + alpha)[cite: 2]
    phi_shoulder = math.atan2(z_w, r_w) + alpha
    qpos_2 = phi_shoulder + 1.8000

    # Joint 3: 肘ピッチ (外側へ屈曲する Elbow Up 姿勢)[cite: 2]
    qpos_3 = -(math.pi - beta)

    # Joint 4: 手首ピッチ (手先ピッチ角 pitch_rad を机面に対して維持)[cite: 2]
    qpos_4 = pitch_rad - phi_shoulder - (qpos_3 + 1.5708) + 1.0000

    # 4. ラジアンからサーボ Raw カウント値へ変換[cite: 4, 13]
    raw_targets = {
        1: radian_to_raw(1, qpos_1),
        2: radian_to_raw(2, qpos_2),
        3: radian_to_raw(3, qpos_3),
        4: radian_to_raw(4, qpos_4),
        5: SIM_OFFSETS.get(5, 3050),
        6: JOINT_CONFIG[6]["init"],
    }

    # 可動限界チェック (アーム軸 ID 1〜4)[cite: 4, 13]
    for sid in [1, 2, 3, 4]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            f_min = cfg["f_min"]
            f_max = cfg["f_max"]
            if not (f_min <= raw_targets[sid] <= f_max):
                return None

    return raw_targets