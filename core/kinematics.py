"""
==============================================================================
角度変換・運動学計算モジュール (core/kinematics.py)
==============================================================================
【役割】
1. リーダーアームの生値 (Raw: 0〜4095) をフォロワーアームの目標値 (Target: 0〜4095) へ変換
   - 有限可動軸（ID 1〜4, 6）: 線形補間（正規化 + スケーリング + 境界跨ぎ補正）
   - 無限回転軸（ID 5）: 前後フレームの差分積分（360度ループ補正）
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
L3_HAND_TCP    = 0.0950       # 手首(Joint 4)からグリッパー把持中心(TCP)までの実効長

# STS3215 サーボの分解能定数 (4096カウント / 360度)
COUNTS_PER_RAD = 4096.0 / (2.0 * math.pi)


def calculate_target(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> int:
    """
    リーダーの生値 (0〜4095) からフォロワーの目標位置 (0〜4095) を計算する

    【引数】
    - sid                    : サーボ ID (1〜6)
    - raw_leader             : リーダーから読み取った現在の生値 (0〜4095)
    - prev_raw_cache         : リーダーの前回値を保持する辞書 {sid: prev_raw} (ID5用)
    - follower_current_cache : フォロワーの現在の累積目標値を保持する辞書 {sid: current_target} (ID5用)

    【戻り値】
    - フォロワーへ送信する目標角度 (整数値)
    """
    config = JOINT_CONFIG[sid]
    direction = DIRECTION.get(sid, 1)

    # --------------------------------------------------------------------------
    # パターンA: 有限可動軸 (ID 1, 2, 3, 4, 6)
    # リーダーの可動域 [r_min, r_max] を フォロワーの可動域 [f_min, f_max] に線形マッピング
    # --------------------------------------------------------------------------
    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        r_cross = config["r_cross"]
        f_min = config["f_min"]
        f_max = config["f_max"]
        f_cross = config["f_cross"]

        # 1. リーダー側の境界跨ぎ補正 (例: 4095 を超えて 0 に戻る場合のリニア化)
        raw_l = raw_leader
        if r_cross and raw_l < (r_max - 4096):
            raw_l += 4096

        # 2. リーダーの現在位置を 0.0 〜 1.0 に正規化
        if r_max == r_min:
            ratio = 0.0
        else:
            ratio = (raw_l - r_min) / (r_max - r_min)
        # 可動範囲外へはみ出さないように 0.0〜1.0 にクリップ
        ratio = max(0.0, min(1.0, ratio))

        # 3. フォロワー側の境界跨ぎ補正
        f_max_linear = f_max
        if f_cross and f_max < f_min:
            f_max_linear += 4096

        # 4. フォロワーの目標値を線形スケーリング
        target_linear = f_min + ratio * (f_max_linear - f_min)

        # 5. 回転方向が反転設定 (-1) の場合は範囲内で反転
        if direction == -1:
            target_linear = f_min + (f_max_linear - target_linear)

        # 安全範囲 (0〜4095) に収めて整数で返す
        return int(max(0, min(4095, target_linear)))

    # --------------------------------------------------------------------------
    # パターンB: 無限回転軸 (ID 5 - 手首ロール)
    # 差分（移動量）を計算してフォロワーの目標値に累積加算（相対追従）
    # --------------------------------------------------------------------------
    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw

        # 360度（4096カウント）の境界跨ぎの差分補正（最短経路判定）
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)

        return int(new_target)


def raw_to_radian(sid: int, target_val: int) -> float:
    """
    フォロワーの目標角度 (0〜4095) を MuJoCo 用のラジアン角に変換する
    """
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)

    # 4096カウント = 360度 = 2π [rad]
    diff = (target_val - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def radian_to_raw(sid: int, angle_rad: float) -> int:
    """
    MuJoCo / 幾何学ラジアン角度をフォロワー実機サーボのRaw値 (0〜4095) に変換する
    """
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    offset = SIM_OFFSETS.get(sid, 2048)
    raw_val = int(offset + direction * (angle_rad * COUNTS_PER_RAD))
    return max(0, min(4095, raw_val))


def solve_ik_polar(
    r: float, 
    theta_deg: float, 
    z: float, 
    pitch_deg: float = -45.0
) -> Optional[Dict[int, int]]:
    """
    極座標 (r, theta, z) から SO-ARM100 の各サーボ目標 Raw 値を解析的に逆算する (Elbow Up 拘束)
    
    Parameters:
        r (float): アーム旋回中心からの水平距離 [m] (例: 0.18〜0.30)
        theta_deg (float): 旋回角度 [度] (0: 正面, 正: 左, 負: 右)
        z (float): 机上面からの高さ [m] (例: 0.02〜0.20)
        pitch_deg (float): 手先の進入ピッチ角度 [度] (0: 水平前向き, -90: 真下向き)
        
    Returns:
        Optional[Dict[int, int]]: {ID: raw_value} 辞書。物理的・幾何学的に到達不能な場合は None
    """
    theta_rad = math.radians(theta_deg)
    pitch_rad = math.radians(pitch_deg)

    # 1. ID1: 台座旋回角 (Base Pan)
    joint1_rad = theta_rad

    # 2. 手首ピッチ軸 (Joint 4) の目標位置を逆算
    # グリッパー先端 (TCP) から手首中心までのオフセットを考慮
    r_w = r - L3_HAND_TCP * math.cos(pitch_rad)
    z_w = z - L0_BASE_HEIGHT - L3_HAND_TCP * math.sin(pitch_rad)

    # 肩関節から手首中心までの直線距離
    d_sq = r_w**2 + z_w**2
    d = math.sqrt(d_sq)

    # アーム長を超える、または近すぎて届かない場合の判定
    if d > (L1_UPPER_ARM + L2_LOWER_ARM) or d < abs(L1_UPPER_ARM - L2_LOWER_ARM):
        return None  # 物理的に到達不能

    # 3. 余弦定理により Upper Arm と Lower Arm の成す角度を計算 (Elbow Up 拘束)
    cos_alpha = (L1_UPPER_ARM**2 + d_sq - L2_LOWER_ARM**2) / (2.0 * L1_UPPER_ARM * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)

    cos_beta = (L1_UPPER_ARM**2 + L2_LOWER_ARM**2 - d_sq) / (2.0 * L1_UPPER_ARM * L2_LOWER_ARM)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)

    # 4. 各関節角度の幾何学的決定
    # ID2: 肩ピッチ (Shoulder Lift)
    base_elevation = math.atan2(z_w, r_w)
    joint2_rad = base_elevation + alpha

    # ID3: 肘ピッチ (Elbow Flex) - Elbow Up のため常に屈曲方向
    joint3_rad = -(math.pi - beta)

    # ID4: 手首ピッチ (Wrist Flex) - 手先のピッチ角度を維持
    joint4_rad = pitch_rad - (joint2_rad + joint3_rad)

    # 5. ラジアンからサーボ Raw カウント値へ変換
    raw_targets = {
        1: radian_to_raw(1, joint1_rad),
        2: radian_to_raw(2, joint2_rad),
        3: radian_to_raw(3, joint3_rad),
        4: radian_to_raw(4, joint4_rad),
        5: SIM_OFFSETS[5],  # ロールは中立維持
        6: SIM_OFFSETS[6],  # グリッパー状態維持
    }

    # 可動限界チェック (Bounded 軸のみ)
    for sid, target in raw_targets.items():
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            f_min = cfg["f_min"]
            f_max = cfg["f_max"]
            # 範囲外なら到達不能として None を返す
            if not (f_min <= target <= f_max):
                return None

    return raw_targets