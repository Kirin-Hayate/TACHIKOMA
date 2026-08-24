"""
==============================================================================
角度変換・運動学計算モジュール (core/kinematics.py)
==============================================================================
【役割】
1. リーダーアームの生値 (Raw: 0〜4095) をフォロワーアームの目標値 (Target: 0〜4095) へ変換
   - 有限可動軸（ID 1〜4, 6）: 線形補間（正規化 + スケーリング + 境界跨ぎ補正）
   - 無限回転軸（ID 5）: 前後フレームの差分積分（360度ループ補正）
2. フォロワー目標値 (0〜4095) を MuJoCo シミュレータ用のラジアン角 (-π〜+π) へ変換
==============================================================================
"""

import math
from config.joint_config import JOINT_CONFIG, DIRECTION, SIM_OFFSETS, SIM_DIRECTIONS


def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
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
        # 前回のリーダー値を取得（初回は現在の値を使用）
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw

        # 360度（4096カウント）の境界跨ぎの差分補正（最短経路判定）
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        # フォロワーの現在目標値に移動量を加算
        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)

        # ※急反転バグを防ぐため、剰余 (%) を取らずに連続値のまま返す
        return int(new_target)


def raw_to_radian(sid, target_val):
    """
    フォロワーの目標角度 (0〜4095) を MuJoCo 用のラジアン角に変換する

    【変換式】
    radian = (target_val - SIM_OFFSETS[sid]) * SIM_DIRECTIONS[sid] * (2π / 4096)
    """
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)

    # 4096カウント = 360度 = 2π [rad]
    diff = (target_val - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)