"""
==============================================================================
TACHIKOMA (SO-ARM100) ハードウェア・関節パラメータ設定ファイル
(config/joint_config.py)
==============================================================================
【役割】
実機（リーダー / フォロワー）および MuJoCo シミュレータの「可動範囲」「初期姿勢」
「回転方向」「オフセット」を一元管理します。

【環境適応機能】
Windows (COMポート) と Linux (/dev/ttyUSB*, /dev/tachikoma_arm) のシリアルポートを
自動で判別・フォールバックします。
==============================================================================
"""

import os
import platform

# ==============================================================================
# 1. 基本通信 & 制御ループ設定 (クロスプラットフォーム自動判定)
# ==============================================================================
IS_LINUX = platform.system() == "Linux"

def get_default_serial_port(role: str = "follower") -> str:
    """OSおよび接続状態に応じた適切なシリアルポート名を自動判定"""
    if IS_LINUX:
        # udev ルールで永続化されたシンボリックリンクを最優先
        if role == "follower" and os.path.exists("/dev/tachikoma_arm"):
            return "/dev/tachikoma_arm"
        # フォールバック (ttyUSB0 または ttyACM0)
        if os.path.exists("/dev/ttyUSB0"):
            return "/dev/ttyUSB0"
        if os.path.exists("/dev/ttyACM0"):
            return "/dev/ttyACM0"
        return "/dev/ttyUSB0"
    else:
        # Windows 環境のデフォルト
        return 'COM4' if role == "follower" else 'COM3'

# 環境変数 TACHIKOMA_SERIAL_PORT が指定されていれば最優先、無ければOS自動判定
FOLLOWER_PORT = os.getenv("TACHIKOMA_SERIAL_PORT", get_default_serial_port("follower"))
LEADER_PORT = os.getenv("TACHIKOMA_LEADER_PORT", get_default_serial_port("leader"))

BAUDRATE = 1000000        # STS3215サーボの通信速度（1Mbps）
SAMPLING_RATE_HZ = 50     # 制御・記録のループ周期（50Hz = 0.02秒間隔）

# 制御対象のサーボID一覧
SERVO_IDS = [1, 2, 3, 4, 5, 6]

# 実機サーボの回転方向反転フラグ (通常: 1, 反転: -1)
DIRECTION = {
    1: 1,  # ID1: 台座旋回
    2: 1,  # ID2: 肩ピッチ
    3: 1,  # ID3: 肘ピッチ
    4: 1,  # ID4: 手首ピッチ
    5: 1,  # ID5: 手首ロール
    6: 1,  # ID6: グリッパー開閉
}


# ==============================================================================
# 2. 実機リーダー・フォロワー可動範囲プロファイル (JOINT_CONFIG)
# ==============================================================================
JOINT_CONFIG = {
    # ID 1: 台座旋回 (Base) - 境界跨ぎあり
    1: {
        "type": "bounded",
        "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,
        "f_min": 850,  "f_max": 3400,        "f_cross": False,
        "init": 2130,  # 基準中心位置
    },

    # ID 2: 肩ピッチ (Shoulder) - 境界跨ぎあり
    2: {
        "type": "bounded",
        "r_min": 1715, "r_max": 4096 + 100,  "r_cross": True,
        "f_min": 942,  "f_max": 3270,        "f_cross": False,
        "init": 973,   # 折りたたみ下限初期位置
    },

    # ID 3: 肘ピッチ (Elbow)
    3: {
        "type": "bounded",
        "r_min": 900,  "r_max": 3100,        "r_cross": False,
        "f_min": 834,  "f_max": 3061,        "f_cross": False,
        "init": 3061,  # 屈曲初期位置
    },

    # ID 4: 手首ピッチ (Wrist Pitch)
    4: {
        "type": "bounded",
        "r_min": 1650, "r_max": 4015,        "r_cross": False,
        "f_min": 735,  "f_max": 3214,        "f_cross": False,
        "init": 735,   # 内側折りたたみ初期位置
    },

    # ID 5: 手首ロール (Wrist Roll: 相対追従・無限回転)
    5: {
        "type": "infinite",
        "init": 1028,  # 初期回転角
    },

    # ID 6: グリッパー開閉 (Gripper)
    6: {
        "type": "bounded",
        "r_min": 1990, "r_max": 3000,        "r_cross": False,
        "f_min": 1891, "f_max": 2833,        "f_cross": False,
        "init": 1891,  # 閉初期位置
    },
}


# ==============================================================================
# 3. MuJoCoシミュレータ整合用パラメータ
# ==============================================================================
SIM_OFFSETS = {
    1: 2130,             # ID 1: 中心位置
    2: 973,              # ID 2: 下限位置
    3: 3061,             # ID 3: 屈曲位置
    4: (735 + 4001) / 2, # ID 4: 手首がまっすぐ伸びる中立値 (約 2368)
    5: 3050,             # ID 5: 初期回転角
    6: 1837,             # ID 6: 閉位置
}

SIM_DIRECTIONS = {
    1: -1.0,  # 台座旋回方向の整合
    2:  1.0,
    3:  1.0,
    4:  1.0,
    5:  1.0,
    6:  1.0,
}