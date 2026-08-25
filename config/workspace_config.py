"""
作業空間の角度・位置マッピング設定 (config/workspace_config.py)
【角度規約】
- 正面: 0度
- 右方向: ＋（プラス）
- 左方向: −（マイナス）
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import JOINT_CONFIG

# ID1 の基準中心位置 (2130) と可動域限界
CENTER_RAW = JOINT_CONFIG[1]["init"]      # 2130 (0度)
MIN_RAW = JOINT_CONFIG[1]["f_min"]        # 850 (最大左)
MAX_RAW = JOINT_CONFIG[1]["f_max"]        # 3400 (最大右)

RAW_PER_DEG = 4096.0 / 360.0  # 1度あたり約 11.378 raw

# 物理的な最大角度（度数法）: 右が +111.6度、左が -112.5度
MAX_RIGHT_DEG = (MAX_RAW - CENTER_RAW) / RAW_PER_DEG   # +111.6度 (右限界)
MAX_LEFT_DEG = (MIN_RAW - CENTER_RAW) / RAW_PER_DEG    # -112.5度 (左限界)

# 地点・方向の角度定義（右: ＋, 左: −）
LOCATION_ANGLES = {
    # 限界・極端な指定
    "最大右": MAX_RIGHT_DEG,
    "右限界": MAX_RIGHT_DEG,
    "右端": MAX_RIGHT_DEG,
    "最大左": MAX_LEFT_DEG,
    "左限界": MAX_LEFT_DEG,
    "左端": MAX_LEFT_DEG,

    # 一般的な方位
    "右": 60.0,
    "右前方": 35.0,
    "正面": 0.0,
    "中央": 0.0,
    "左前方": -35.0,
    "左": -60.0,

    # 時計方位（右側/時計回りがプラス）
    "1時方向": 30.0,
    "2時方向": 60.0,
    "3時方向": MAX_RIGHT_DEG,
    "12時方向": 0.0,
    "11時方向": -30.0,
    "10時方向": -60.0,
    "9時方向": MAX_LEFT_DEG,

    # アルファベット地点（右側がA、左側がC）
    "A": 45.0,
    "B": 0.0,
    "C": -45.0,
}

def deg_to_raw(deg: float) -> int:
    """度数（右: +, 左: -）を ID1 のサーボ値に変換"""
    # ★ 右 (+) ほど Raw 値が増加、左 (-) ほど Raw 値が減少するように修正
    val = int(CENTER_RAW + (deg * RAW_PER_DEG))
    return max(MIN_RAW, min(MAX_RAW, val))  # joint_configの安全限界でクランプ

def location_to_raw(name_or_deg) -> int:
    """地点名または度数から ID1 の raw 値を取得"""
    if isinstance(name_or_deg, (int, float)):
        return deg_to_raw(float(name_or_deg))
    
    name_str = str(name_or_deg).strip()
    if name_str in LOCATION_ANGLES:
        return deg_to_raw(LOCATION_ANGLES[name_str])
    
    clean_str = name_str.replace("度", "").replace("deg", "").strip()
    try:
        deg_val = float(clean_str)
        return deg_to_raw(deg_val)
    except ValueError:
        pass

    return CENTER_RAW