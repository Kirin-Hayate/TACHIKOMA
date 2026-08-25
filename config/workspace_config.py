"""
作業空間の角度・位置マッピング設定 (config/workspace_config.py)
"""

# ID1 の基準値 (正面 = 2048)
# 4096 = 360度 なので 1度 ≒ 11.37 raw
CENTER_RAW = 2048
RAW_PER_DEG = 4096.0 / 360.0

# よく使う目標地点の角度定義（度数法：正面が0度、右がマイナス、左がプラス）
LOCATION_ANGLES = {
    "A": -35.0,        # 右前方
    "B": 0.0,          # 正面
    "C": 35.0,         # 左前方
    "右": -45.0,
    "正面": 0.0,
    "中央": 0.0,
    "左": 45.0,
    "1時方向": 60.0,
    "2時方向": 30.0,
    "12時方向": 0.0,
    "11時方向": -30.0,
    "10時方向": -60.0,
}

def deg_to_raw(deg: float) -> int:
    """度数（-90度〜+90度）を ID1 のサーボ値に変換"""
    val = int(CENTER_RAW + (deg * RAW_PER_DEG))
    return max(500, min(3500, val))  # 安全リミット

def location_to_raw(name_or_deg) -> int:
    """地点名または度数から ID1 の raw 値を取得"""
    if isinstance(name_or_deg, (int, float)):
        return deg_to_raw(float(name_or_deg))
    if str(name_or_deg) in LOCATION_ANGLES:
        return deg_to_raw(LOCATION_ANGLES[str(name_or_deg)])
    return CENTER_RAW