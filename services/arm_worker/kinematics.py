"""
==============================================================================
TACHIKOMA 角度変換・運動学・自重たわみ補正・非対称把持モジュール (core/kinematics.py)
==============================================================================
【モジュールの役割と構成】
本モジュールは、マニピュレータの幾何計算・座標変換・把持計画・実機通信インターフェースを担います。
HeroBox (Celeron J4125 等の AVX 非対応 CPU) によるヘッドレス運用環境と、
GPU/AVX を備えた開発用 PC (MuJoCo 3D シミュレータ稼働環境) の両方で同一コードベースとして
動作するよう、MuJoCo ライブラリの動的フォールバック設計を内包しています。

1. 動的エンジンフォールバックレイヤー
   - Celeron などの AVX 非対応 CPU 環境で発生する `Illegal instruction (コアダンプ)` を回避
   - MuJoCo が利用可能な環境では XML モデルを用いたヤコビアン IK 計算・接地判定を有効化
   - MuJoCo が利用できないヘッドレス環境では、純粋な幾何学 IK・角度変換・サーボ通信のみで稼働

2. 単位変換レイヤー (Raw ⇄ Radian)
   - raw_to_radian: サーボ生値 (0〜4095) を物理ジョイント角度 (rad) に変換
   - radian_to_raw: 物理ジョイント角度 (rad) をサーボ生値 (0〜4095) に変換
   - get_home_radians: ホーム姿勢の標準関節角度辞書を取得

3. リーダー追従計算レイヤー (テレオペ用)
   - calculate_target_rad: 操作側リーダー生値から追従目標ラジアンを計算 (ID5 相対追従対応)
   - calculate_target: 既存 Raw 値インターフェース互換ラッパー

4. 物理たわみ補正レイヤー (2次元完全連成2次多項式モデル)
   - calculate_sag_compensation: リーチ r と旋回角 θ から実機アームの自重沈み込み量 Δz を推計

5. 機構干渉・接地判定レイヤー
   - check_ground_penetration: アーム各リンクおよび爪先端の机面めり込みを検知

6. 逆運動学 (IK) ソルバーレイヤー (ヨー角アライメント ＆ 非対称把持対応)
   - solve_ik_wrist_and_pitch: 手首目標座標・進入ピッチ角・手首ロール角から 6 軸関節角を算出
   - solve_ik_tabletop_grasp: 机上直交座標 (X, Y, Z) と物体の傾き角 (angle_deg) を受け取り、
     手首ロール角のアライメントおよび非対称爪（固定爪干渉回避）の偏心補正を自動計算した上で、
     最適な把持姿勢 (grasp) と上空待機姿勢 (waypoint) の安全な姿勢ペアを出力
   - solve_ik_tabletop_place: 接地目標座標へのアプローチおよび安全なリリース姿勢を算出
==============================================================================
"""

import os
import math
from typing import Dict, Optional, Tuple
import numpy as np

# ==============================================================================
# MuJoCo の安全な動的インポート (AVX 非対応 CPU 対策)
# ==============================================================================
# Intel Celeron J4125 などの AVX 命令非対応 CPU では、公式 MuJoCo バイナリを
# インポートした瞬間に Illegal instruction (コアダンプ) が発生します。
# これを防ぐため、インポート失敗時は None として安全にフォールバックします。
try:
    import mujoco
    HAS_MUJOCO = True
except Exception:
    mujoco = None
    HAS_MUJOCO = False

from config.joint_config import (
    SERVO_IDS,
    JOINT_CONFIG,
    DIRECTION,
    SIM_OFFSETS,
    SIM_DIRECTIONS
)

# STS3215 サーボの分解能定数 (4096カウント / 360度 = 約 651.8986 counts/rad)
COUNTS_PER_RAD = 4096.0 / (2.0 * math.pi)

# ==============================================================================
# MuJoCo IK 専用内部モデルの読み込み (HAS_MUJOCO 有効時のみ初期化)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XML_PATH = os.path.join(BASE_DIR, "assets", "so100_scene.xml")

_IK_MODEL = None
_IK_DATA = None
_WRIST_BODY_ID = -1
_JAW_BODY_ID = -1

# MuJoCo が正常にインポートできた場合のみ、XML モデルと各リンク ID を読み込む
if HAS_MUJOCO and os.path.exists(XML_PATH):
    try:
        _IK_MODEL = mujoco.MjModel.from_xml_path(XML_PATH)
        _IK_DATA = mujoco.MjData(_IK_MODEL)
        for name in ["wrist_pitch_link", "wrist_pitch", "wrist", "gripper"]:
            bid = mujoco.mj_name2id(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                _WRIST_BODY_ID = bid
                break
        _JAW_BODY_ID = mujoco.mj_name2id(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, "jaw")
    except Exception as e:
        # モデル読み込み時の例外発生時もクラッシュさせずに無効化
        _IK_MODEL = None
        _IK_DATA = None
        HAS_MUJOCO = False

# 爪先端 (TCP: Tool Center Point) のローカルオフセット [m]
JAW_LOCAL_TCP_OFFSET = np.array([0.0, -0.045, 0.0])

# 幾何定数 [m]
L_GRIPPER = 0.160            # 手首ピッチ回転軸から爪先端把持点までの実効長 (160mm)
DELTA_Z_WRIST_WP = 0.035     # 把持点に対する上空アプローチ待機点の垂直マージン (+50mm)
DEFAULT_APPROACH_DEG = 90.0  # デフォルト進入角 (90°: 真上からの垂直降下)
MIN_APPROACH_DEG = 5.0       # 許容する最小進入角 (5°: 斜めアプローチ限界)

# SO-ARM100 非対称爪補正定数 [m]
# 固定爪を物体外縁から逃がすためのローカル側方偏心マージン
GRIPPER_ASYM_OFFSET_M = 0.022


# ==============================================================================
# 1. 単位変換レイヤー (Raw ⇄ Radian)
# ==============================================================================
def raw_to_radian(sid: int, target_raw: int) -> float:
    """サーボの生カウント値 (0〜4095) を物理ジョイントラジアン (rad) に変換"""
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    diff = (target_raw - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def radian_to_raw(sid: int, angle_rad: float) -> int:
    """物理ジョイントラジアン (rad) をサーボ生カウント値 (0〜4095) に変換"""
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    offset = SIM_OFFSETS.get(sid, 2048)
    raw_val = int(round(offset + (angle_rad / direction) * COUNTS_PER_RAD))
    return int(max(0, min(4095, raw_val)))


# 標準姿勢定数
WRIST_ROLL_HORIZONTAL_RAD = raw_to_radian(5, 2000)                         # 水平基準姿勢 (約 -1.61 rad)
GRIPPER_OPEN_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_max", 2600))   # グリッパー全開
GRIPPER_CLOSE_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_min", 1400))  # グリッパー完全把持


def get_home_radians() -> Dict[int, float]:
    """直立退避・ホーム姿勢における各軸の物理ラジアン角辞書を取得"""
    home_rad = {}
    for sid in SERVO_IDS:
        init_raw = JOINT_CONFIG[sid]["init"]
        home_rad[sid] = raw_to_radian(sid, init_raw)
    home_rad[5] = WRIST_ROLL_HORIZONTAL_RAD
    home_rad[6] = GRIPPER_OPEN_RAD
    return home_rad


# ==============================================================================
# 2. リーダー追従計算レイヤー (テレオペ用)
# ==============================================================================
def calculate_target_rad(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> float:
    """リーダー側アームの現在生値を受け取り、フォロワー側の追従目標ラジアンを算出"""
    config = JOINT_CONFIG[sid]
    direction = DIRECTION.get(sid, 1)

    # 有限回転軸の処理 (線形等倍マッピング)
    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        r_cross = config["r_cross"]
        f_min = config["f_min"]
        f_max = config["f_max"]
        f_cross = config["f_cross"]

        raw_l = raw_leader
        # 0/4096 境界跨ぎの連続値展開
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

    # 無限回転軸の処理 (差分・相対追従)
    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw

        # ラップアラウンド境界の補正
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)
        follower_current_cache[sid] = new_target

        return raw_to_radian(sid, int(new_target))


def calculate_target(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> int:
    """Raw 生値ベースの追従計算ラッパー関数"""
    target_rad = calculate_target_rad(sid, raw_leader, prev_raw_cache, follower_current_cache)
    return radian_to_raw(sid, target_rad)


# ==============================================================================
# 3. 物理たわみ補正レイヤー (2次元完全連成2次多項式モデル)
# ==============================================================================
def calculate_sag_compensation(r_m: float, theta_rad: float) -> float:
    """
    アームの水平リーチ r [m] と旋回角 theta [rad] から、
    実機アームの自重による机面沈み込み量 Δz [m] を推計して目標 Z 座標に補加する補正値を算出
    """
    r_c = max(0.15, min(0.40, r_m))
    th_sq = theta_rad ** 2

    # 実測キャリブレーションデータからフィッティングした 2 次多項式モデル [mm]
    delta_z_mm = (
        959.9708 * (r_c ** 2)
        - 211.4899 * r_c
        - 3.4854 * th_sq
        + 45.5363 * (r_c * th_sq)
        + 36.2444
    )
    delta_z_mm = max(0.0, min(20.0, delta_z_mm))
    return delta_z_mm / 1000.0


# ==============================================================================
# 4. 機構干渉・接地判定レイヤー
# ==============================================================================
def check_ground_penetration(min_z_threshold: float = -0.005) -> Tuple[bool, str]:
    """
    アーム各部および爪先端の机面めり込みを検知
    戻り値: (衝突有無 bool, 衝突箇所の詳細メッセージ str)
    """
    # MuJoCo モデルがない環境では机面干渉チェックを安全にスキップ
    if _IK_MODEL is None or _IK_DATA is None:
        return False, "OK (Headless mode)"

    # 1. アーム各リンク (body 2 以降) のめり込み判定
    for i in range(2, _IK_MODEL.nbody):
        bname = mujoco.mj_id2name(_IK_MODEL, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        z_val = _IK_DATA.xpos[i][2]
        # 机面より 5mm 以上深く沈み込んでいる場合のみ衝突と判定
        if z_val < min_z_threshold:
            return True, f"{bname} が沈下 (Z={z_val*1000:+.1f}mm < {min_z_threshold*1000:.0f}mm)"

    # 2. 爪先端 (TCP)
    if _JAW_BODY_ID != -1:
        rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
        tcp_z = (_IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET)[2]
        if tcp_z < min_z_threshold:
            return True, f"爪先端(TCP) が沈下 (Z={tcp_z*1000:+.1f}mm < {min_z_threshold*1000:.0f}mm)"

    return False, "OK"


# ==============================================================================
# 5. 逆運動学 (IK) ソルバーレイヤー (ヨー角・非対称把持対応)
# ==============================================================================
def _solve_ik_geometric_analytical(
    r_wrist: float,
    theta_deg: float,
    z_wrist: float,
    target_pitch_deg: float,
    wrist_roll_rad: float,
    gripper_rad: float
) -> Optional[Dict[int, float]]:
    """
    MuJoCo 非稼働環境（Celeron / ヘッドレス環境）用の純粋数理 IK ソルバー。
    SO-ARM100 の 3 軸平面リンク幾何（肩・肘・手首ピッチ）を余弦定理により解きます。
    """
    # リンク幾何長 [m]
    L1 = 0.1025   # ベース旋回軸から肩リフト軸までの垂直オフセット
    L2 = 0.11257  # 上腕リンク長
    L3 = 0.1349   # 前腕リンク長

    # ベース旋回角 (ID 1)
    q1 = math.radians(-theta_deg)

    # 肩軸基準の手首相対位置
    r = r_wrist
    z = z_wrist - L1
    D_sq = r**2 + z**2
    D = math.sqrt(D_sq)

    # 到達限界チェック
    if D > (L2 + L3) or D < abs(L2 - L3):
        return None

    # 肘関節角 q3 (余弦定理)
    cos_q3 = (D_sq - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3 = -math.acos(cos_q3)  # 下エルボー姿勢

    # 肩関節角 q2
    alpha = math.atan2(z, r)
    beta = math.atan2(L3 * math.sin(-q3), L2 + L3 * math.cos(-q3))
    q2 = math.pi / 2.0 - (alpha + beta)

    # 手首ピッチ角 q4 (目標進入角を維持)
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad

    return {
        1: float(q1),
        2: float(q2),
        3: float(q3),
        4: float(q4),
        5: float(wrist_roll_rad),
        6: float(gripper_rad),
    }


def solve_ik_wrist_and_pitch(
    r_wrist: float, 
    theta_deg: float, 
    z_wrist: float, 
    target_pitch_deg: float,
    gripper_rad: float,
    wrist_roll_rad: float = WRIST_ROLL_HORIZONTAL_RAD,
    init_qpos_custom: Optional[np.ndarray] = None,
    return_reason: bool = False
) -> Tuple[Optional[Dict[int, float]], str]:
    """
    手首ピッチ軸の位置・ピッチ角・ロール角から 6 軸角度を算出。
    MuJoCo が利用可能な場合はヤコビアン反復法、利用できない場合は数理幾何ソルバーに自動切替。
    """
    # --------------------------------------------------------------------------
    # パターン A: MuJoCo が存在しない環境 (数理幾何ソルバーによるフォールバック)
    # --------------------------------------------------------------------------
    if not HAS_MUJOCO or _IK_MODEL is None or _IK_DATA is None or _WRIST_BODY_ID == -1:
        rad_targets = _solve_ik_geometric_analytical(
            r_wrist, theta_deg, z_wrist, target_pitch_deg, wrist_roll_rad, gripper_rad
        )
        if rad_targets is None:
            msg = f"幾何学到達限界外 (手首到達距離: r={r_wrist*1000:.1f}mm, z={z_wrist*1000:.1f}mm)"
            return (None, msg) if return_reason else None

        # サーボ可動域リミットチェック (ID 1〜5)
        for sid in [1, 2, 3, 4, 5]:
            cfg = JOINT_CONFIG.get(sid)
            if cfg and cfg["type"] == "bounded":
                raw_val = radian_to_raw(sid, rad_targets[sid])
                if not (cfg["f_min"] <= raw_val <= cfg["f_max"]):
                    diff = raw_val - cfg["f_max"] if raw_val > cfg["f_max"] else cfg["f_min"] - raw_val
                    bound_type = "上限超過" if raw_val > cfg["f_max"] else "下限未満"
                    msg = f"サーボ ID{sid} 可動域外 ({bound_type} +{diff}ステップ: Raw={raw_val}, 許容: {cfg['f_min']}〜{cfg['f_max']})"
                    return (None, msg) if return_reason else None
        
        # ⚠️ 【追加】自己干渉防止: 肩関節 (ID 2) が後ろに倒れ込みすぎる姿勢 (Raw < 1350) を除外
        raw_id2 = radian_to_raw(2, rad_targets[2])
        if raw_id2 < 1350:
            msg = f"自己干渉リスク (肩ID2が後方反りすぎ: Raw={raw_id2} < 1350)"
            return (None, msg) if return_reason else None

        return (rad_targets, "OK") if return_reason else rad_targets

    # --------------------------------------------------------------------------
    # パターン B: MuJoCo 利用可能環境 (ヤコビアン反復 IK による精密計算)
    # --------------------------------------------------------------------------
    theta_rad = math.radians(-theta_deg)
    target_wrist_pos = np.array([
        r_wrist * math.sin(theta_rad),
        -r_wrist * math.cos(theta_rad),
        z_wrist
    ])

    if init_qpos_custom is not None:
        init_qpos = np.copy(init_qpos_custom)
    else:
        init_qpos = np.array([theta_rad, 1.2, -1.8, -1.5, wrist_roll_rad, 0.0])

    _IK_DATA.qpos[:6] = init_qpos
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    jacp = np.zeros((3, _IK_MODEL.nv))
    max_iters = 20 if init_qpos_custom is not None else 40

    for _ in range(max_iters):
        current_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
        error = target_wrist_pos - current_wrist_pos

        if np.linalg.norm(error) < 1.5e-3:
            break

        mujoco.mj_jacBody(_IK_MODEL, _IK_DATA, jacp, None, _WRIST_BODY_ID)
        J = jacp[:, :3]
        J_inv = J.T @ np.linalg.inv(J @ J.T + (0.02**2) * np.eye(3))
        delta_q = np.clip(J_inv @ error, -0.25, 0.25)

        _IK_DATA.qpos[:3] += delta_q
        mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    # 位置収束判定
    final_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
    dist_err = np.linalg.norm(target_wrist_pos - final_wrist_pos)
    if dist_err > 0.020:
        msg = f"手首位置の収束失敗 (誤差: {dist_err*1000:.1f}mm > 20mm)"
        return (None, msg) if return_reason else None

    # 手首ピッチ・ロール姿勢拘束
    q2 = _IK_DATA.qpos[1]
    q3 = _IK_DATA.qpos[2]
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad
    _IK_DATA.qpos[3] = q4
    _IK_DATA.qpos[4] = wrist_roll_rad
    _IK_DATA.qpos[5] = gripper_rad
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    rad_targets = {
        1: float(_IK_DATA.qpos[0]),
        2: float(_IK_DATA.qpos[1]),
        3: float(_IK_DATA.qpos[2]),
        4: float(_IK_DATA.qpos[3]),
        5: float(wrist_roll_rad),
        6: float(gripper_rad),
    }

    # サーボ可動域リミットチェック (ID 1〜5)
    for sid in [1, 2, 3, 4, 5]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            raw_val = radian_to_raw(sid, rad_targets[sid])
            if not (cfg["f_min"] <= raw_val <= cfg["f_max"]):
                diff = raw_val - cfg["f_max"] if raw_val > cfg["f_max"] else cfg["f_min"] - raw_val
                bound_type = "上限超過" if raw_val > cfg["f_max"] else "下限未満"
                msg = f"サーボ ID{sid} 可動域外 ({bound_type} +{diff}ステップ: Raw={raw_val}, 許容: {cfg['f_min']}〜{cfg['f_max']})"
                return (None, msg) if return_reason else None

    return (rad_targets, "OK") if return_reason else rad_targets


# 爪構造幾何定数 [m]
FIXED_JAW_OFFSET_M = 0.020    # 手先中心軸から固定爪先端までの距離 (短手方向: 20mm)
LANDING_MARGIN_M = 0.015      # 物体外縁から固定爪着地点までの安全マージン (短手方向: 15mm)


def solve_ik_tabletop_grasp(
    x_phys_mm: float,
    y_phys_mm: float,
    z_phys_mm: float,
    angle_deg: float,
    obj_thickness_mm: float = 15.0,
    gripper_open_rad: float = GRIPPER_OPEN_RAD,
    enable_sag_compensation: bool = True,
    offset_major_mm: float = 0.0,
    offset_minor_mm: float = 0.0,
    verbose: bool = True
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[float]]:
    """
    固定爪着地モデルに基づく把持 IK 計算:
    offset_major_mm / offset_minor_mm により、物体の長辺・短辺方向の目標点を独立調整可能。
    """
    # 1. 物体中心のワールド座標 [m]
    p_obj = np.array([x_phys_mm / 1000.0, y_phys_mm / 1000.0])
    base_theta_rad = math.atan2(p_obj[1], p_obj[0])
    base_theta_deg = math.degrees(base_theta_rad)

    # 2. 物体の長辺・短辺の幾何単位ベクトル
    obj_rad = math.radians(angle_deg)
    u_axis = np.array([math.cos(obj_rad), math.sin(obj_rad)])      # 角度方向 (長軸)
    n_axis = np.array([-math.sin(obj_rad), math.cos(obj_rad)])     # 直交方向 (短軸)

    # 3. 手首ロール角 (ID 5) の決定 (長辺に爪を直交させる)
    rel_roll_1 = -(angle_deg - base_theta_deg)
    while rel_roll_1 > 90.0:
        rel_roll_1 -= 180.0
    while rel_roll_1 <= -90.0:
        rel_roll_1 += 180.0

    rel_roll_2 = rel_roll_1 + 180.0 if rel_roll_1 < 0 else rel_roll_1 - 180.0
    chosen_rel_roll = rel_roll_2 if abs(rel_roll_2) < abs(rel_roll_1) else rel_roll_1
    wrist_roll_rad = math.radians(chosen_rel_roll)

    # 4. 短手方向の固定爪オフセット (ベース位置の決定)
    half_thick_m = (obj_thickness_mm / 1000.0) / 2.0
    total_offset_minor_m = half_thick_m + LANDING_MARGIN_M + FIXED_JAW_OFFSET_M

    p_tcp_cand1 = p_obj + total_offset_minor_m * n_axis
    p_tcp_cand2 = p_obj - total_offset_minor_m * n_axis

    if np.linalg.norm(p_tcp_cand1) >= np.linalg.norm(p_tcp_cand2):
        p_tcp_base = p_tcp_cand1
        n_effective = n_axis
    else:
        p_tcp_base = p_tcp_cand2
        n_effective = -n_axis

    # 5. 長辺・短辺方向の独立オフセット付加
    shift_vector = (offset_major_mm / 1000.0) * u_axis + (offset_minor_mm / 1000.0) * n_effective
    p_tcp = p_tcp_base + shift_vector

    # 作業半径クランプ (最大 450mm まで許容)
    r_tcp = np.linalg.norm(p_tcp)
    MAX_REACH_M = 0.450
    if r_tcp > MAX_REACH_M:
        p_tcp = p_tcp * (MAX_REACH_M / r_tcp)
        r_tcp = MAX_REACH_M

    theta_deg = math.degrees(math.atan2(p_tcp[1], p_tcp[0]))
    corr_z_m = z_phys_mm / 1000.0

    sag_offset_mm = 0.0
    if enable_sag_compensation:
        sag_offset = calculate_sag_compensation(r_tcp, math.radians(theta_deg))
        effective_z = corr_z_m + sag_offset
        sag_offset_mm = sag_offset * 1000.0
    else:
        effective_z = corr_z_m

    if verbose:
        print("   [把持幾何計算 (可動域拡張モード)]")
        print(f"      物体中心       : ({p_obj[0]*1000:.1f}, {p_obj[1]*1000:.1f}) mm | 入力Z: {z_phys_mm:.1f} mm")
        print(f"      到達半径 r_tcp : {r_tcp*1000:.1f} mm (上限: {MAX_REACH_M*1000:.0f} mm) | たわみ補正: +{sag_offset_mm:.1f} mm (実効Z: {effective_z*1000:.1f} mm)")
        print(f"      目標 TCP       : ({p_tcp[0]*1000:.1f}, {p_tcp[1]*1000:.1f}) mm | 手首ロール: {chosen_rel_roll:.1f}°")

    # 7. ピッチ角探索ループ (浅い進入角 10°, 5° を含む)
    # --- 修正後: 自己干渉を防ぐ安定角を最優先 ---
    candidate_pitches = [30.0, 40.0, 20.0, 50.0, 10.0]
    failure_logs = []

    for pitch_deg in candidate_pitches:
        pitch_rad = math.radians(pitch_deg)
        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = effective_z + L_GRIPPER * math.sin(pitch_rad)

        ik_target, reason_target = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target,
            theta_deg=theta_deg,
            z_wrist=z_wrist_target,
            target_pitch_deg=pitch_deg,
            gripper_rad=gripper_open_rad,
            wrist_roll_rad=wrist_roll_rad,
            return_reason=True
        )

        if ik_target is None:
            failure_logs.append(f"ピッチ {pitch_deg:4.1f}° [把持点NG]: {reason_target}")
            continue

        warm_start_qpos = np.array([ik_target[sid] for sid in range(1, 7)])
        r_wrist_wp = r_wrist_target
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP

        ik_wp, reason_wp = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp,
            theta_deg=theta_deg,
            z_wrist=z_wrist_wp,
            target_pitch_deg=pitch_deg,
            gripper_rad=gripper_open_rad,
            wrist_roll_rad=wrist_roll_rad,
            init_qpos_custom=warm_start_qpos,
            return_reason=True
        )

        if ik_wp is None:
            failure_logs.append(f"ピッチ {pitch_deg:4.1f}° [上空点NG]: {reason_wp}")
            continue

        if verbose:
            raw_tgt = {sid: radian_to_raw(sid, ik_target[sid]) for sid in SERVO_IDS}
            print(f"   ✅ [把持 IK 探索成功] 選択ピッチ: {pitch_deg:.1f}° | Raw目標値: {raw_tgt}")

        return ik_target, ik_wp, float(pitch_deg)

    if verbose:
        print("   🔍 【把持 IK 探索失敗トレース】:")
        for log in failure_logs:
            print(f"      ✖ {log}")

    return None, None, None


# ==============================================================================
# 互換ラッパー (auto_calibrate_workspace.py 等の旧ツール用)
# ==============================================================================
def solve_ik_adaptive_approach(
    r_tcp: float,
    theta_deg: float,
    z_tcp: float,
    gripper_rad: float = GRIPPER_CLOSE_RAD,
    wrist_roll_rad: float = WRIST_ROLL_HORIZONTAL_RAD,
    enable_sag_compensation: bool = True
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[float]]:
    """
    極座標 (r_tcp [m], theta_deg [deg], z_tcp [m]) から進入可能なピッチ角を探索し、
    目標姿勢と上空待機姿勢のペアを算出する (旧 API 互換用)。
    """
    theta_rad = math.radians(theta_deg)
    
    if enable_sag_compensation:
        sag_offset = calculate_sag_compensation(r_tcp, theta_rad)
        effective_z = z_tcp + sag_offset
    else:
        effective_z = z_tcp

    candidate_pitches = [30.0, 40.0, 20.0, 50.0, 10.0]

    for pitch_deg in candidate_pitches:
        pitch_rad = math.radians(pitch_deg)
        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = effective_z + L_GRIPPER * math.sin(pitch_rad)

        ik_target, _ = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target,
            theta_deg=theta_deg,
            z_wrist=z_wrist_target,
            target_pitch_deg=pitch_deg,
            gripper_rad=gripper_rad,
            wrist_roll_rad=wrist_roll_rad,
            return_reason=True
        )

        if ik_target is None:
            continue

        warm_start_qpos = np.array([ik_target[sid] for sid in range(1, 7)])
        r_wrist_wp = r_wrist_target
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP

        ik_wp, _ = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp,
            theta_deg=theta_deg,
            z_wrist=z_wrist_wp,
            target_pitch_deg=pitch_deg,
            gripper_rad=gripper_rad,
            wrist_roll_rad=wrist_roll_rad,
            init_qpos_custom=warm_start_qpos,
            return_reason=True
        )

        if ik_wp is None:
            continue

        return ik_target, ik_wp, float(pitch_deg)

    return None, None, None


# ==============================================================================
# 6. 配置 (Place) 専用 IK 計算レイヤー (詳細ログ強化版)
# ==============================================================================
def solve_ik_tabletop_place(
    x_phys_mm: float,
    y_phys_mm: float,
    z_phys_mm: float = 12.0,
    place_angle_deg: float = 0.0,
    enable_sag_compensation: bool = True,
    verbose: bool = True
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[float]]:
    """
    指定された机上目標座標 (X, Y, Z [mm]) および配置方位角 (deg) から、
    物体を静かに置くための接地姿勢 (target) と上空進入姿勢 (waypoint) を算出する。
    """
    p_place = np.array([x_phys_mm / 1000.0, y_phys_mm / 1000.0])
    r_place = np.linalg.norm(p_place)
    base_theta_rad = math.atan2(p_place[1], p_place[0])
    base_theta_deg = math.degrees(base_theta_rad)

    # 手首ロール角 (ID 5) の計算
    rel_roll_1 = -(place_angle_deg - base_theta_deg)
    while rel_roll_1 > 90.0:
        rel_roll_1 -= 180.0
    while rel_roll_1 <= -90.0:
        rel_roll_1 += 180.0

    rel_roll_2 = rel_roll_1 + 180.0 if rel_roll_1 < 0 else rel_roll_1 - 180.0
    chosen_rel_roll = rel_roll_2 if abs(rel_roll_2) < abs(rel_roll_1) else rel_roll_1
    wrist_roll_rad = math.radians(chosen_rel_roll)

    corr_z_m = z_phys_mm / 1000.0
    sag_offset_mm = 0.0
    if enable_sag_compensation:
        sag_offset = calculate_sag_compensation(r_place, base_theta_rad)
        effective_z = corr_z_m + sag_offset
        sag_offset_mm = sag_offset * 1000.0
    else:
        effective_z = corr_z_m

    if verbose:
        print("   [配置幾何計算 (Place 計画)]")
        print(f"      目標座標       : ({x_phys_mm:.1f}, {y_phys_mm:.1f}) mm | 入力Z: {z_phys_mm:.1f} mm")
        print(f"      到達半径 r     : {r_place*1000:.1f} mm | 旋回角 θ: {base_theta_deg:.1f}°")
        print(f"      たわみ補正     : +{sag_offset_mm:.1f} mm (実効Z: {effective_z*1000:.1f} mm) | 手首ロール: {chosen_rel_roll:.1f}°")

    # --- 修正後: 自然に前方に突き出して置く角度を優先 ---
    candidate_pitches = [30.0, 40.0, 20.0, 50.0, 10.0]
    failure_logs = []

    for pitch_deg in candidate_pitches:
        pitch_rad = math.radians(pitch_deg)
        r_wrist_target = r_place - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = effective_z + L_GRIPPER * math.sin(pitch_rad)

        # 接地目標点 (爪は閉じたまま降下)
        ik_target, reason_target = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target,
            theta_deg=base_theta_deg,
            z_wrist=z_wrist_target,
            target_pitch_deg=pitch_deg,
            gripper_rad=GRIPPER_CLOSE_RAD,
            wrist_roll_rad=wrist_roll_rad,
            return_reason=True
        )
        if ik_target is None:
            failure_logs.append(f"ピッチ {pitch_deg:4.1f}° [配置接地NG]: {reason_target}")
            continue

        # 上空進入点
        warm_start_qpos = np.array([ik_target[sid] for sid in range(1, 7)])
        r_wrist_wp = r_wrist_target
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP

        ik_wp, reason_wp = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp,
            theta_deg=base_theta_deg,
            z_wrist=z_wrist_wp,
            target_pitch_deg=pitch_deg,
            gripper_rad=GRIPPER_CLOSE_RAD,
            wrist_roll_rad=wrist_roll_rad,
            init_qpos_custom=warm_start_qpos,
            return_reason=True
        )
        if ik_wp is None:
            failure_logs.append(f"ピッチ {pitch_deg:4.1f}° [配置上空NG]: {reason_wp}")
            continue

        if verbose:
            raw_tgt = {sid: radian_to_raw(sid, ik_target[sid]) for sid in SERVO_IDS}
            print(f"   ✅ [Place 計画成立] 選択ピッチ: {pitch_deg:.1f}° | Raw目標値: {raw_tgt}")

        return ik_target, ik_wp, float(pitch_deg)

    if verbose:
        print("   🔍 【Place IK 探索失敗トレース】:")
        for log in failure_logs:
            print(f"      ✖ {log}")

    return None, None, None