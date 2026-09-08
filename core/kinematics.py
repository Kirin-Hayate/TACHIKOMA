"""
==============================================================================
TACHIKOMA 角度変換・運動学・自重たわみ補正モジュール (core/kinematics.py)
==============================================================================
【モジュールの役割と構成】
本モジュールは、マニピュレータの幾何計算・座標変換・実機通信インターフェースを担います。

1. 単位変換レイヤー (Raw ⇄ Radian)
   - raw_to_radian: サーボ生値 (0〜4095) を MuJoCo 物理ジョイント角度 (rad) に変換
   - radian_to_raw: 物理ジョイント角度 (rad) をサーボ生値 (0〜4095) に変換
   - get_home_radians: ホーム姿勢の標準関節角度辞書を取得

2. リーダー追従計算レイヤー (テレオペ用)
   - calculate_target_rad: 操作側リーダー生値から追従目標ラジアンを計算 (ID5 相対追従対応)
   - calculate_target: 既存 Raw 値インターフェース互換ラッパー

3. 物理たわみ補正レイヤー (新規追加: 方法 2 採用)
   - calculate_sag_compensation: リーチ r と旋回角 θ から実機アームの沈み込み量 Δz を推計

4. 機構干渉・接地判定レイヤー
   - check_ground_penetration: アーム各リンクおよび爪先端の机面めり込みを検知

5. 逆運動学 (IK) ソルバーレイヤー
   - solve_ik_wrist_and_pitch: 手首目標座標と進入角からヤコビアン反復法で 6 軸関節角を算出
   - solve_ik_adaptive_approach: たわみ補正を適用し、進入角を自動調整しながら
     把持点 (target) と上空待機点 (waypoint) の安全な姿勢ペアを出力
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

# STS3215 サーボの分解能定数 (4096カウント / 360度 = 約 651.8986 counts/rad)
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

# 幾何定数 [m]
L_GRIPPER = 0.160            # 手首ピッチ回転軸から爪先端把持点までの実効長
DELTA_Z_WRIST_WP = 0.050     # 把持点に対する上空アプローチ待機点の垂直マージン (+50mm)
DEFAULT_APPROACH_DEG = 90.0  # デフォルト進入角 (90°: 真上からの垂直降下)
MIN_APPROACH_DEG = 30.0      # 許容する最小進入角 (30°: 斜めアプローチ限界)


# ==============================================================================
# 1. 単位変換レイヤー (Raw ⇄ Radian)
# ==============================================================================
def raw_to_radian(sid: int, target_raw: int) -> float:
    """
    フォロワーサーボの Raw カウント値 (0〜4095) を物理ラジアン角度へ変換する。
    """
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    diff = (target_raw - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def radian_to_raw(sid: int, angle_rad: float) -> int:
    """
    物理ラジアン角度をフォロワーサーボ送信用の Raw カウント値 (0〜4095) へ変換・クランプする。
    """
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    offset = SIM_OFFSETS.get(sid, 2048)
    raw_val = int(round(offset + (angle_rad / direction) * COUNTS_PER_RAD))
    return int(max(0, min(4095, raw_val)))


# 標準姿勢定数
WRIST_ROLL_HORIZONTAL_RAD = raw_to_radian(5, 2000)  # 手首ロール横挟み基準角 (約 -1.61 rad / -92.3°)
GRIPPER_OPEN_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_max", 2600))   # グリッパー全開
GRIPPER_CLOSE_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_min", 1400))  # グリッパー完全把持


def get_home_radians() -> Dict[int, float]:
    """
    システム規定の Home (待機) 姿勢における全関節の物理ラジアン辞書を取得する。
    """
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
    """
    リーダー操作側の Raw 値から、フォロワーの目標物理ラジアン角を算出する。
    - bounded (ID 1〜4, 6): 線形範囲マッピング
    - infinite (ID 5): 仮想アンバウンド累積による相対差分追従 (リミット衝突時の誤差蓄積を防止)
    """
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

        # 物理限界でクリップしない仮想連続カウント
        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)

        # 内部キャッシュをアンバウンド状態で更新
        follower_current_cache[sid] = new_target

        return raw_to_radian(sid, int(new_target))


def calculate_target(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> int:
    """
    既存スクリプトとの下位互換用ラッパー (目標ラジアンを算出した後に Raw 値で返す)。
    """
    target_rad = calculate_target_rad(sid, raw_leader, prev_raw_cache, follower_current_cache)
    return radian_to_raw(sid, target_rad)


# ==============================================================================
# 3. 物理たわみ補正レイヤー (2次元完全連成2次多項式モデル)
# ==============================================================================
def calculate_sag_compensation(r_m: float, theta_rad: float) -> float:
    """
    【モデルB: 2次元完全連成2次多項式による総たわみ補正量】
    Δz [mm] = 959.97 * r^2 - 211.49 * r - 3.49 * θ^2 + 45.54 * (r * θ^2) + 36.24
    
    引数:
        r_m: リーチ長 [m]
        theta_rad: 旋回角度 [rad]
    戻り値:
        補正持上げ量 Δz [m]
    """
    # 物理限界レンジ [0.15m, 0.40m] に安全クランプ
    r_c = max(0.15, min(0.40, r_m))
    th_sq = theta_rad ** 2

    # 連成多項式計算 (単位: mm)
    delta_z_mm = (
        959.9708 * (r_c ** 2)
        - 211.4899 * r_c
        - 3.4854 * th_sq
        + 45.5363 * (r_c * th_sq)
        + 36.2444
    )

    return delta_z_mm / 1000.0  # メートルに換算して返却


# ==============================================================================
# 4. 機構干渉・接地判定レイヤー
# ==============================================================================
def check_ground_penetration(min_z_threshold: float = 0.002) -> bool:
    """
    MuJoCo の順運動学 (FK) 結果をもとに、アーム各可動部リンクおよび爪先端が
    机上面（高さ min_z_threshold [m] 未満）にめり込んでいないかを検証する。
    めり込みを検知した場合は True を返す。
    """
    if _IK_MODEL is None or _IK_DATA is None:
        return False

    # アーム各リンク (body 2 以降) のめり込み判定
    for i in range(2, _IK_MODEL.nbody):
        if _IK_DATA.xpos[i][2] < min_z_threshold:
            return True

    # 爪先端 (TCP) のめり込み判定
    if _JAW_BODY_ID != -1:
        rot_mat = _IK_DATA.xmat[_JAW_BODY_ID].reshape(3, 3)
        tcp_z = (_IK_DATA.xpos[_JAW_BODY_ID] + rot_mat @ JAW_LOCAL_TCP_OFFSET)[2]
        if tcp_z < min_z_threshold:
            return True

    return False


# ==============================================================================
# 5. 逆運動学 (IK) ソルバーレイヤー
# ==============================================================================
def solve_ik_wrist_and_pitch(
    r_wrist: float, 
    theta_deg: float, 
    z_wrist: float, 
    target_pitch_deg: float,
    gripper_rad: float
) -> Optional[Dict[int, float]]:
    """
    手首ピッチ軸の位置 (r_wrist, theta_deg, z_wrist) と手先進入ピッチ角から、
    減衰付き最小二乗法 (DLS) ヤコビアン IK により ID 1〜6 の物理ラジアン角度を算出する。
    
    制約条件（収束誤差 20mm 未満、机面非干渉、実機サーボ可動域内）をすべて満たした場合のみ
    関節角度辞書 {sid: rad} を返し、満たさない場合は None を返す。
    """
    if _IK_MODEL is None or _IK_DATA is None or _WRIST_BODY_ID == -1:
        return None

    # 台座旋回角 (時計回りを正とする座標系から MuJoCo 座標系へ変換)
    theta_rad = math.radians(-theta_deg)
    target_wrist_pos = np.array([
        r_wrist * math.sin(theta_rad),
        -r_wrist * math.cos(theta_rad),
        z_wrist
    ])

    # 初期探索姿勢のセット
    init_qpos = np.array([theta_rad, 1.2, -1.8, -1.5, WRIST_ROLL_HORIZONTAL_RAD, 0.0])
    _IK_DATA.qpos[:6] = init_qpos
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    jacp = np.zeros((3, _IK_MODEL.nv))

    # --- ステップ 1: ID 1〜3 (肩・肘) で手首位置を目標点へ収束させる ---
    for _ in range(40):
        current_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
        error = target_wrist_pos - current_wrist_pos

        if np.linalg.norm(error) < 1.5e-3:
            break

        mujoco.mj_jacBody(_IK_MODEL, _IK_DATA, jacp, None, _WRIST_BODY_ID)
        J = jacp[:, :3]

        # 特異点付近での発散を防ぐ DLS (Levenberg-Marquardt) 法
        J_inv = J.T @ np.linalg.inv(J @ J.T + (0.015**2) * np.eye(3))
        delta_q = np.clip(J_inv @ error, -0.25, 0.25)

        _IK_DATA.qpos[:3] += delta_q
        mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    # 収束判定 (手首目標位置から 20mm 以上乖離している場合は解なし)
    final_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
    if np.linalg.norm(target_wrist_pos - final_wrist_pos) > 0.020:
        return None

    # --- ステップ 2: ID 4 (手首ピッチ) を指定進入角に幾何拘束 ---
    q2 = _IK_DATA.qpos[1]
    q3 = _IK_DATA.qpos[2]
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad
    _IK_DATA.qpos[3] = q4

    # --- ステップ 3: ID 5 (ロール横挟み) および ID 6 (爪開閉) の設定 ---
    _IK_DATA.qpos[4] = WRIST_ROLL_HORIZONTAL_RAD
    _IK_DATA.qpos[5] = gripper_rad
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    # 机面衝突の検証
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

    # 実機サーボの物理限界 (f_min, f_max) チェック
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
    init_pitch_deg: float = DEFAULT_APPROACH_DEG,
    enable_sag_compensation: bool = True
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[float]]:
    """
    指定された目標座標 [r, theta, z] に対して、自重たわみ補正を加算した上で、
    進入角を 90°(真上) から 30°(斜め) まで 5° 刻みで自動探索し、
    安全に到達可能な「目標姿勢 (target)」と「上空待機姿勢 (waypoint)」のペアを出力する。
    
    戻り値:
        (ik_target_rad, ik_waypoint_rad, adopted_pitch_deg)
        ※ 到達不能な場合は (None, None, None)
    """
    theta_rad = math.radians(theta_deg)

    # --------------------------------------------------------------------------
    # 物理たわみ補正の自動適用 (方法 2)
    # 机面沈み込み量 Δz(r, θ) を目標高さに上乗せして IK を解く
    # --------------------------------------------------------------------------
    if enable_sag_compensation:
        sag_offset = calculate_sag_compensation(r_tcp, theta_rad)
        effective_z = z_tcp + sag_offset
    else:
        effective_z = z_tcp

    # 進入角を急峻な角度 (90°) から緩やかな角度 (30°) へ適応探索
    for pitch_deg in np.arange(init_pitch_deg, MIN_APPROACH_DEG - 1.0, -5.0):
        pitch_rad = math.radians(pitch_deg)

        # 爪先端 (TCP) から手首ピッチ関節位置を逆算
        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = effective_z + L_GRIPPER * math.sin(pitch_rad)

        # 上空アプローチ経由点 (垂直方向に待機マージンを加算)
        r_wrist_wp = r_wrist_target
        z_wrist_wp = z_wrist_target + DELTA_Z_WRIST_WP

        ik_target = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_target,
            theta_deg=theta_deg,
            z_wrist=z_wrist_target,
            target_pitch_deg=pitch_deg,
            gripper_rad=gripper_rad
        )
        ik_wp = solve_ik_wrist_and_pitch(
            r_wrist=r_wrist_wp,
            theta_deg=theta_deg,
            z_wrist=z_wrist_wp,
            target_pitch_deg=pitch_deg,
            gripper_rad=gripper_rad
        )

        # 把持点と上空点の双方が安全に解けた進入角を採用
        if ik_target is not None and ik_wp is not None:
            return ik_target, ik_wp, float(pitch_deg)

    return None, None, None