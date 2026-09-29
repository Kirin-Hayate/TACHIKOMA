"""
==============================================================================
TACHIKOMA 角度変換・運動学・自重たわみ補正・非対称把持モジュール (core/kinematics.py)
==============================================================================
【モジュールの役割と構成】
本モジュールは、マニピュレータの幾何計算・座標変換・把持計画・実機通信インターフェースを担います。

1. 単位変換レイヤー (Raw ⇄ Radian)
   - raw_to_radian: サーボ生値 (0〜4095) を MuJoCo 物理ジョイント角度 (rad) に変換
   - radian_to_raw: 物理ジョイント角度 (rad) をサーボ生値 (0〜4095) に変換
   - get_home_radians: ホーム姿勢の標準関節角度辞書を取得

2. リーダー追従計算レイヤー (テレオペ用)
   - calculate_target_rad: 操作側リーダー生値から追従目標ラジアンを計算 (ID5 相対追従対応)
   - calculate_target: 既存 Raw 値インターフェース互換ラッパー

3. 物理たわみ補正レイヤー (2次元完全連成2次多項式モデル)
   - calculate_sag_compensation: リーチ r と旋回角 θ から実機アームの沈み込み量 Δz を推計

4. 機構干渉・接地判定レイヤー
   - check_ground_penetration: アーム各リンクおよび爪先端の机面めり込みを検知

5. 逆運動学 (IK) ソルバーレイヤー (ヨー角アライメント ＆ 非対称把持対応)
   - solve_ik_wrist_and_pitch: 手首目標座標・進入ピッチ角・手首ロール角からヤコビアン反復法で 6 軸関節角を算出
   - solve_ik_tabletop_grasp: 机上直交座標 (X, Y, Z) と物体の傾き角 (angle_deg) を受け取り、
     手首ロール角のアライメントおよび非対称爪（固定爪干渉回避）の偏心補正を自動計算した上で、
     最適な把持姿勢 (grasp) と上空待機姿勢 (waypoint) の安全な姿勢ペアを出力
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
L_GRIPPER = 0.160            # 手首ピッチ回転軸から爪先端把持点までの実効長[cite: 12]
DELTA_Z_WRIST_WP = 0.050     # 把持点に対する上空アプローチ待機点の垂直マージン (+50mm)[cite: 12]
DEFAULT_APPROACH_DEG = 90.0  # デフォルト進入角 (90°: 真上からの垂直降下)[cite: 12]
MIN_APPROACH_DEG = 5.0      # 許容する最小進入角 (5°: 斜めアプローチ限界)[cite: 12]

# SO-ARM100 非対称爪補正定数 [m]
# 固定爪を物体外縁から逃がすためのローカル側方偏心マージン (実機の構造に合わせて微調整)
GRIPPER_ASYM_OFFSET_M = 0.012  # 約 12mm 


# ==============================================================================
# 1. 単位変換レイヤー (Raw ⇄ Radian)
# ==============================================================================
def raw_to_radian(sid: int, target_raw: int) -> float:
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    diff = (target_raw - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def radian_to_raw(sid: int, angle_rad: float) -> int:
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    offset = SIM_OFFSETS.get(sid, 2048)
    raw_val = int(round(offset + (angle_rad / direction) * COUNTS_PER_RAD))
    return int(max(0, min(4095, raw_val)))


# 標準姿勢定数
WRIST_ROLL_HORIZONTAL_RAD = raw_to_radian(5, 2000)  # 約 -1.61 rad[cite: 12]
GRIPPER_OPEN_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_max", 2600))   # グリッパー全開[cite: 12]
GRIPPER_CLOSE_RAD = raw_to_radian(6, JOINT_CONFIG[6].get("f_min", 1400))  # グリッパー完全把持[cite: 12]


def get_home_radians() -> Dict[int, float]:
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

        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)
        follower_current_cache[sid] = new_target

        return raw_to_radian(sid, int(new_target))


def calculate_target(sid: int, raw_leader: int, prev_raw_cache: dict, follower_current_cache: dict) -> int:
    target_rad = calculate_target_rad(sid, raw_leader, prev_raw_cache, follower_current_cache)
    return radian_to_raw(sid, target_rad)


# ==============================================================================
# 3. 物理たわみ補正レイヤー (2次元完全連成2次多項式モデル)
# ==============================================================================
def calculate_sag_compensation(r_m: float, theta_rad: float) -> float:
    r_c = max(0.15, min(0.40, r_m))
    th_sq = theta_rad ** 2

    delta_z_mm = (
        959.9708 * (r_c ** 2)
        - 211.4899 * r_c
        - 3.4854 * th_sq
        + 45.5363 * (r_c * th_sq)
        + 36.2444
    )
    return delta_z_mm / 1000.0


# ==============================================================================
# 4. 機構干渉・接地判定レイヤー
# ==============================================================================
def check_ground_penetration(min_z_threshold: float = -0.005) -> Tuple[bool, str]:
    """
    机面衝突判定:
    戻り値: (衝突有無 bool, 衝突箇所の詳細メッセージ str)
    """
    if _IK_MODEL is None or _IK_DATA is None:
        return False, "OK"

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
    return_reason=True の場合、失敗理由の文字列を併せて返す。
    """
    if _IK_MODEL is None or _IK_DATA is None or _WRIST_BODY_ID == -1:
        res = (None, "MuJoCo IK モデル未初期化") if return_reason else None
        return res if return_reason else None

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

    # 1. 位置収束判定
    final_wrist_pos = _IK_DATA.xpos[_WRIST_BODY_ID]
    dist_err = np.linalg.norm(target_wrist_pos - final_wrist_pos)
    if dist_err > 0.020:
        msg = f"手首位置の収束失敗 (誤差: {dist_err*1000:.1f}mm > 20mm)"
        return (None, msg) if return_reason else None

    # 2. 手首ピッチ・ロール姿勢拘束
    q2 = _IK_DATA.qpos[1]
    q3 = _IK_DATA.qpos[2]
    delta_pitch_rad = math.radians(90.0 - target_pitch_deg)
    q4 = -(math.radians(90.0) + q2 + q3 - (math.pi * 0.9)) - delta_pitch_rad
    _IK_DATA.qpos[3] = q4
    _IK_DATA.qpos[4] = wrist_roll_rad
    _IK_DATA.qpos[5] = gripper_rad
    mujoco.mj_forward(_IK_MODEL, _IK_DATA)

    # 3. 机面干渉チェック
    #is_penetrated, pen_reason = check_ground_penetration(min_z_threshold=-0.005)
    #if is_penetrated:
    #    msg = f"机面衝突検知: {pen_reason}"
    #    return (None, msg) if return_reason else None

    rad_targets = {
        1: float(_IK_DATA.qpos[0]),
        2: float(_IK_DATA.qpos[1]),
        3: float(_IK_DATA.qpos[2]),
        4: float(_IK_DATA.qpos[3]),
        5: float(wrist_roll_rad),
        6: float(gripper_rad),
    }

    # 4. サーボ可動域リミットチェック (ID 1〜5)
    for sid in [1, 2, 3, 4, 5]:
        cfg = JOINT_CONFIG.get(sid)
        if cfg and cfg["type"] == "bounded":
            raw_val = radian_to_raw(sid, rad_targets[sid])
            if not (cfg["f_min"] <= raw_val <= cfg["f_max"]):
                msg = f"サーボ ID{sid} 可動域外 (Raw={raw_val}, 許容: {cfg['f_min']}〜{cfg['f_max']})"
                return (None, msg) if return_reason else None

    return (rad_targets, "OK") if return_reason else rad_targets

def solve_ik_tabletop_grasp(
x_phys_mm: float,
    y_phys_mm: float,
    z_phys_mm: float,
    angle_deg: float,
    obj_thickness_mm: float = 15.0,
    gripper_open_rad: float = GRIPPER_OPEN_RAD,
    enable_sag_compensation: bool = True,
    verbose: bool = True  # 👉 デバッグログフラグを追加
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], Optional[float]]:
    """
    机上の物理直交座標 (X_mm, Y_mm, Z_mm) およびカメラ検出角度 (angle_deg) から、
    非対称爪の干渉回避オフセットを算出して最適な把持姿勢と上空待機姿勢のペアを出力する。

    引数:
        x_phys_mm: ロボット基準前方奥行き (mm)
        y_phys_mm: ロボット基準横方向変位 (mm, 右側が正)
        z_phys_mm: 把持高さ (机面からの高さ mm, 通常は物体厚みの半分)
        angle_deg: OpenCV で検出した物体の傾き角 (-90°〜+90°)
        obj_thickness_mm: 挟み込む厚み (mm)
    戻り値:
        (ik_grasp_rad, ik_waypoint_rad, adopted_pitch_deg)
    """
    # 1. 極座標系変換
    base_theta_deg = math.degrees(math.atan2(y_phys_mm, x_phys_mm))

    # 2. 手首ロール角 (ID 5) アライメント
    rel_roll_deg_1 = angle_deg - base_theta_deg
    while rel_roll_deg_1 > 90.0:
        rel_roll_deg_1 -= 180.0
    while rel_roll_deg_1 <= -90.0:
        rel_roll_deg_1 += 180.0

    rel_roll_deg_2 = rel_roll_deg_1 + 180.0 if rel_roll_deg_1 < 0 else rel_roll_deg_1 - 180.0

    cand_rad_1 = WRIST_ROLL_HORIZONTAL_RAD + math.radians(rel_roll_deg_1)
    cand_rad_2 = WRIST_ROLL_HORIZONTAL_RAD + math.radians(rel_roll_deg_2)

    # 候補1・候補2 の選定
    if abs(cand_rad_1) <= abs(cand_rad_2):
        rel_roll_deg = rel_roll_deg_1
        wrist_roll_rad = cand_rad_1
    else:
        rel_roll_deg = rel_roll_deg_2
        wrist_roll_rad = cand_rad_2

    # 3. 非対称爪補正
    asym_shift_m = min(0.015, GRIPPER_ASYM_OFFSET_M + (min(25.0, obj_thickness_mm) / 2000.0))
    global_yaw_rad = math.radians(base_theta_deg + rel_roll_deg)
    shift_dx_m = -asym_shift_m * math.sin(global_yaw_rad)
    shift_dy_m = asym_shift_m * math.cos(global_yaw_rad)

    corr_x_m = (x_phys_mm / 1000.0) + shift_dx_m
    corr_y_m = (y_phys_mm / 1000.0) + shift_dy_m
    corr_z_m = z_phys_mm / 1000.0

    r_check = math.hypot(corr_x_m, corr_y_m)
    if r_check > 0.360:
        scale = 0.360 / r_check
        corr_x_m *= scale
        corr_y_m *= scale

    r_tcp = math.hypot(corr_x_m, corr_y_m)
    theta_deg = math.degrees(math.atan2(corr_y_m, corr_x_m))

    if verbose:
        print(f"   [幾何解析] 補正目標: r={r_tcp*1000:.1f}mm, θ={theta_deg:.1f}° | 手首ロール目標: {math.degrees(wrist_roll_rad):.1f}° (Raw≈{radian_to_raw(5, wrist_roll_rad)})")

    # 4. たわみ補正
    if enable_sag_compensation:
        sag_offset = calculate_sag_compensation(r_tcp, math.radians(theta_deg))
        effective_z = corr_z_m + sag_offset
    else:
        effective_z = corr_z_m

    # 5. ピッチ角探索ループ
    candidate_pitches = [80.0, 60.0, 40.0, 20.0]
    failure_logs = []

    for pitch_deg in candidate_pitches:
        pitch_rad = math.radians(pitch_deg)
        r_wrist_target = r_tcp - L_GRIPPER * math.cos(pitch_rad)
        z_wrist_target = effective_z + L_GRIPPER * math.sin(pitch_rad)

        # 把持点 (Target)
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

        # 上空点 (Waypoint)
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

        return ik_target, ik_wp, float(pitch_deg)

    if verbose:
        print("   🔍 【IK 探索失敗トレース】:")
        for log in failure_logs:
            print(f"      ✖ {log}")

    return None, None, None