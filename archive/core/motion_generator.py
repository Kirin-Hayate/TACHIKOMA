"""
==============================================================================
動的 IK 軌道ジェネレータ (core/motion_generator.py)
==============================================================================
【役割】
固定テンプレート CSV に依存せず、与えられた Pick 座標と Place 座標 [r, theta, z] から
その場で直接 MuJoCo ヤコビアン逆運動学 (IK) を解き、全軸の物理ラジアン軌道を生成します。

【動作シーケンス設計】
1. フェーズ a (進入アプローチ):
   Home ➔ Pick上空 (200 steps / 4.0秒: ゆったり大移動) ➔ 把持点降下 (50 steps / 1.0秒)
2. フェーズ b (把持・移載):
   把持 (24 steps / 0.48秒) ➔ 持ち上げ (36 steps / 0.72秒) ➔
   Place上空へ旋回・伸縮 (70 steps / 1.4秒) ➔ 接地降下 (36 steps / 0.72秒) ➔ 開放 (24 steps / 0.48秒)
3. フェーズ c (退避・帰還):
   Place上空退避 (50 steps / 1.0秒) ➔ Home復帰 (200 steps / 4.0秒: ゆったり大移動)
==============================================================================
"""

import os
import sys
import math
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import SERVO_IDS, SAMPLING_RATE_HZ
from core.kinematics import (
    get_home_radians,
    solve_ik_adaptive_approach,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)


class ParametricMotionGenerator:
    def __init__(self, template_csv=None):
        """
        初期化: 基準となる Home 姿勢のラジアン配列を読み込む
        (引数 template_csv は互換性のために残していますが、ファイルには依存しません)
        """
        self.home_rad = get_home_radians()

    def _interpolate_segment(self, start_rad: dict, end_rad: dict, steps: int) -> list:
        """
        開始姿勢から目標姿勢へコサイン S 字加減速で補間したフレームリストを生成。
        初速と終速がゼロになるため、モーターへの衝撃を防ぎます。
        """
        frames = []
        for step in range(1, steps + 1):
            ratio = step / steps
            s_ratio = (1.0 - math.cos(ratio * math.pi)) / 2.0
            frame = {
                sid: start_rad[sid] + s_ratio * (end_rad[sid] - start_rad[sid])
                for sid in SERVO_IDS
            }
            frames.append(frame)
        return frames

    def generate_from_coords(self, pick_coord: dict, place_coord: dict) -> Tuple[bool, List, str]:
        """
        指定された Pick 座標と Place 座標から、完全な Pick & Place 軌道を動的に計算して出力する。
        
        引数:
            pick_coord:  {"r": float, "theta_deg": float, "z": float}
            place_coord: {"r": float, "theta_deg": float, "z": float}
        戻り値:
            (成功成否: bool, フレームリスト: [(t_sec, {sid: rad})], ログメッセージ: str)
        """
        # --- 1. Pick 地点の逆運動学 (IK) を適応型進入角度で解く ---
        ik_pk_target, ik_pk_wp, pitch_pk = solve_ik_adaptive_approach(
            r_tcp=pick_coord["r"],
            theta_deg=pick_coord["theta_deg"],
            z_tcp=pick_coord["z"],
            gripper_rad=GRIPPER_OPEN_RAD
        )
        if ik_pk_target is None:
            return False, [], (
                f"Pick 座標 (r={pick_coord['r']*100:.1f}cm, θ={pick_coord['theta_deg']:+.1f}°) "
                f"への到達姿勢を算出できませんでした (可動域外または机面干渉)。"
            )

        # --- 2. Place 地点の逆運動学 (IK) を適応型進入角度で解く ---
        ik_pl_target, ik_pl_wp, pitch_pl = solve_ik_adaptive_approach(
            r_tcp=place_coord["r"],
            theta_deg=place_coord["theta_deg"],
            z_tcp=place_coord["z"],
            gripper_rad=GRIPPER_OPEN_RAD
        )
        if ik_pl_target is None:
            return False, [], (
                f"Place 座標 (r={place_coord['r']*100:.1f}cm, θ={place_coord['theta_deg']:+.1f}°) "
                f"への到達姿勢を算出できませんでした (可動域外または机面干渉)。"
            )

        # グリッパーを閉じた状態の姿勢辞書を作成
        ik_pk_target_c = dict(ik_pk_target)
        ik_pk_target_c[6] = GRIPPER_CLOSE_RAD
        ik_pk_wp_c = dict(ik_pk_wp)
        ik_pk_wp_c[6] = GRIPPER_CLOSE_RAD

        ik_pl_target_c = dict(ik_pl_target)
        ik_pl_target_c[6] = GRIPPER_CLOSE_RAD
        ik_pl_wp_c = dict(ik_pl_wp)
        ik_pl_wp_c[6] = GRIPPER_CLOSE_RAD

        raw_frames = []

        # ======================================================================
        # 【フェーズ a: アプローチ】 大移動のため落ち着いたステップ数
        # ======================================================================
        # 1. Home ➔ Pick 上空 (爪: 開) : 4.0秒 (200 steps)
        raw_frames.extend(self._interpolate_segment(self.home_rad, ik_pk_wp, steps=200))
        # 2. Pick 上空 ➔ 把持点降下 (爪: 開) : 1.0秒 (50 steps)
        raw_frames.extend(self._interpolate_segment(ik_pk_wp, ik_pk_target, steps=50))

        # ======================================================================
        # 【フェーズ b: 把持・移載】 良好な速度感をそのまま維持
        # ======================================================================
        # 3. 把持 (爪: 閉) : 0.48秒 (24 steps)
        raw_frames.extend(self._interpolate_segment(ik_pk_target, ik_pk_target_c, steps=24))
        # 4. 把持点 ➔ Pick 上空持ち上げ (爪: 閉維持) : 0.72秒 (36 steps)
        raw_frames.extend(self._interpolate_segment(ik_pk_target_c, ik_pk_wp_c, steps=36))
        # 5. Pick 上空 ➔ Place 上空へ旋回・伸縮 (爪: 閉維持) : 3秒 (150 steps)
        raw_frames.extend(self._interpolate_segment(ik_pk_wp_c, ik_pl_wp_c, steps=150))
        # 6. Place 上空 ➔ 接地降下 (爪: 閉維持) : 0.72秒 (36 steps)
        raw_frames.extend(self._interpolate_segment(ik_pl_wp_c, ik_pl_target_c, steps=36))
        # 7. 開放 (爪: 開) : 0.48秒 (24 steps)
        raw_frames.extend(self._interpolate_segment(ik_pl_target_c, ik_pl_target, steps=24))

        # ======================================================================
        # 【フェーズ c: 退避・帰還】 急激な戻りを防ぐため減速
        # ======================================================================
        # 8. 接地点 ➔ Place 上空退避 (爪: 開) : 1.0秒 (50 steps)
        raw_frames.extend(self._interpolate_segment(ik_pl_target, ik_pl_wp, steps=50))
        # 9. Place 上空 ➔ Home 復帰 (爪: 開) : 4.0秒 (200 steps)
        raw_frames.extend(self._interpolate_segment(ik_pl_wp, self.home_rad, steps=200))

        # 50Hz (0.02秒刻み) のタイムスタンプを付与
        dt = 1.0 / SAMPLING_RATE_HZ
        timed_frames = [(round(i * dt, 4), frame) for i, frame in enumerate(raw_frames)]

        log_msg = f"IK成功 (Pick進入角={pitch_pk:.0f}°, Place進入角={pitch_pl:.0f}°)"
        return True, timed_frames, log_msg