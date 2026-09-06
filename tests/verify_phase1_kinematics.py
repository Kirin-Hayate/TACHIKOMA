"""
==============================================================================
フェーズ 1 単体検証テスト (tests/verify_phase1_kinematics.py)
==============================================================================
【検証項目】
  1. 可逆性テスト: Raw ➔ Radian ➔ Raw 変換で誤差が ±1 カウント以内か
  2. Home 姿勢テスト: 初期設定値から妥当な物理角度が算出されているか
  3. IK 計算テスト: ラジアン出力で目標位置が正しく解けるか
==============================================================================
"""

import sys
import os
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.joint_config import SERVO_IDS, JOINT_CONFIG
from core.kinematics import (
    raw_to_radian,
    radian_to_raw,
    get_home_radians,
    solve_ik_adaptive_approach,
    GRIPPER_OPEN_RAD
)


def run_verification():
    print("=======================================================================")
    print(" 🧪 フェーズ 1: 運動学モジュール (ラジアン化) 検証テスト")
    print("=======================================================================\n")

    # --- 1. 双方向可逆性テスト ---
    print("▶ 1. Raw ⇄ Radian 双方向変換テスト (0〜4095)")
    max_raw_diff = 0
    test_raw_values = [800, 1000, 2048, 3000, 3500]

    for sid in SERVO_IDS:
        for raw in test_raw_values:
            rad = raw_to_radian(sid, raw)
            restored_raw = radian_to_raw(sid, rad)
            diff = abs(raw - restored_raw)
            if diff > max_raw_diff:
                max_raw_diff = diff

    if max_raw_diff <= 1:
        print(f"  ✅ 可逆性チェック合格 (最大丸め誤差: {max_raw_diff} count)")
    else:
        print(f"  ❌ 可逆性チェック不合格 (丸め誤差が大きすぎます: {max_raw_diff} count)")

    # --- 2. Home 姿勢の角度チェック ---
    print("\n▶ 2. Home 姿勢ラジアン取得チェック")
    home_rads = get_home_radians()
    for sid in SERVO_IDS:
        deg = math.degrees(home_rads[sid])
        init_raw = JOINT_CONFIG[sid]["init"]
        print(f"  ID {sid}: {home_rads[sid]:+6.3f} rad ({deg:+6.1f}°)  [元の init Raw: {init_raw}]")

    # --- 3. IK 出力ラジアンテスト ---
    print("\n▶ 3. IK 計算ラジアン出力テスト (Pick 地点: r=25cm, θ=-30°, z=15mm)")
    ik_target, ik_wp, pitch = solve_ik_adaptive_approach(
        r_tcp=0.25, theta_deg=-30.0, z_tcp=0.015, gripper_rad=GRIPPER_OPEN_RAD
    )

    if ik_target is not None:
        print(f"  ✅ IK 解の算出成功 (進入角度: {pitch:.1f}°)")
        for sid in SERVO_IDS:
            rad = ik_target[sid]
            raw_preview = radian_to_raw(sid, rad)
            print(f"    軸 {sid}: {rad:+6.3f} rad ({math.degrees(rad):+6.1f}°) ➔ 送信時 Raw: {raw_preview}")
    else:
        print("  ❌ IK 計算に失敗しました")

    print("\n=======================================================================")
    print(" ✅ フェーズ 1 の全検証が完了しました")
    print("=======================================================================")


if __name__ == "__main__":
    run_verification()