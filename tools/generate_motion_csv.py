"""
==============================================================================
Pick & Place 動作 CSV エクスポートツール (ラジアン統一版)(0.5倍速・高密度サンプリング版)
(tools/generate_motion_csv.py)
==============================================================================
【役割】
  - 指定された A点(Pick) から B点(Place) への動作軌道を物理ラジアン角で生成。
  - ヘッダー形式:
      timestamp_sec, q1, q2, q3, q4, q5, q6
  - ハードウェア固有の Raw 値ではなく、物理角度 [rad] のまま保存。
==============================================================================
"""

import sys
import os
import csv
import math
import argparse
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir and os.path.basename(current_dir) != "tachikoma":
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config.joint_config import SERVO_IDS, SAMPLING_RATE_HZ 
from core.kinematics import (
    get_home_radians,
    solve_ik_adaptive_approach,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)


def interpolate_segment(start_rad: dict, end_rad: dict, steps: int) -> list:
    """2姿勢間をコサイン S 字曲線で補間"""
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


def build_full_trajectory(pick_ik_tuple, place_ik_tuple, home_rad: dict) -> list:
    """
    全動作シーケンスを生成 (ステップ数を2倍にして50Hzでの滑らかな低速動作を実現)
    """
    ik_pk_target_open, ik_pk_wp_open = pick_ik_tuple
    ik_pl_target_open, ik_pl_wp_open = place_ik_tuple

    ik_pk_target_closed = ik_pk_target_open.copy()
    ik_pk_target_closed[6] = GRIPPER_CLOSE_RAD
    ik_pk_wp_closed = ik_pk_wp_open.copy()
    ik_pk_wp_closed[6] = GRIPPER_CLOSE_RAD

    ik_pl_target_closed = ik_pl_target_open.copy()
    ik_pl_target_closed[6] = GRIPPER_CLOSE_RAD
    ik_pl_wp_closed = ik_pl_wp_open.copy()
    ik_pl_wp_closed[6] = GRIPPER_CLOSE_RAD

    full_frames = []

    # 各ステップ数を従来の約2倍にスケール (50Hzを保ったまま0.5倍速化)
    # 1. Home ➔ Pick 上空 (爪: 開)
    full_frames.extend(interpolate_segment(home_rad, ik_pk_wp_open, steps=50))
    # 2. Pick 上空 ➔ 把持点降下 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pk_wp_open, ik_pk_target_open, steps=36))
    # 3. 把持 (爪: 閉)
    full_frames.extend(interpolate_segment(ik_pk_target_open, ik_pk_target_closed, steps=24))
    # 4. 把持点 ➔ Pick 上空持ち上げ (爪: 閉維持)
    full_frames.extend(interpolate_segment(ik_pk_target_closed, ik_pk_wp_closed, steps=36))
    # 5. Pick 上空 ➔ Place 上空へ旋回 (爪: 閉維持)
    full_frames.extend(interpolate_segment(ik_pk_wp_closed, ik_pl_wp_closed, steps=70))
    # 6. Place 上空 ➔ 接地降下 (爪: 閉維持)
    full_frames.extend(interpolate_segment(ik_pl_wp_closed, ik_pl_target_closed, steps=36))
    # 7. 開放 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pl_target_closed, ik_pl_target_open, steps=24))
    # 8. 接地点 ➔ Place 上空退避 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pl_target_open, ik_pl_wp_open, steps=36))
    # 9. Place 上空 ➔ Home 復帰 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pl_wp_open, home_rad, steps=50))

    return full_frames


def save_to_csv(frames: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dt = 1.0 / SAMPLING_RATE_HZ 

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_sec", "q1", "q2", "q3", "q4", "q5", "q6"])
        for idx, frame in enumerate(frames):
            t_sec = round(idx * dt, 4)
            row = [f"{t_sec:.4f}"]
            for sid in SERVO_IDS: 
                row.append(f"{frame[sid]:.5f}")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Pick & Place 動作 CSV 自動生成")
    parser.add_argument("--pick", nargs=3, type=float, metavar=("DIST_CM", "ANGLE_DEG", "Z_MM"),
                        help="Pick地点 (例: --pick 25 -30 25)")
    parser.add_argument("--place", nargs=3, type=float, metavar=("DIST_CM", "ANGLE_DEG", "Z_MM"),
                        help="Place地点 (例: --place 30 0 25)")
    parser.add_argument("--output", type=str, default=None,
                        help="出力先 CSV ファイルパス")
    args = parser.parse_args()

    print("=======================================================================")
    print(" 🚀 Pick & Place 軌道 CSV エクスポート (低速高密度・たわみ補正版)")
    print("=======================================================================")

    # たわみを考慮し、デフォルト高さを 15mm ➔ 25mm にオフセット
    if args.pick is not None:
        p_dist, p_deg, p_z = args.pick
    else:
        p_dist, p_deg, p_z = 25.0, -30.0, 25.0

    if args.place is not None:
        pl_dist, pl_deg, pl_z = args.place
    else:
        pl_dist, pl_deg, pl_z = 30.0, 0.0, 25.0

    pick_r, pick_th, pick_z = p_dist / 100.0, p_deg, p_z / 1000.0
    place_r, place_th, place_z = pl_dist / 100.0, pl_deg, pl_z / 1000.0

    print(f"📍 Pick  地点: 距離={p_dist:.1f}cm, 旋回角={pick_th:.1f}°, 高さ={p_z:.1f}mm")
    print(f"🎯 Place 地点: 距離={pl_dist:.1f}cm, 旋回角={place_th:.1f}°, 高さ={pl_z:.1f}mm")

    ik_pk_tgt, ik_pk_wp, pitch_pk = solve_ik_adaptive_approach(
        r_tcp=pick_r, theta_deg=pick_th, z_tcp=pick_z, gripper_rad=GRIPPER_OPEN_RAD
    )
    ik_pl_tgt, ik_pl_wp, pitch_pl = solve_ik_adaptive_approach(
        r_tcp=place_r, theta_deg=place_th, z_tcp=place_z, gripper_rad=GRIPPER_OPEN_RAD
    )

    if ik_pk_tgt is None or ik_pl_tgt is None:
        print("\n❌ エラー: 目標座標の IK 解が見つかりませんでした。")
        sys.exit(1)

    home_rad = get_home_radians()
    frames = build_full_trajectory((ik_pk_tgt, ik_pk_wp), (ik_pl_tgt, ik_pl_wp), home_rad)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv_path = args.output or os.path.join(current_dir, "motions", f"rad_smooth_{now_str}.csv")
    save_to_csv(frames, out_csv_path)

    total_sec = len(frames) / SAMPLING_RATE_HZ 
    print(f"\n💾 CSV を保存しました: {os.path.relpath(out_csv_path, current_dir)}")
    print(f"   (総フレーム数: {len(frames)} / 所要時間: {total_sec:.2f} 秒 @ 50Hz)")


if __name__ == "__main__":
    main()