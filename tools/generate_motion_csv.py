"""
==============================================================================
Pick & Place 動作 CSV エクスポートツール (ラジアン統一版)
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
    """2姿勢間をコサイン S 字曲線で補間 (ラジアン配列)"""
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
    """全動作シーケンスをラジアンで生成"""
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

    # 1. Home ➔ Pick 上空 (爪: 開)
    full_frames.extend(interpolate_segment(home_rad, ik_pk_wp_open, steps=25))
    # 2. Pick 上空 ➔ 把持点降下 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pk_wp_open, ik_pk_target_open, steps=18))
    # 3. 把持 (爪: 閉)
    full_frames.extend(interpolate_segment(ik_pk_target_open, ik_pk_target_closed, steps=12))
    # 4. 把持点 ➔ Pick 上空持ち上げ (爪: 閉維持)
    full_frames.extend(interpolate_segment(ik_pk_target_closed, ik_pk_wp_closed, steps=18))
    # 5. Pick 上空 ➔ Place 上空へ旋回 (爪: 閉維持)
    full_frames.extend(interpolate_segment(ik_pk_wp_closed, ik_pl_wp_closed, steps=35))
    # 6. Place 上空 ➔ 接地降下 (爪: 閉維持)
    full_frames.extend(interpolate_segment(ik_pl_wp_closed, ik_pl_target_closed, steps=18))
    # 7. 開放 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pl_target_closed, ik_pl_target_open, steps=12))
    # 8. 接地点 ➔ Place 上空退避 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pl_target_open, ik_pl_wp_open, steps=18))
    # 9. Place 上空 ➔ Home 復帰 (爪: 開)
    full_frames.extend(interpolate_segment(ik_pl_wp_open, home_rad, steps=25))

    return full_frames


def save_to_csv(frames: list, output_path: str):
    """ラジアン CSV として保存 (小数点以下4桁)"""
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
    parser = argparse.ArgumentParser(description="Pick & Place 動作 CSV 自動生成 (ラジアン版)")
    parser.add_argument("--pick", nargs=3, type=float, metavar=("DIST_CM", "ANGLE_DEG", "Z_MM"),
                        help="Pick地点の極座標 (例: --pick 25 -30 15)")
    parser.add_argument("--place", nargs=3, type=float, metavar=("DIST_CM", "ANGLE_DEG", "Z_MM"),
                        help="Place地点の極座標 (例: --place 30 0 15)")
    parser.add_argument("--output", type=str, default=None,
                        help="出力先 CSV ファイルパス")
    args = parser.parse_args()

    print("=======================================================================")
    print(" 🚀 Pick & Place 軌道 CSV エクスポートツール (ラジアン統一版)")
    print("=======================================================================")

    if args.pick is not None:
        p_dist, p_deg, p_z = args.pick
    else:
        try:
            print("\n▼ Pick 地点 (A点) の座標を入力してください (Enter でデフォルト: 25cm -30° 15mm)")
            in_pk = input("  距離[cm] 角度[deg] 高さ[mm] > ").strip()
            if in_pk:
                p_dist, p_deg, p_z = map(float, in_pk.split())
            else:
                p_dist, p_deg, p_z = 25.0, -30.0, 15.0
        except Exception:
            p_dist, p_deg, p_z = 25.0, -30.0, 15.0

    if args.place is not None:
        pl_dist, pl_deg, pl_z = args.place
    else:
        try:
            print("\n▼ Place 地点 (B点) の座標を入力してください (Enter でデフォルト: 30cm 0° 15mm)")
            in_pl = input("  距離[cm] 角度[deg] 高さ[mm] > ").strip()
            if in_pl:
                pl_dist, pl_deg, pl_z = map(float, in_pl.split())
            else:
                pl_dist, pl_deg, pl_z = 30.0, 0.0, 15.0
        except Exception:
            pl_dist, pl_deg, pl_z = 30.0, 0.0, 15.0

    pick_r = p_dist / 100.0
    pick_th = p_deg
    pick_z = p_z / 1000.0

    place_r = pl_dist / 100.0
    place_th = pl_deg
    place_z = pl_z / 1000.0

    print(f"\n📍 Pick  地点 (A): 距離={p_dist:.1f}cm, 旋回角={pick_th:.1f}°, 高さ={p_z:.1f}mm")
    print(f"🎯 Place 地点 (B): 距離={pl_dist:.1f}cm, 旋回角={place_th:.1f}°, 高さ={pl_z:.1f}mm")

    ik_pk_tgt, ik_pk_wp, pitch_pk = solve_ik_adaptive_approach(
        r_tcp=pick_r, theta_deg=pick_th, z_tcp=pick_z, gripper_rad=GRIPPER_OPEN_RAD
    )
    if ik_pk_tgt is None:
        print("\n❌ エラー: Pick 地点に到達できませんでした。")
        sys.exit(1)

    ik_pl_tgt, ik_pl_wp, pitch_pl = solve_ik_adaptive_approach(
        r_tcp=place_r, theta_deg=place_th, z_tcp=place_z, gripper_rad=GRIPPER_OPEN_RAD
    )
    if ik_pl_tgt is None:
        print("\n❌ エラー: Place 地点に到達できませんでした。")
        sys.exit(1)

    print(f"✅ 逆運動学 (IK) 成立: Pick進入角={pitch_pk:.0f}°, Place進入角={pitch_pl:.0f}°")

    home_rad = get_home_radians()

    frames = build_full_trajectory(
        pick_ik_tuple=(ik_pk_tgt, ik_pk_wp),
        place_ik_tuple=(ik_pl_tgt, ik_pl_wp),
        home_rad=home_rad
    )

    if args.output:
        out_csv_path = args.output
        if not os.path.isabs(out_csv_path):
            out_csv_path = os.path.join(current_dir, out_csv_path)
    else:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rad_pick_and_place_{now_str}.csv"
        out_csv_path = os.path.join(current_dir, "motions", filename)

    save_to_csv(frames, out_csv_path)

    total_sec = len(frames) / SAMPLING_RATE_HZ 
    print(f"\n💾 ラジアン CSV を書き出しました:")
    print(f"   ➔ {os.path.relpath(out_csv_path, current_dir)}")
    print(f"   (総フレーム数: {len(frames)} / 再生時間: {total_sec:.2f} 秒)")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()