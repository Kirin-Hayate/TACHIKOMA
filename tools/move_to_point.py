"""
==============================================================================
TACHIKOMA 任意座標到達テストツール (tools/move_to_point.py)
==============================================================================
【役割】
指定した円筒座標 [r, theta, z] に対して、tachikoma_agent_2.py と全く同じ
制御パイプライン（たわみ補正、適応進入角 IK、コサイン S 字軌道）を用いて
実機フォロワーおよび 3D シミュレータを移動させます。
机面の実座標グリッド（方眼紙やマーカー位置）との照合確認に使用します。

【操作方法】
  対話プロンプトに以下を入力:
    r [cm], theta [deg], z [mm]
    (例: 「25, 0, 5」 ➔ r=25cm, θ=0°, 机面から 5mm)
    (例: 「30, 45, 10」➔ r=30cm, θ=+45°, 机面から 10mm)
    (例: 「h」または「home」➔ ホーム姿勢へ復帰)
    (例: 「q」➔ 終了)

【実行コマンド例】
  - 実機フォロワー ＋ 3D シミュレータ:
      python tools/move_to_point.py --arm --sim
  - 3D シミュレータのみ（安全確認）:
      python tools/move_to_point.py --sim
  - 実機フォロワーのみ:
      python tools/move_to_point.py --arm
==============================================================================
"""

import sys
import os
import time
import math
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import (
    FOLLOWER_PORT,
    BAUDRATE,
    SERVO_IDS,
    JOINT_CONFIG
)
from core.sts3215 import STS3215Driver
from core.sim_viewer import MujocoSimViewer
from core.kinematics import (
    get_home_radians,
    solve_ik_adaptive_approach,
    calculate_sag_compensation,
    raw_to_radian,
    radian_to_raw,
    GRIPPER_CLOSE_RAD
)

# 移動時間設定 [秒]
MOVE_DURATION = 2.0


def smooth_move_rad(follower, sim, target_rad, current_rad, duration=2.0, steps=60):
    """コサイン S 字加減速による安全補間移動"""
    interval = duration / steps
    for step in range(1, steps + 1):
        ratio = (1.0 - math.cos(step / steps * math.pi)) / 2.0
        step_rad = {
            sid: current_rad[sid] + ratio * (target_rad[sid] - current_rad[sid])
            for sid in SERVO_IDS
        }
        if follower is not None:
            for sid in SERVO_IDS:
                follower.write_position(sid, radian_to_raw(sid, step_rad[sid]))
        if sim is not None:
            sim.update_joints_rad(step_rad)
        time.sleep(interval)
    current_rad.update(target_rad)


def parse_arguments():
    parser = argparse.ArgumentParser(description="TACHIKOMA 任意座標到達テストツール")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー送信を有効化")
    parser.add_argument("--sim", action="store_true", help="3D シミュレータ表示を有効化")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("==================================================")
    print(" 🎯 TACHIKOMA 任意座標到達テストツール")
    print(f" 🤖 実機フォロワー: {'有効 (--arm)' if args.arm else 'OFF'}")
    print(f" 🖥️ 3D画面描画  : {'有効 (--sim)' if args.sim else 'OFF'}")
    print("==================================================")

    # 実機ドライバ初期化
    follower = None
    if args.arm:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"❌ 実機フォロワー接続失敗: {e}")
            return

    # 3D シミュレータ初期化
    sim = None
    if args.sim:
        try:
            sim = MujocoSimViewer()
            print("✅ 3D シミュレータ初期化完了")
        except Exception as e:
            print(f"⚠️ 3D シミュレータ初期化失敗: {e}")
            sim = None

    home_rad = get_home_radians()
    current_rad = dict(home_rad)

    # 起動時の現在姿勢読み取りとトルク ON
    if follower is not None:
        for sid in SERVO_IDS:
            pos = follower.read_position(sid)
            current_rad[sid] = raw_to_radian(sid, pos) if pos is not None else home_rad[sid]
            follower.write_position(sid, radian_to_raw(sid, current_rad[sid]))
            follower.set_torque(sid, True)
        print("✅ 全サーボのトルクを ON にしました。")

    # 3D 画面の初期同期
    if sim is not None:
        sim.update_joints_rad(current_rad)

    # 初期位置として Home へゆっくり移動
    print(f"🏠 Home 姿勢へ移動中 ({MOVE_DURATION}秒)...")
    smooth_move_rad(follower, sim, home_rad, current_rad, duration=MOVE_DURATION)

    print("\n【入力フォーマット】")
    print("  r [cm], theta [deg], z [mm]  (カンマ区切り、またはスペース区切り)")
    print("  例: 25, 0, 5    ➔ 距離 25cm, 正面(0°), 机面上 5mm")
    print("  例: 30, 45, 10  ➔ 距離 30cm, 右45°, 机面上 10mm")
    print("  コマンド: 'h' = Home 復帰 / 'q' = 終了\n")

    def run_cli_loop():
        nonlocal current_rad
        while True:
            if sim is not None and not sim.is_running():
                break

            cmd = input("📍 移動先を入力 > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ['q', 'quit', 'exit']:
                break
            if cmd.lower() in ['h', 'home']:
                print("🏠 Home 姿勢へ移動します。")
                smooth_move_rad(follower, sim, home_rad, current_rad, duration=MOVE_DURATION)
                continue

            # 入力文字列のパース
            parts = cmd.replace(',', ' ').split()
            if len(parts) != 3:
                print("⚠️ 入力形式が正しくありません。「r, theta, z」の 3 つの数値を入力してください。")
                continue

            try:
                r_cm = float(parts[0])
                th_deg = float(parts[1])
                z_mm = float(parts[2])
            except ValueError:
                print("⚠️ 数値を正しく解釈できませんでした。")
                continue

            r_m = r_cm / 100.0
            z_m = z_mm / 1000.0
            th_rad = math.radians(th_deg)

            # たわみ補正量の事前計算（確認表示用）
            sag_offset_m = calculate_sag_compensation(r_m, th_rad)
            sag_offset_mm = sag_offset_m * 1000.0

            print(f"\n--- 🚀 目標座標へ移動開始 ---")
            print(f"  指示座標: r = {r_cm:.1f} cm, θ = {th_deg:+.1f}°, z = {z_mm:.1f} mm")
            print(f"  たわみ補正: +{sag_offset_mm:.1f} mm (実効目標 z = {z_mm + sag_offset_mm:.1f} mm)")

            # tachikoma_agent_2 と全く同じ IK ソルバー呼び出し（たわみ補正適用）
            tgt_rad, wp_rad, pitch_deg = solve_ik_adaptive_approach(
                r_tcp=r_m,
                theta_deg=th_deg,
                z_tcp=z_m,
                gripper_rad=GRIPPER_CLOSE_RAD,
                enable_sag_compensation=True
            )

            if tgt_rad is None:
                print("❌ [IK 算出不能] 指定された座標は可動範囲外または干渉リスクのため到達できません。")
                continue

            print(f"  進入角度: {pitch_deg:.0f}°")

            # 1. 上空待機点 (waypoint) へ S 字移動
            smooth_move_rad(follower, sim, wp_rad, current_rad, duration=1.5, steps=45)
            # 2. 目的の接地/把持高さ (target) へゆっくり降下
            smooth_move_rad(follower, sim, tgt_rad, current_rad, duration=0.8, steps=30)

            print("✅ 目標地点に到達しました。\n")

    try:
        if sim is not None:
            with sim.launch():
                run_cli_loop()
        else:
            run_cli_loop()

    except KeyboardInterrupt:
        print("\n\n🛑 中断しました。")
    finally:
        print("🏠 Home 姿勢へ復帰中...")
        try:
            smooth_move_rad(follower, sim, home_rad, current_rad, duration=1.5)
        except Exception:
            pass

        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
            print("✅ フォロワーのトルクを OFF にし、ポートを閉じました。")


if __name__ == "__main__":
    main()