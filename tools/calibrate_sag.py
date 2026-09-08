"""
==============================================================================
TACHIKOMA 机面タッチダウン・たわみ計測キャリブレーションツール
(tools/calibrate_sag.py)
==============================================================================
【役割】
距離 r [m] および 旋回角 theta [deg] の各測定グリッドに対して、
キーボード操作で高さを微調整し、爪先が机上面にジャスト接地した瞬間の「理論指示値 z」を記録します。
机上面の真値は z = 0 であるため、接地時の理論値 z がそのまま「自重たわみによる沈み込み量 Δz」となります。

【操作方法】
  - [↑] (Up)    : 高さ +1mm (上昇)
  - [↓] (Down)  : 高さ -1mm (降下)
  - [PageUp]    : 高さ +5mm (大まかに上昇)
  - [PageDown]  : 高さ -5mm (大まかに降下)
  - [Space/Enter]: 接地を確定して記録し、次の計測グリッドへ移動
  - [S]         : この地点をスキップ
  - [Q / Esc]   : 計測を中断して、そこまでの結果を保存して終了

【実行コマンド例】
  - 実機フォロワーと 3D シミュレータを両方動かして計測:
      python tools/calibrate_sag.py
==============================================================================
"""

import sys
import os
import time
import csv
import math
import msvcrt
from datetime import datetime

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
    radian_to_raw,
    raw_to_radian,
    GRIPPER_CLOSE_RAD
)

# ------------------------------------------------------------------------------
# 測定グリッドの定義 (必要に応じて追加・変更可能)
# ------------------------------------------------------------------------------
# 距離 r: 18cm, 21cm, 24cm, 27cm, 30cm
R_GRID_CM = [10.0,15.0, 20.0,25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

# 旋回角 θ: 左 (-45°), 正面 (0°), 右 (+45°)
THETA_GRID_DEG = [-45.0, 0.0, 45.0]

# 計測開始時の初期高さ [mm] (たわみを考慮して最初は浮かせた状態から降下)
INITIAL_TEST_Z_MM = 60.0

# 安全のための上限・下限 [mm]
Z_UPPER_LIMIT_MM = 180.0
Z_LOWER_LIMIT_MM = -10.0


def smooth_move_rad(follower, sim, target_rad, current_rad, duration=2.0, steps=50):
    """S字加減速補間で姿勢遷移"""
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


def get_key():
    """Windows コンソールでの特殊キー（矢印キーなど）の即時取得"""
    ch = msvcrt.getch()
    if ch in (b'\x00', b'\xe0'):
        ch2 = msvcrt.getch()
        if ch2 == b'H':
            return 'UP'
        elif ch2 == b'P':
            return 'DOWN'
        elif ch2 == b'I':
            return 'PGUP'
        elif ch2 == b'Q':
            return 'PGDN'
    elif ch in (b'\r', b'\n', b' '):
        return 'CONFIRM'
    elif ch in (b'q', b'Q', b'\x1b'):
        return 'QUIT'
    elif ch in (b's', b'S'):
        return 'SKIP'
    return None


def main():
    print("=======================================================================")
    print(" 📏 TACHIKOMA 机面タッチダウン・たわみ計測キャリブレーションツール")
    print("=======================================================================")
    print(f" 🎯 測定距離 r : {R_GRID_CM} cm")
    print(f" 🧭 測定角度 θ : {THETA_GRID_DEG} deg")
    print(f" 📊 総測定ポイント: {len(R_GRID_CM) * len(THETA_GRID_DEG)} 箇所")
    print("=======================================================================")

    # 1. ハードウェアおよびシミュレータ初期化
    follower = None
    try:
        follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
        print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
    except Exception as e:
        print(f"⚠️ 実機接続失敗: {e}")
        print("👉 シミュレーションモードのみで実行します。")

    sim = None
    try:
        sim = MujocoSimViewer()
        print("✅ 3Dシミュレータ初期化完了")
    except Exception as e:
        print(f"⚠️ 3Dシミュレータ初期化失敗: {e}")

    home_rad = get_home_radians()
    current_rad = dict(home_rad)

    # 起動時トルクON
    if follower is not None:
        for sid in SERVO_IDS:
            pos = follower.read_position(sid)
            current_rad[sid] = raw_to_radian(sid, pos) if pos is not None else home_rad[sid]
            follower.write_position(sid, radian_to_raw(sid, current_rad[sid]))
            follower.set_torque(sid, True)
        print("✅ トルクをONにしました。")

    # Home 姿勢へ移動
    smooth_move_rad(follower, sim, home_rad, current_rad, duration=2.5)

    # 計測結果の保存用リスト
    records = []
    total_points = len(R_GRID_CM) * len(THETA_GRID_DEG)
    point_idx = 0

    print("\n【操作ガイド】")
    print("  [↑ / ↓]      : 高さ ±1mm 微調整")
    print("  [PgUp / PgDn]: 高さ ±5mm 粗調整")
    print("  [Space / Enter]: 接地を確定して次へ")
    print("  [S]          : スキップ")
    print("  [Q / Esc]    : 計測終了")

    try:
        for theta in THETA_GRID_DEG:
            for r_cm in R_GRID_CM:
                point_idx += 1
                r_m = r_cm / 100.0
                curr_z_mm = INITIAL_TEST_Z_MM

                print(f"\n-----------------------------------------------------------------------")
                print(f" 📍 [{point_idx}/{total_points}] グリッド移動中: r = {r_cm:.1f} cm, θ = {theta:+.1f}°")

                # 初期位置（浮かせた高さ）への IK 計算
                tgt_rad, _, _ = solve_ik_adaptive_approach(
                    r_tcp=r_m, theta_deg=theta, z_tcp=(curr_z_mm / 1000.0), gripper_rad=GRIPPER_CLOSE_RAD
                )
                if tgt_rad is None:
                    print(f"⚠️ [IK解なし] r={r_cm:.1f}cm, θ={theta:+.1f}° は機構限界外のためスキップします。")
                    continue

                # その地点上空へ移動
                smooth_move_rad(follower, sim, tgt_rad, current_rad, duration=1.5)

                # 微調整ループ
                confirmed = False
                while True:
                    sys.stdout.write(f"\r 👉 現在の指示高さ z = {curr_z_mm:5.1f} mm  (沈み込み相当量 Δz = {curr_z_mm:+5.1f} mm)   ")
                    sys.stdout.flush()

                    key = get_key()
                    if key is None:
                        time.sleep(0.01)
                        continue

                    if key == 'UP':
                        curr_z_mm = min(Z_UPPER_LIMIT_MM, curr_z_mm + 1.0)
                    elif key == 'DOWN':
                        curr_z_mm = max(Z_LOWER_LIMIT_MM, curr_z_mm - 1.0)
                    elif key == 'PGUP':
                        curr_z_mm = min(Z_UPPER_LIMIT_MM, curr_z_mm + 5.0)
                    elif key == 'PGDN':
                        curr_z_mm = max(Z_LOWER_LIMIT_MM, curr_z_mm - 5.0)
                    elif key == 'CONFIRM':
                        confirmed = True
                        break
                    elif key == 'SKIP':
                        print("\n⏩ このポイントをスキップしました。")
                        break
                    elif key == 'QUIT':
                        print("\n🛑 計測を中断しました。")
                        raise KeyboardInterrupt

                    # 新しい高さで即座に姿勢更新
                    new_tgt, _, _ = solve_ik_adaptive_approach(
                        r_tcp=r_m, theta_deg=theta, z_tcp=(curr_z_mm / 1000.0), gripper_rad=GRIPPER_CLOSE_RAD
                    )
                    if new_tgt is not None:
                        if follower is not None:
                            for sid in SERVO_IDS:
                                follower.write_position(sid, radian_to_raw(sid, new_tgt[sid]))
                        if sim is not None:
                            sim.update_joints_rad(new_tgt)
                        current_rad.update(new_tgt)

                if confirmed:
                    # 机上面を z=0 としたとき、接地したときの理論指示値 curr_z_mm が「たわみ量 (沈み込み量)」
                    sag_amount_mm = curr_z_mm
                    print(f"\n✅ 確定記録: r={r_cm:.1f}cm, θ={theta:+.1f}° ➔ たわみ量 Δz = {sag_amount_mm:.1f} mm")
                    records.append({
                        "r_cm": r_cm,
                        "r_m": r_m,
                        "theta_deg": theta,
                        "sag_delta_z_mm": sag_amount_mm,
                        "sag_delta_z_m": sag_amount_mm / 1000.0
                    })

                # 次の地点へ行く前に少し上に逃げる（机を擦らないための安全退避）
                retract_rad, _, _ = solve_ik_adaptive_approach(
                    r_tcp=r_m, theta_deg=theta, z_tcp=((curr_z_mm + 15.0) / 1000.0), gripper_rad=GRIPPER_CLOSE_RAD
                )
                if retract_rad is not None:
                    smooth_move_rad(follower, sim, retract_rad, current_rad, duration=0.6, steps=20)

    except KeyboardInterrupt:
        pass
    finally:
        # 計測終了後 Home 復帰
        print("\n🏠 Home 姿勢へ復帰中...")
        try:
            smooth_move_rad(follower, sim, home_rad, current_rad, duration=2.0)
        except Exception:
            pass

        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
            print("✅ フォロワーのトルクをOFFにしました。")

        # データの CSV 保存
        if records:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_dir = os.path.join(BASE_DIR, "data")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"sag_calibration_{now_str}.csv")

            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["r_cm", "r_m", "theta_deg", "sag_delta_z_mm", "sag_delta_z_m"])
                writer.writeheader()
                writer.writerows(records)

            print("\n=======================================================================")
            print(f"💾 計測結果 ({len(records)} 件) を保存しました:")
            print(f"   ➔ {os.path.relpath(csv_path, BASE_DIR)}")
            print("=======================================================================\n")
            print("【計測データ一覧】")
            for r in records:
                print(f"  r = {r['r_cm']:4.1f} cm, θ = {r['theta_deg']:+5.1f}° ➔ Δz = {r['sag_delta_z_mm']:+5.1f} mm")
        else:
            print("\n⚠️ 記録されたデータはありませんでした。")


if __name__ == "__main__":
    main()