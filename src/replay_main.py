"""
==============================================================================
TACHIKOMA 統合モーション自動再生スクリプト (ラジアン統一・速度制御版)
(src/replay_main.py)
==============================================================================
【役割】
指定された複数の CSV モーションファイルを、それぞれの指定回数ずつ順番に連続再生します。
内部演算・軌道制御を「物理関節ラジアン」で統一し、機体の個体差や再組み立ての影響を受けない
高精度な再生を実現しています。
--arm（実機） / --sim（3D画面） / --speed（再生速度倍率）により安全な検証が可能です。

【安全機能】
1. 再生速度スケーリング (--speed)
   実機テスト時の安全のため、任意の倍率（例: 0.5倍速）で全シーケンスを低速実行可能。
2. 起動直後の安全初期化（急発進防止）
   トルクOFFの状態で実機の静止角度を読み取ってラジアン換算し、そこからトルクをONにして
   初速0のコサインS字加減速でゆっくり規定のHome位置へ遷移します。
3. 物理ラジアン空間でのコサイン補間（S字カーブ加減速）
   動作中・終了時・Rキー復帰時のすべての遷移で加速度を連続にし、急激な負荷を防ぎます。
4. 実機通信境界での一括 Raw 値変換
   実機サーボへの送信直前にのみ radian_to_raw() を通して書き込みます。

【実行コマンド例】
  - 0.5倍速（ゆっくり）でシミュレーション再生
      python src/replay_main.py motions/rad_test.csv --sim --speed 0.5
  - 0.5倍速で実機フォロワー ＋ 画面表示で再生
      python src/replay_main.py motions/rad_test.csv --arm --sim --speed 0.5
==============================================================================
"""

import sys
import time
import csv
import os
import math
import argparse

# ==============================================================================
# 1. 共通モジュール・設定のインポート
# ==============================================================================
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
from core.kinematics import (
    raw_to_radian,
    radian_to_raw,
    get_home_radians
)
from core.sim_viewer import MujocoSimViewer

# ==============================================================================
# 2. 安全動作パラメータ基準設定 (秒単位, speed=1.0 基準)
# ==============================================================================
BASE_STARTUP_HOME_DURATION = 3.0       # 起動直後のHome復帰基準時間
BASE_HOME_RETURN_DURATION = 2.5        # 通常Home復帰基準時間
BASE_MOTION_START_TRANSITION = 2.0     # 開始姿勢アプローチ基準時間
BASE_RESET_REWIND_DURATION = 2.0       # Rキー巻き戻し基準時間


# ==============================================================================
# 3. コマンドライン引数の解析
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TACHIKOMA 複数モーション連続再生 統合スクリプト (速度調整・ラジアン版)"
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="再生するモーションファイル（例: motions/A.csv motions/B.csv:3）"
    )

    parser.add_argument(
        "--arm",
        action="store_true",
        help="実機フォロワーでの自動再生を有効化します（デフォルト: OFF）"
    )

    parser.add_argument(
        "--sim",
        action="store_true",
        help="MuJoCo 3Dシミュレーション描画を有効化します（デフォルト: OFF）"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="再生速度倍率 (例: 0.5 で半分、2.0 で倍速。デフォルト: 1.0)"
    )

    return parser.parse_args()


# ==============================================================================
# 4. 補助関数 (データ読み込み・補間)
# ==============================================================================
def parse_sequence_items(raw_items):
    sequence = []
    for item in raw_items:
        if ":" in item:
            path, count_str = item.rsplit(":", 1)
            try:
                count = int(count_str)
            except ValueError:
                path = item
                count = 1
        else:
            path = item
            count = 1
        sequence.append((path, max(1, count)))
    return sequence


def load_motion_data(filepath):
    if not os.path.exists(filepath):
        print(f"❌ CSVファイルが見つかりません: {filepath}")
        return None

    frames = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        is_radian = "q1" in headers

        for row in reader:
            t = float(row["timestamp_sec"])
            positions = {}
            for sid in SERVO_IDS:
                if is_radian:
                    positions[sid] = float(row[f"q{sid}"])
                else:
                    raw_val = int(row[f"id_{sid}"])
                    positions[sid] = raw_to_radian(sid, raw_val)
            frames.append((t, positions))
    return frames


def smooth_move(follower_driver, sim_viewer, target_positions, fallback_state=None, duration=2.5, steps=75):
    start_positions = {}
    for sid in SERVO_IDS:
        if follower_driver is not None:
            p_raw = follower_driver.read_position(sid)
            if p_raw is not None:
                start_positions[sid] = raw_to_radian(sid, p_raw)
            elif fallback_state is not None and sid in fallback_state:
                start_positions[sid] = fallback_state[sid]
            else:
                start_positions[sid] = raw_to_radian(sid, JOINT_CONFIG[sid]["init"])
        else:
            if fallback_state is not None and sid in fallback_state:
                start_positions[sid] = fallback_state[sid]
            else:
                start_positions[sid] = raw_to_radian(sid, JOINT_CONFIG[sid]["init"])

    interval = duration / steps

    for step in range(1, steps + 1):
        t = step / steps
        ratio = (1.0 - math.cos(t * math.pi)) / 2.0

        current_step_positions = {}
        for sid in SERVO_IDS:
            start_rad = start_positions[sid]
            target_rad = target_positions.get(sid, raw_to_radian(sid, JOINT_CONFIG[sid]["init"]))
            current_rad = start_rad + ratio * (target_rad - start_rad)
            current_step_positions[sid] = current_rad

            if follower_driver is not None:
                raw_val = radian_to_raw(sid, current_rad)
                follower_driver.write_position(sid, raw_val)

        if sim_viewer is not None:
            sim_viewer.update_joints_rad(current_step_positions)

        time.sleep(interval)


# ==============================================================================
# 5. メイン再生処理
# ==============================================================================
def main():
    args = parse_arguments()

    enable_follower = args.arm
    enable_sim = args.sim
    speed = max(0.05, float(args.speed))
    raw_file_items = args.files

    if not enable_follower and not enable_sim:
        print("💡 --arm も --sim も指定されていないため、デフォルトで [--sim] (3D画面表示) を有効化します。")
        enable_sim = True

    # 速度倍率に応じたアプローチ時間の計算
    startup_home_dur = BASE_STARTUP_HOME_DURATION / speed
    home_return_dur = BASE_HOME_RETURN_DURATION / speed
    motion_start_dur = BASE_MOTION_START_TRANSITION / speed
    reset_rewind_dur = BASE_RESET_REWIND_DURATION / speed

    mode_list = []
    if enable_follower:
        mode_list.append("🤖 実機フォロワー再生")
    if enable_sim:
        mode_list.append("🖥️ 3D画面描画")

    sequence_items = parse_sequence_items(raw_file_items)
    loaded_motions = []

    for filepath, repeat_count in sequence_items:
        frames = load_motion_data(filepath)
        if frames is None or len(frames) == 0:
            print(f"🚨 有効なデータが取得できないため中止: {filepath}")
            return
        loaded_motions.append({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "repeat_count": repeat_count,
            "frames": frames
        })

    print("==================================================")
    print(f" 🎬 TACHIKOMA 統合モーションシーケンス再生 (ラジアン統一版)")
    print(f" ⚙️ 有効機能: {' + '.join(mode_list)}")
    print(f" ⏩ 再生速度: {speed:.2f}倍速 (所要時間: 約 {1.0/speed:.2f}倍)")
    print(f" 📋 再生シーケンス (全 {len(loaded_motions)} モーション):")
    for idx, item in enumerate(loaded_motions, start=1):
        orig_dur = item["frames"][-1][0]
        actual_dur = orig_dur / speed
        print(f"   {idx}. {item['filename']} ({actual_dur:.1f}s / {item['repeat_count']}回 / {len(item['frames'])}フレーム)")
    print("==================================================")

    follower = None
    sim = None

    if enable_follower:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            print(f"✅ 実機フォロワーポート接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"❌ 実機フォロワー接続失敗: {e}")
            return
    else:
        print("💡 --arm が指定されていないため、実機送信は行いません。")

    if enable_sim:
        try:
            sim = MujocoSimViewer()
            print("✅ 3Dシミュレータ初期化完了")
        except Exception as e:
            print(f"⚠️ 3Dシミュレータ初期化失敗: {e}")
            sim = None

    try:
        home_positions = get_home_radians()
        current_state = {}

        if follower is not None:
            print("\n🔍 起動時の実機静止姿勢をスキャン中（トルクOFF安全状態）...")
            for sid in SERVO_IDS:
                p_raw = follower.read_position(sid)
                current_state[sid] = raw_to_radian(sid, p_raw) if p_raw is not None else home_positions[sid]
            
            for sid in SERVO_IDS:
                follower.write_position(sid, radian_to_raw(sid, current_state[sid]))
                follower.set_torque(sid, True)
            print("✅ トルクをONにしました（姿勢維持中）。")
        else:
            current_state = dict(home_positions)

        if sim is not None:
            sim.update_joints_rad(current_state)

        print(f"🏠 初期位置（Home）へ滑らかに復帰中 ({startup_home_dur:.1f}秒)...")
        smooth_move(follower, sim, home_positions, fallback_state=current_state, duration=startup_home_dur)
        current_state = dict(home_positions)
        time.sleep(0.3)

        def run_playback():
            nonlocal current_state
            total_motions = len(loaded_motions)

            for seq_idx, motion_item in enumerate(loaded_motions, start=1):
                filename = motion_item["filename"]
                repeat_count = motion_item["repeat_count"]
                frames = motion_item["frames"]
                total_duration = frames[-1][0]

                first_targets = frames[0][1]

                print(f"\n##################################################")
                print(f" 🎬 [モーション {seq_idx}/{total_motions}] : {filename}")
                print(f"##################################################")

                for loop_idx in range(repeat_count):
                    print(f"\n--- 🔄 {filename} [{loop_idx + 1} / {repeat_count} 周目] ---")

                    if current_state != home_positions:
                        print(f"🏠 規定の初期位置へ復帰中 ({home_return_dur:.1f}秒)...")
                        smooth_move(follower, sim, home_positions, fallback_state=current_state, duration=home_return_dur)
                        current_state = dict(home_positions)
                        time.sleep(0.3)

                    print(f"🎯 開始姿勢へアプローチ中 ({motion_start_dur:.1f}秒)...")
                    smooth_move(follower, sim, first_targets, fallback_state=current_state, duration=motion_start_dur)
                    current_state = dict(first_targets)
                    time.sleep(0.4)

                    print(f"▶️ 再生実行中... ({speed:.2f}倍速)")
                    playback_start_time = time.time()
                    frame_idx = 0

                    while frame_idx < len(frames):
                        if sim is not None and not sim.is_running():
                            return

                        if sim is not None and sim.paused:
                            t_target, _ = frames[frame_idx]
                            sys.stdout.write(
                                f" \r          {t_target:6.2f}s / {total_duration:6.2f}s [Frame{frame_idx + 1:4d} / {len(frames):4d}]  "
                            )
                            sys.stdout.flush()
                            time.sleep(0.02)
                            playback_start_time = time.time() - (frames[frame_idx][0] / speed)
                            continue

                        if sim is not None and sim.reset_requested:
                            sim.reset_requested = False
                            print("\n🔄 [Rキー検知] 安全に巻き戻しています...")
                            smooth_move(follower, sim, home_positions, fallback_state=current_state, duration=reset_rewind_dur)
                            smooth_move(follower, sim, first_targets, fallback_state=home_positions, duration=motion_start_dur)
                            current_state = dict(first_targets)
                            frame_idx = 0
                            playback_start_time = time.time()
                            print("▶️ モーションを最初から再開します。\n")
                            continue

                        t_target, rad_positions = frames[frame_idx]

                        # 速度倍率を反映した経過時間判定
                        elapsed = (time.time() - playback_start_time) * speed
                        if elapsed < t_target:
                            time.sleep(0.001)
                            continue

                        for sid in SERVO_IDS:
                            current_state[sid] = rad_positions[sid]
                            if follower is not None:
                                raw_val = radian_to_raw(sid, rad_positions[sid])
                                follower.write_position(sid, raw_val)

                        if sim is not None:
                            sim.update_joints_rad(rad_positions)

                        sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {total_duration:6.2f}s [Frame {frame_idx + 1}/{len(frames)}]  ")
                        sys.stdout.flush()

                        frame_idx += 1

                    print("")
                    print(f"🏠 モーション終了 ➔ Homeへ復帰中 ({home_return_dur:.1f}秒)...")
                    smooth_move(follower, sim, home_positions, fallback_state=current_state, duration=home_return_dur)
                    current_state = dict(home_positions)
                    time.sleep(0.3)

            print("\n✅ 全てのモーションシーケンス再生が完了しました。")

            if sim is not None and follower is None:
                print("💡 [R]キーでシーケンス全体を最初からリプレイ可能です。（ウィンドウを閉じると終了）")
                while sim.is_running():
                    if sim.reset_requested:
                        sim.reset_requested = False
                        run_playback()
                        break
                    time.sleep(0.05)

        if sim is not None:
            with sim.launch():
                run_playback()
        else:
            run_playback()

    except KeyboardInterrupt:
        print("\n\n🛑 再生を中断しました。")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
    finally:
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
            print("✅ フォロワーのトルクをOFFにし、ポートをクローズしました。")


if __name__ == "__main__":
    main()