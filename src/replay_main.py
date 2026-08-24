"""
==============================================================================
TACHIKOMA 統合モーション自動再生スクリプト (src/replay_main.py)
==============================================================================
【役割】
指定された複数の CSV モーションファイルを、それぞれの指定回数ずつ順番に連続再生します。
--arm（実機） / --sim（3D画面）オプションにより、画面・実機・両方を自由に選択可能です。

【実行コマンド例】
  1. 単一ファイルを1回再生（ファイル名のみ）
      python src/replay_main.py motions/A.csv --sim

  2. 単一ファイルを3回再生（コロンで回数指定 "ファイル:回数"）
      python src/replay_main.py motions/A.csv:3 --arm --sim

  3. 複数ファイルを連続シーケンス実行 (Aを1回 ➔ Bを3回 ➔ Cを2回)
      python src/replay_main.py motions/A.csv:1 motions/B.csv:3 motions/C.csv:2 --arm --sim
==============================================================================
"""

import sys
import time
import csv
import os
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
from core.kinematics import calculate_target
from core.sim_viewer import MujocoSimViewer

# ==============================================================================
# 2. 安全動作パラメータ設定
# ==============================================================================
HOME_RETURN_DURATION = 2.5         # 規定の Home 位置への復帰にかける秒数[cite: 5]
MOTION_START_TRANSITION = 2.0      # モーション開始姿勢への位置合わせにかける秒数[cite: 5]


# ==============================================================================
# 3. 実行時コマンドライン引数の解析 (argparse)
# ==============================================================================
def parse_arguments():
    """コマンドライン引数を定義・取得する関数"""
    parser = argparse.ArgumentParser(
        description="TACHIKOMA 複数モーション連続再生 統合スクリプト"
    )

    # 再生対象の CSV ファイル（複数指定可能、"ファイルパス:回数" の形式に対応）
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="再生するモーションファイル（例: motions/A.csv motions/B.csv:3 motions/C.csv:2）"
    )

    # --arm オプション (実機フォロワーへの送信有効化)
    parser.add_argument(
        "--arm",
        action="store_true",
        help="実機フォロワーでの自動再生を有効化します（デフォルト: OFF）"
    )

    # --sim オプション (MuJoCo 3D画面の表示有効化)
    parser.add_argument(
        "--sim",
        action="store_true",
        help="MuJoCo 3Dシミュレーション描画を有効化します（デフォルト: OFF）"
    )

    return parser.parse_args()


# ==============================================================================
# 4. 補助関数
# ==============================================================================
def parse_sequence_items(raw_items):
    """
    コマンドラインの文字列リストから「ファイルパス」と「再生回数」のリストを作成する
    例: ["motions/A.csv", "motions/B.csv:3"] -> [("motions/A.csv", 1), ("motions/B.csv", 3)]
    """
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
    """CSV ファイルからタイムスタンプと各軸のリーダー生値を読み込む"""
    if not os.path.exists(filepath):
        print(f"❌ CSVファイルが見つかりません: {filepath}")
        return None

    frames = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_sec"])
            positions = {sid: int(row[f"id_{sid}"]) for sid in SERVO_IDS}
            frames.append((t, positions))
    return frames


def smooth_move(follower_driver, sim_viewer, start_positions, target_positions, duration=2.0, steps=60):
    """
    開始姿勢から目標姿勢へ指定秒数かけて補間移動する
    （実機フォロワーおよび 3D シミュレータ画面の両方を同期更新）
    """
    interval = duration / steps
    for step in range(1, steps + 1):
        ratio = step / steps
        current_step_positions = {}

        for sid in SERVO_IDS:
            start_p = start_positions.get(sid, JOINT_CONFIG[sid]["init"])
            target_p = target_positions.get(sid, JOINT_CONFIG[sid]["init"])
            current_p = int(start_p + ratio * (target_p - start_p))
            current_step_positions[sid] = current_p

            if follower_driver is not None:
                follower_driver.write_position(sid, current_p)

        if sim_viewer is not None:
            sim_viewer.update_joints(current_step_positions)

        time.sleep(interval)


# ==============================================================================
# 5. メイン再生処理
# ==============================================================================
def main():
    args = parse_arguments()

    enable_follower = args.arm
    enable_sim = args.sim
    raw_file_items = args.files

    # どちらも指定されていない場合は、安全のため画面シミュレーションをデフォルト起動
    if not enable_follower and not enable_sim:
        print("💡 --arm も --sim も指定されていないため、デフォルトで [--sim] (3D画面表示) を有効化します。")
        enable_sim = True

    # 動作モード文字列の作成
    mode_list = []
    if enable_follower:
        mode_list.append("🤖 実機フォロワー再生")
    if enable_sim:
        mode_list.append("🖥️ 3D画面描画")

    # ファイルパスと再生回数の解析・事前ロード
    sequence_items = parse_sequence_items(raw_file_items)
    loaded_motions = []

    for filepath, repeat_count in sequence_items:
        frames = load_motion_data(filepath)
        if frames is None or len(frames) == 0:
            print(f"🚨 有効なデータが取得できないファイルがあるため、実行を中止します: {filepath}")
            return
        loaded_motions.append({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "repeat_count": repeat_count,
            "frames": frames
        })

    print("==================================================")
    print(f" 🎬 TACHIKOMA 統合モーションシーケンス再生")
    print(f" ⚙️ 有効機能: {' + '.join(mode_list)}")
    print(f" 📋 再生シーケンス (全 {len(loaded_motions)} モーション):")
    for idx, item in enumerate(loaded_motions, start=1):
        print(f"   {idx}. {item['filename']} (再生回数: {item['repeat_count']}回 / {len(item['frames'])}フレーム)")
    print("==================================================")

    follower = None
    sim = None

    # --- 1. 実機フォロワーの初期化 ---
    if enable_follower:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            for sid in SERVO_IDS:
                follower.set_torque(sid, True)
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"❌ 実機フォロワー接続失敗: {e}")
            return
    else:
        print("💡 --arm が指定されていないため、実機送信は行いません。")

    # --- 2. MuJoCo 3Dシミュレータビューアの初期化 ---
    if enable_sim:
        try:
            sim = MujocoSimViewer()
            print("✅ 3Dシミュレータ初期化完了")
        except Exception as e:
            print(f"⚠️ 3Dシミュレータ初期化失敗: {e}")
            sim = None

    try:
        home_positions = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

        # 起動直後の姿勢を取得
        current_state = {}
        if follower is not None:
            print("\n🔍 フォロワーの現在姿勢を確認中...")
            for sid in SERVO_IDS:
                pos = follower.read_position(sid)
                current_state[sid] = pos if pos is not None else home_positions[sid]
        else:
            current_state = dict(home_positions)

        # 最初の周回前に、現在位置 ➔ Home への復帰
        print(f"🏠 規定の初期位置（Home）へ復帰中 ({HOME_RETURN_DURATION}秒)...")
        smooth_move(follower, sim, current_state, home_positions, duration=HOME_RETURN_DURATION)
        current_state = dict(home_positions)
        time.sleep(0.3)

        # 再生ループ本体
        def run_playback():
            nonlocal current_state
            total_motions = len(loaded_motions)

            for seq_idx, motion_item in enumerate(loaded_motions, start=1):
                filename = motion_item["filename"]
                repeat_count = motion_item["repeat_count"]
                frames = motion_item["frames"]
                total_duration = frames[-1][0]

                # このモーションの開始姿勢（第1フレーム）を算出
                first_raw_positions = frames[0][1]
                first_targets = {}
                temp_prev = {}
                temp_curr = dict(home_positions)
                for sid in SERVO_IDS:
                    raw_val = first_raw_positions[sid]
                    first_targets[sid] = calculate_target(sid, raw_val, temp_prev, temp_curr)

                print(f"\n##################################################")
                print(f" 🎬 [モーション {seq_idx}/{total_motions}] : {filename}")
                print(f"##################################################")

                for loop_idx in range(repeat_count):
                    print(f"\n--- 🔄 {filename} [{loop_idx + 1} / {repeat_count} 周目] ---")

                    # 直前の位置が Home でない場合は Home に戻す
                    if current_state != home_positions:
                        print(f"🏠 規定の初期位置へ復帰中 ({HOME_RETURN_DURATION}秒)...")
                        smooth_move(follower, sim, current_state, home_positions, duration=HOME_RETURN_DURATION)
                        current_state = dict(home_positions)
                        time.sleep(0.3)

                    # Home ➔ モーション開始姿勢へアプローチ
                    print(f"🎯 開始姿勢へアプローチ中 ({MOTION_START_TRANSITION}秒)...")
                    smooth_move(follower, sim, current_state, first_targets, duration=MOTION_START_TRANSITION)
                    current_state = dict(first_targets)
                    time.sleep(0.4)

                    print("▶️ 再生実行中...")
                    prev_leader_cache = {}
                    follower_current_cache = dict(home_positions)
                    playback_start_time = time.time()
                    frame_idx = 0

                    while frame_idx < len(frames):
                        if sim is not None and not sim.is_running():
                            return

                        # 3D画面での一時停止
                        if sim is not None and sim.paused:
                            time.sleep(0.02)
                            playback_start_time = time.time() - frames[frame_idx][0]
                            continue

                        # 3D画面での最初からリプレイ
                        if sim is not None and sim.reset_requested:
                            sim.reset_requested = False
                            frame_idx = 0
                            playback_start_time = time.time()
                            continue

                        t_target, raw_positions = frames[frame_idx]

                        # CSV のタイムスタンプに合わせて実時間同期
                        elapsed = time.time() - playback_start_time
                        if elapsed < t_target:
                            time.sleep(0.001)
                            continue

                        # 全関節の目標値を計算
                        target_positions = {}
                        for sid in SERVO_IDS:
                            raw_val = raw_positions[sid]
                            target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                            prev_leader_cache[sid] = raw_val
                            follower_current_cache[sid] = target_val
                            target_positions[sid] = target_val
                            current_state[sid] = target_val

                            if follower is not None:
                                follower.write_position(sid, target_val)

                        if sim is not None:
                            sim.update_joints(target_positions)

                        sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {total_duration:6.2f}s [Frame {frame_idx + 1}/{len(frames)}]  ")
                        sys.stdout.flush()

                        frame_idx += 1

                    print("")

                    # 周回終了後、Home へ安全に戻す
                    print(f"🏠 モーション終了 ➔ Homeへ復帰中 ({HOME_RETURN_DURATION}秒)...")
                    smooth_move(follower, sim, current_state, home_positions, duration=HOME_RETURN_DURATION)
                    current_state = dict(home_positions)
                    time.sleep(0.3)

            print("\n✅ 全てのモーションシーケンス再生が完了しました。")

            # 3Dビューア単体再生時、終了後もウィンドウを開いたまま待機
            if sim is not None and follower is None:
                print("💡 [R]キーでシーケンス全体を最初からリプレイ可能です。（ウィンドウを閉じると終了）")
                while sim.is_running():
                    if sim.reset_requested:
                        sim.reset_requested = False
                        run_playback()
                        break
                    time.sleep(0.05)

        # 3D画面が有効なら launch コンテキスト内で実行
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