"""
==============================================================================
実機フォロワーアーム モーション自動再生スクリプト (src/teleop_replay.py)
==============================================================================
【役割】
1. motions/ フォルダ内の記録済み CSV ファイルを読み込みます。
2. 動作開始前にフォロワーアームを安全な「Home位置」および「モーション開始位置」へ
   滑らかに移動（補間移動）させます。
3. 実機フォロワーへタイムスタンプに同期させて目標角度を送信し、動作を忠実に再現します。
==============================================================================
"""

import sys
import time
import csv
import os

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

# ==============================================================================
# 2. 再生シーケンス・安全動作設定
# ==============================================================================
# 実行したい CSV ファイル名と再生回数のリスト [ [ファイル名, 再生回数], ... ]
MOTION_SEQUENCE = [
    ["motion_20260822_221201.csv", 1]
]

MOTIONS_DIR = os.path.join(BASE_DIR, "motions")

# 安全動作の時間設定（急激な動きによる破損を防ぐための秒数）
HOME_RETURN_DURATION = 2.5         # 規定の Home 位置への復帰にかける秒数
MOTION_START_TRANSITION = 2.0      # モーション開始姿勢への位置合わせにかける秒数


# ==============================================================================
# 3. 再生用補助関数
# ==============================================================================
def smooth_move_follower(follower_driver, start_positions, target_positions, duration=2.0, steps=60):
    """
    開始姿勢 (start_positions) から 目標姿勢 (target_positions) へ
    指定秒数 (duration) かけて細かくステップ補間し、フォロワーを滑らかに移動させる
    """
    interval = duration / steps
    for step in range(1, steps + 1):
        ratio = step / steps
        for sid in SERVO_IDS:
            start_p = start_positions.get(sid, JOINT_CONFIG[sid]["init"])
            target_p = target_positions.get(sid, JOINT_CONFIG[sid]["init"])

            # 線形補間計算
            current_p = int(start_p + ratio * (target_p - start_p))
            follower_driver.write_position(sid, current_p)
        time.sleep(interval)


def load_motion_data(filepath):
    """CSV ファイルからタイムスタンプと各軸のリーダー生値を読み込む"""
    if not os.path.exists(filepath):
        print(f"❌ ファイルが存在しません: {filepath}")
        return None

    frames = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_sec"])
            positions = {sid: int(row[f"id_{sid}"]) for sid in SERVO_IDS}
            frames.append((t, positions))
    return frames


# ==============================================================================
# 4. 再生メイン処理
# ==============================================================================
def main():
    if not MOTION_SEQUENCE:
        print("⚠️ 再生シーケンス（MOTION_SEQUENCE）が空です。")
        return

    # 事前に全 CSV ファイルの存在確認と読み込み
    loaded_motions = []
    for filename, count in MOTION_SEQUENCE:
        filepath = os.path.join(MOTIONS_DIR, filename)
        frames = load_motion_data(filepath)
        if frames is None:
            print("🚨 読み込めないモーションファイルが存在するため、実行を中止します。")
            return
        loaded_motions.append({
            "filename": filename,
            "filepath": filepath,
            "repeat_count": count,
            "frames": frames
        })

    print(f"📋 登録されたシーケンス数: {len(loaded_motions)} 種類")
    for idx, item in enumerate(loaded_motions, start=1):
        print(f"  {idx}. {item['filename']} (再生回数: {item['repeat_count']}回 / {len(item['frames'])}フレーム)")

    follower = None

    try:
        # --- 通信ドライバの初期化 ---
        follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)

        # 規定の初期位置（Home）の定義
        home_positions = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

        # フォロワーの全サーボトルクを ON
        for sid in SERVO_IDS:
            follower.set_torque(sid, True)

        # 起動直後の物理姿勢を取得
        print("\n🔍 フォロワーの現在姿勢を確認中...")
        current_state = {}
        for sid in SERVO_IDS:
            pos = follower.read_position(sid)
            current_state[sid] = pos if pos is not None else home_positions[sid]

        # 最初の周回前に、現在姿勢 ➔ Home 姿勢へゆっくり復帰
        print(f"🏠 規定の初期位置（Home）へ復帰中 ({HOME_RETURN_DURATION}秒)...")
        smooth_move_follower(follower, current_state, home_positions, duration=HOME_RETURN_DURATION)
        current_state = dict(home_positions)
        time.sleep(0.3)

        # 複数モーションのシーケンス実行
        total_types = len(loaded_motions)
        for seq_idx, motion_item in enumerate(loaded_motions, start=1):
            filename = motion_item["filename"]
            repeat_count = motion_item["repeat_count"]
            frames = motion_item["frames"]

            # このモーションの第1フレーム（開始目標姿勢）を算出
            first_raw_positions = frames[0][1]
            first_targets = {}
            temp_prev = {}
            temp_curr = dict(home_positions)
            for sid in SERVO_IDS:
                raw_val = first_raw_positions[sid]
                first_targets[sid] = calculate_target(sid, raw_val, temp_prev, temp_curr)

            print(f"\n##################################################")
            print(f" 🎬 [モーション {seq_idx}/{total_types}] : {filename}")
            print(f"##################################################")

            for loop_idx in range(repeat_count):
                print(f"\n--- 🔄 {filename} [{loop_idx + 1} / {repeat_count} 周目] ---")

                # 直前の動作終了位置から Home へ復帰（現在位置が Home でない場合）
                if current_state != home_positions:
                    print(f"🏠 規定の初期位置へ復帰中 ({HOME_RETURN_DURATION}秒)...")
                    smooth_move_follower(follower, current_state, home_positions, duration=HOME_RETURN_DURATION)
                    current_state = dict(home_positions)
                    time.sleep(0.3)

                # Home ➔ モーション開始姿勢へアプローチ
                print(f"🎯 開始姿勢へアプローチ中 ({MOTION_START_TRANSITION}秒)...")
                smooth_move_follower(follower, current_state, first_targets, duration=MOTION_START_TRANSITION)
                current_state = dict(first_targets)
                time.sleep(0.4)

                # モーション再生実行
                print(f"▶️ 再生実行中...")
                prev_leader_cache = {}
                follower_current_cache = dict(home_positions)
                playback_start_time = time.time()

                for t_target, raw_positions in frames:
                    # CSV のタイムスタンプに合わせて実時間待機
                    while (time.time() - playback_start_time) < t_target:
                        time.sleep(0.001)

                    # 全関節の目標値を計算してフォロワーへ送信
                    for sid in SERVO_IDS:
                        raw_val = raw_positions[sid]
                        target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                        follower.write_position(sid, target_val)

                        prev_leader_cache[sid] = raw_val
                        follower_current_cache[sid] = target_val
                        current_state[sid] = target_val

                    sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {frames[-1][0]:6.2f}s")
                    sys.stdout.flush()
                print("")

                # 周回終了後、Home へ安全に戻す
                print(f"🏠 モーション終了 ➔ Homeへ復帰中 ({HOME_RETURN_DURATION}秒)...")
                smooth_move_follower(follower, current_state, home_positions, duration=HOME_RETURN_DURATION)
                current_state = dict(home_positions)
                time.sleep(0.3)

        print("\n✅ 全てのモーションシーケンスが安全に完了しました。")

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