"""
==============================================================================
MuJoCo 3Dシミュレーション モーション再生スクリプト (src/mujoco_replay.py)
==============================================================================
【役割】
1. motions/ フォルダ内の記録済み CSV ファイルを読み込みます。
2. core/sim_viewer.py (MuJoCo 3D描画) および core/kinematics.py (目標値計算) を使用して、
   実機フォロワーが動くはずの挙動を画面上で完全再現します。
3. Spaceキー（一時停止/再開）、Rキー（最初から）、Lキー（ループ再生）に対応しています。
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

from config.joint_config import SERVO_IDS, JOINT_CONFIG
from core.kinematics import calculate_target
from core.sim_viewer import MujocoSimViewer

# ==============================================================================
# 2. 再生対象ファイル設定
# ==============================================================================
CSV_FILENAME = "動作確認min_to_max.csv"  # 再生したいモーションファイル名
CSV_PATH = os.path.join(BASE_DIR, "motions", CSV_FILENAME)


# ==============================================================================
# 3. CSVデータ読み込み関数
# ==============================================================================
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


# ==============================================================================
# 4. 再生メイン処理
# ==============================================================================
def main():
    # CSVデータのロード
    frames = load_motion_data(CSV_PATH)
    if frames is None or len(frames) == 0:
        return

    print(f"📖 ロード完了: {CSV_PATH} ({len(frames)} フレーム)")

    # MuJoCo ビューア管理クラスの初期化
    try:
        sim = MujocoSimViewer()
    except Exception as e:
        print(f"❌ シミュレータの初期化に失敗しました: {e}")
        return

    # 3Dビューアを起動してメインループ開始
    with sim.launch():
        while sim.is_running():
            # キャッシュ・再生開始時刻の初期化
            prev_leader_cache = {}
            follower_current_cache = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}
            frame_idx = 0
            sim.reset_requested = False
            playback_start = time.time()
            total_duration = frames[-1][0]

            while sim.is_running() and frame_idx < len(frames):
                # Rキー押下（リセット要求）でループを抜けて最初から再スタート
                if sim.reset_requested:
                    break

                # Spaceキー押下中（一時停止中）の処理（時間を進めずに待機）
                if sim.paused:
                    time.sleep(0.02)
                    playback_start = time.time() - frames[frame_idx][0]
                    continue

                t_target, raw_positions = frames[frame_idx]

                # CSV のタイムスタンプに合わせて実時間同期
                elapsed = time.time() - playback_start
                if elapsed < t_target:
                    time.sleep(0.001)
                    continue

                # 各関節のフォロワー目標値 (Raw: 0〜4095) を計算
                target_positions = {}
                for sid in SERVO_IDS:
                    raw_val = raw_positions[sid]
                    target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                    prev_leader_cache[sid] = raw_val
                    follower_current_cache[sid] = target_val
                    target_positions[sid] = target_val

                # 3Dモデルの関節姿勢を一括更新
                sim.update_joints(target_positions)

                loop_text = "[Loop: ON]" if sim.loop_mode else "[Loop: OFF]"
                sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {total_duration:6.2f}s [Frame {frame_idx + 1}/{len(frames)}] {loop_text}  ")
                sys.stdout.flush()

                frame_idx += 1

            # 1回の再生が終了したときの処理
            if not sim.reset_requested:
                if sim.loop_mode:
                    time.sleep(0.3)
                    continue
                else:
                    print("\n✅ シミュレーション再生が完了しました。[R]キーで再再生、[Space]で一時停止を解除できます。")
                    # 再生終了後はビューアを開いたまま待機し、Rキー押下で再ループ
                    while sim.is_running() and not sim.reset_requested:
                        time.sleep(0.05)


if __name__ == "__main__":
    main()