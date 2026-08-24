"""
==============================================================================
TACHIKOMA 統合遠隔操作スクリプト (src/teleop_main.py)
==============================================================================
【役割】
コマンドライン引数（実行時オプション）によって、以下の3機能を自由に組み合わせて実行します。
  1. 実機フォロワーの追従  （デフォルト: OFF / 有効化: --arm）
  2. 画面上の3D描画       （デフォルト: OFF / 有効化: --sim）
  3. 動作データのCSV記録  （デフォルト: OFF / 有効化: --record）

【実行コマンド例】
  - 画面シミュレーションのみ（安全なテスト用）
      python src/teleop_main.py --sim
  - 実機同期 ＋ 画面表示
      python src/teleop_main.py --arm --sim
  - 実機同期 ＋ 画面表示 ＋ CSV記録（フル稼働）
      python src/teleop_main.py --arm --sim --record
  - 実機同期のみ
      python src/teleop_main.py --arm
==============================================================================
"""

import sys
import time
import csv
import os
import argparse
from datetime import datetime

# ==============================================================================
# 1. 共通モジュール・設定のインポート
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import (
    LEADER_PORT,
    FOLLOWER_PORT,
    BAUDRATE,
    SAMPLING_RATE_HZ,
    SERVO_IDS,
    JOINT_CONFIG
)
from core.sts3215 import STS3215Driver
from core.kinematics import calculate_target
from core.sim_viewer import MujocoSimViewer


# ==============================================================================
# 2. 実行時コマンドライン引数の解析 (argparse)
# ==============================================================================
def parse_arguments():
    """コマンドライン引数を定義・取得する関数"""
    parser = argparse.ArgumentParser(
        description="TACHIKOMA リアルタイム遠隔操作 統合メインスクリプト"
    )

    # --arm オプション (指定すると True になり、実機フォロワーへ送信する)
    parser.add_argument(
        "--arm",
        action="store_true",
        help="実機フォロワーへのコマンド送信・追従動作を有効化します（デフォルト: OFF）"
    )

    # --sim オプション (指定すると True になる)
    parser.add_argument(
        "--sim",
        action="store_true",
        help="MuJoCo 3Dシミュレーション描画ウィンドウを起動します（デフォルト: OFF）"
    )

    # --record オプション (指定すると True になる)
    parser.add_argument(
        "--record",
        action="store_true",
        help="CSVファイルへの動作データ記録を有効化します（デフォルト: OFF）"
    )

    return parser.parse_args()


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    args = parse_arguments()

    # 引数から各機能の有効/無効フラグを決定（すべて明示指定で ON になる設計）
    enable_follower = args.arm
    enable_sim = args.sim
    enable_record = args.record

    # 動作モードの表示文字列を作成
    mode_list = []
    if enable_follower:
        mode_list.append("🤖 実機フォロワー同期")
    if enable_sim:
        mode_list.append("🖥️ 3D画面描画")
    if enable_record:
        mode_list.append("📁 CSV記録")
    if not mode_list:
        mode_list.append("ターミナル数値モニタのみ")

    print("==================================================")
    print(f" 🎬 TACHIKOMA 統合制御システム")
    print(f" ⚙️ 有効機能: {' + '.join(mode_list)}")
    print("==================================================")

    # --- 保存先ファイルの設定（記録モード時のみ） ---
    output_filepath = None
    csv_file = None
    csv_writer = None

    if enable_record:
        motions_dir = os.path.join(BASE_DIR, "motions")
        os.makedirs(motions_dir, exist_ok=True)
        filename = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_filepath = os.path.join(motions_dir, filename)
        print(f"📁 記録先: {output_filepath}")

    print("👉 リーダーアームを操作してください。")
    print("👉 終了時は [Ctrl+C] または ウィンドウを閉じてください。\n")

    leader = None
    follower = None
    sim = None

    # --- 1. リーダーポート（操作側）のオープン ---
    try:
        leader = STS3215Driver(LEADER_PORT, baudrate=BAUDRATE, timeout=0.01)
        print(f"✅ リーダー接続完了 ({LEADER_PORT})")
    except Exception as e:
        print(f"❌ リーダーの接続に失敗しました ({LEADER_PORT}): {e}")
        return

    # --- 2. フォロワーポート（実機追従側）のオープン判定 ---
    is_follower_active = False
    if enable_follower:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            for sid in SERVO_IDS:
                follower.set_torque(sid, True)
            is_follower_active = True
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"⚠️ 実機フォロワー接続失敗: {e}")
            print("👉 実機送信はスキップして継続します。")
    else:
        print("💡 --arm が指定されていないため、実機フォロワーは動作しません（安全モード）。")

    # --- 3. MuJoCo 3Dシミュレータの初期化判定 ---
    if enable_sim:
        try:
            sim = MujocoSimViewer()
            print("✅ 3Dシミュレータ初期化完了")
        except Exception as e:
            print(f"⚠️ 3Dシミュレータ初期化失敗: {e}")
            print("👉 3D描画はスキップします。")
            sim = None
    else:
        print("💡 --sim が指定されていないため、3D画面描画はスキップします。")

    # --- 4. CSV記録ファイルのオープン ---
    if enable_record:
        csv_file = open(output_filepath, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        header = ["timestamp_sec"] + [f"id_{sid}" for sid in SERVO_IDS]
        csv_writer.writerow(header)

    # 追従計算用キャッシュの準備
    prev_leader_cache = {}
    follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}
    last_valid_positions = {sid: 2048 for sid in SERVO_IDS}

    start_time = time.time()
    interval = 1.0 / SAMPLING_RATE_HZ  # 50Hz = 0.02秒
    frame_count = 0
    first_draw = True

    # ターミナルのカーソルを非表示
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        # メインループ関数
        def run_loop():
            nonlocal frame_count, first_draw
            while True:
                # 3D画面が有効かつウィンドウが閉じられたら終了
                if sim is not None and not sim.is_running():
                    break

                loop_start = time.time()
                current_timestamp = loop_start - start_time

                current_csv_row = [f"{current_timestamp:.4f}"] if enable_record else []
                target_positions = {}

                # ターミナル表示文字列の作成
                lines = []
                lines.append("========================================")
                lines.append(f" ⏱️ 動作中  {current_timestamp:5.2f}s [{frame_count:5d} frames]")
                lines.append(f" 🤖 実機: {'ON' if is_follower_active else 'OFF'} | 🖥️ 3D: {'ON' if sim is not None else 'OFF'} | 📁 記録: {'ON' if enable_record else 'OFF'}")
                lines.append("========================================")

                # --- 各軸の読み取り・計算・送信 ---
                for sid in SERVO_IDS:
                    raw_pos = leader.read_position(sid)
                    if raw_pos is not None:
                        last_valid_positions[sid] = raw_pos

                    current_raw = last_valid_positions[sid]

                    # CSV記録用データへ追加
                    if enable_record:
                        current_csv_row.append(current_raw)

                    # 目標値計算 (0〜4095)
                    target_pos = calculate_target(sid, current_raw, prev_leader_cache, follower_current_cache)
                    prev_leader_cache[sid] = current_raw
                    follower_current_cache[sid] = target_pos
                    target_positions[sid] = target_pos

                    # 実機フォロワーへ送信（--arm 指定時のみ）
                    if is_follower_active:
                        follower.write_position(sid, target_pos)

                    lines.append(f" [ID {sid}]  Raw: {current_raw:4d}  |  Target: {target_pos:4d}")

                # 3Dモデルの描画更新（--sim 指定時のみ）
                if sim is not None:
                    sim.update_joints(target_positions)

                lines.append("========================================")
                lines.append(" [Ctrl+C] で終了")

                # コンソール上書き表示
                if not first_draw:
                    sys.stdout.write(f"\033[{len(lines)}A")
                else:
                    first_draw = False

                sys.stdout.write("\n".join(line + "\033[K" for line in lines) + "\n")
                sys.stdout.flush()

                # CSV書き込み（--record 指定時のみ）
                if enable_record and csv_writer:
                    csv_writer.writerow(current_csv_row)

                frame_count += 1

                # 50Hz周期を維持
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # 3D画面が有効なら launch コンテキスト内で実行、無効なら通常実行
        if sim is not None:
            with sim.launch():
                run_loop()
        else:
            run_loop()

    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n\n🛑 停止しました。\n")
        if enable_record:
            print(f"📁 計 {frame_count} フレームを '{output_filepath}' に保存しました。")
    except Exception as e:
        sys.stdout.write("\033[?25h")
        print(f"\n❌ エラー: {e}")
    finally:
        sys.stdout.write("\033[?25h")  # カーソル再表示
        if csv_file is not None and not csv_file.closed:
            csv_file.close()
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
        if leader is not None:
            leader.close()
        print("✅ 全リソースを安全にクローズしました。")


if __name__ == "__main__":
    main()