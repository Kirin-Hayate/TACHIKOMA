"""
==============================================================================
リアルタイム遠隔操作 & モーション記録スクリプト (src/teleop_sync_and_record.py)
==============================================================================
【役割】
1. 操作側（リーダー）の動きを読み取り、追従側（フォロワー）へリアルタイム送信します。
2. RECORD_MODE = True の場合、動作データを CSV ファイルとして motions/ フォルダへ記録します。
3. 通信ドライバや角度計算などの低レイヤ処理はすべて core/ および config/ から呼び出します。
==============================================================================
"""

import sys
import time
import csv
import os
from datetime import datetime

# ==============================================================================
# 1. 共通モジュール・設定のインポート
# ==============================================================================
# プロジェクトルートディレクトリのパスを通す
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 設定・通信・計算クラスの読み込み
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

# ==============================================================================
# 2. 動作モード & 保存先ファイル設定
# ==============================================================================
RECORD_MODE = True       # True: 同期 + CSV記録 / False: 同期専用 (記録OFF)
TEST_IDS = [1, 2, 3, 4, 5, 6]  # 同期させる関節ID（テスト時はここを絞り込めます）

# モーション保存用フォルダの作成
MOTIONS_DIR = os.path.join(BASE_DIR, "motions")
os.makedirs(MOTIONS_DIR, exist_ok=True)

# 日時付きファイル名の生成
#一時的に任意のファイル名に変更する場合のみ以下の一文をコメントアウト
DEFAULT_FILENAME = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#DEFAULT_FILENAME = f"動作確認min_to_max.csv"
OUTPUT_FILEPATH = os.path.join(MOTIONS_DIR, DEFAULT_FILENAME)


# ==============================================================================
# 3. メイン同期・記録ループ
# ==============================================================================
def main():
    mode_title = "同期 ＆ モーション記録モード" if RECORD_MODE else "同期専用モード (記録OFF)"
    print("========================================")
    print(f" 🎬 TACHIKOMA {mode_title}")
    print("========================================")
    if RECORD_MODE:
        print(f"📁 保存先: {OUTPUT_FILEPATH}")
    print("👉 リーダーアームを操作してください。終了時は [Ctrl+C] を押してください。\n")

    f = None
    writer = None
    leader = None
    follower = None

    try:
        # --- 通信ドライバの初期化 ---
        leader = STS3215Driver(LEADER_PORT, baudrate=BAUDRATE, timeout=0.01)
        follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)

        # 指定IDのフォロワートルクをON（動かさない軸はフリー）
        for sid in SERVO_IDS:
            follower.set_torque(sid, sid in TEST_IDS)

        # 差分追従（ID5）および通信エラー対策用のキャッシュ
        prev_leader_cache = {}
        follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}
        last_valid_positions = {sid: 2048 for sid in SERVO_IDS}

        # 記録モード時のみ CSV ファイルを新規作成してヘッダーを書き込む
        if RECORD_MODE:
            f = open(OUTPUT_FILEPATH, mode='w', newline='', encoding='utf-8')
            writer = csv.writer(f)
            header = ["timestamp_sec"] + [f"id_{sid}" for sid in SERVO_IDS]
            writer.writerow(header)

        start_time = time.time()
        interval = 1.0 / SAMPLING_RATE_HZ  # 50Hz = 0.02秒
        frame_count = 0
        first_draw = True

        # 画面のちらつき防止のためターミナルのカーソルを非表示
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        while True:
            loop_start = time.time()
            current_timestamp = loop_start - start_time

            current_csv_row = [f"{current_timestamp:.4f}"] if RECORD_MODE else []
            mode_label = "記録中" if RECORD_MODE else "同期中"

            # ターミナル表示用バッファの構築
            lines = []
            lines.append("========================================")
            lines.append(f" ⏱️ 状態: {mode_label}  {current_timestamp:5.2f}s [{frame_count:5d} frames]")
            if RECORD_MODE:
                lines.append(f" 📁 {os.path.basename(OUTPUT_FILEPATH)}")
            lines.append("========================================")

            # --- 全軸の読み出し・計算・追従処理 ---
            for sid in SERVO_IDS:
                # リーダーの生値を読み取り
                raw_pos = leader.read_position(sid)

                if raw_pos is not None:
                    last_valid_positions[sid] = raw_pos

                current_raw = last_valid_positions[sid]
                if RECORD_MODE:
                    current_csv_row.append(current_raw)

                # フォロワーへの目標値計算と送信
                if sid in TEST_IDS and raw_pos is not None:
                    # core/kinematics.py の関数で目標値を算出
                    target_pos = calculate_target(sid, current_raw, prev_leader_cache, follower_current_cache)
                    follower.write_position(sid, target_pos)

                    # キャッシュ更新
                    prev_leader_cache[sid] = current_raw
                    follower_current_cache[sid] = target_pos

                    lines.append(f" [ID {sid}]  Raw: {current_raw:4d}  |  Target: {target_pos:4d}  |  (Sync)")
                else:
                    lines.append(f" [ID {sid}]  Raw: {current_raw:4d}  |  (OFF)")

            lines.append("========================================")
            lines.append(" [Ctrl+C] で停止して終了")

            # 2回目以降の描画では、前回描画した行数分カーソルを巻き戻して上書き固定表示
            if not first_draw:
                sys.stdout.write(f"\033[{len(lines)}A")
            else:
                first_draw = False

            sys.stdout.write("\n".join(line + "\033[K" for line in lines) + "\n")
            sys.stdout.flush()

            # CSVへ1フレーム書き込み
            if RECORD_MODE and writer:
                writer.writerow(current_csv_row)
            frame_count += 1

            # 50Hz (0.02秒) 周期を正確に維持
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n\n🛑 停止しました。フォロワーのトルクをOFFにします。\n")
        if RECORD_MODE:
            print(f"📁 計 {frame_count} フレームを '{OUTPUT_FILEPATH}' に保存しました。")
    except Exception as e:
        sys.stdout.write("\033[?25h")
        print(f"\n❌ エラー: {e}")
    finally:
        sys.stdout.write("\033[?25h")  # カーソル再表示
        if f is not None and not f.closed:
            f.close()
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
        if leader is not None:
            leader.close()
        print("✅ フォロワーのトルクをOFFにし、全ポートをクローズしました。")


if __name__ == "__main__":
    main()