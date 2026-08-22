import sys
import time
import serial
import csv
import os
from datetime import datetime

# ==========================================
# 1. 通信設定・ファイル保存先設定
# ==========================================
LEADER_PORT = 'COM3'      # リーダーアームのポート 
BAUDRATE = 1000000        # 通信速度 
SERVO_IDS = [1, 2, 3, 4, 5, 6] 
SAMPLING_RATE_HZ = 50     # 記録周期（50Hz = 0.02秒間隔）

# スクリプトの場所（src/）を基準にして、プロジェクト直下の motions/ を確実に指すよう設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIONS_DIR = os.path.join(BASE_DIR, "motions")

# motionsフォルダが存在しない場合は自動作成
os.makedirs(MOTIONS_DIR, exist_ok=True)

# 日時付きのデフォルトファイル名
DEFAULT_FILENAME = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#DEFAULT_FILENAME = f"motion_B.csv"
OUTPUT_FILEPATH = os.path.join(MOTIONS_DIR, DEFAULT_FILENAME)

# ==========================================
# 2. リーダー読み取りヘルパー関数
# ==========================================
def read_servo_position(ser, servo_id):
    """指定サーボから現在位置の生データ（0〜4095）を読み取る""" 
    addr = 0x38 
    read_len = 2 
    length = 4 
    checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF 
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum]) 
    
    ser.write(packet) 
    time.sleep(0.001) 
    response = ser.read(8) 
    
    if len(response) >= 7 and response[2] == servo_id: 
        pos = response[5] | (response[6] << 8) 
        if 0 <= pos <= 4095: 
            return pos 
    return None 


# ==========================================
# 3. 記録メイン処理
# ==========================================
def main():
    print(f"🎬 記録を開始します。")
    print(f"📁 保存先: {OUTPUT_FILEPATH}")
    print("👉 リーダーアームを動かしてください。終了時は [Ctrl+C] を押してください。\n")

    try:
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.01) 
        ser_leader.reset_input_buffer() 

        with open(OUTPUT_FILEPATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー行を出力（時間経過 + 各関節ID）
            header = ["timestamp_sec"] + [f"id_{sid}" for sid in SERVO_IDS]
            writer.writerow(header)

            start_time = time.time()
            last_valid_positions = {sid: 2048 for sid in SERVO_IDS}
            interval = 1.0 / SAMPLING_RATE_HZ
            frame_count = 0

            while True:
                loop_start = time.time()
                current_timestamp = loop_start - start_time

                current_row = [f"{current_timestamp:.4f}"]

                for sid in SERVO_IDS:
                    raw_pos = read_servo_position(ser_leader, sid) 
                    if raw_pos is not None:
                        last_valid_positions[sid] = raw_pos
                    current_row.append(last_valid_positions[sid])

                writer.writerow(current_row)
                frame_count += 1

                # ターミナルへリアルタイム表示（1行固定更新）
                pos_str = " | ".join([f"ID{sid}:{last_valid_positions[sid]:4d}" for sid in SERVO_IDS])
                sys.stdout.write(f"\r⏱️ 記録中: {current_timestamp:6.2f}s [{frame_count:5d} frames] | {pos_str}")
                sys.stdout.flush()

                # ループ周期を一定に維持
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n🛑 記録を停止しました。")
        print(f"📁 計 {frame_count} フレームを '{OUTPUT_FILEPATH}' に保存しました。")
    except Exception as e:
        print(f"\n❌ エラー: {e}") 
    finally:
        if 'ser_leader' in locals() and ser_leader.is_open: 
            ser_leader.close() 


if __name__ == "__main__":
    main()