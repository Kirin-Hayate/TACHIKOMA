import sys
import time
import serial
import csv
import os
from datetime import datetime

# ==========================================
# 1. 通信設定・ファイル保存先設定
# ==========================================
RECORD_MODE = True       # 👈 True: 同期＋記録 / False: 同期のみ
LEADER_PORT = 'COM3'      # 操作側（リーダー）のシリアルポート
FOLLOWER_PORT = 'COM4'    # 追従側（フォロワー）のシリアルポート
BAUDRATE = 1000000        # 通信速度（1Mbps）
SAMPLING_RATE_HZ = 50     # 記録・同期周期（50Hz = 0.02秒間隔）

# 🧪 【試験用フィルタ】同期させたい関節IDを指定
TEST_IDS = [1, 2, 3, 4, 5, 6]
SERVO_IDS = [1, 2, 3, 4, 5, 6]
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# スクリプトの場所を基準にして motions/ フォルダへ保存
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIONS_DIR = os.path.join(BASE_DIR, "motions")
os.makedirs(MOTIONS_DIR, exist_ok=True)

# 日時付きファイル名の生成
DEFAULT_FILENAME = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
OUTPUT_FILEPATH = os.path.join(MOTIONS_DIR, DEFAULT_FILENAME)

# ==========================================
# 2. リーダー・フォロワー可動範囲プロファイル
# ==========================================
JOINT_CONFIG = {
    1: {
        "type": "bounded", 
        "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,
        "f_min": 850,  "f_max": 3400,        "f_cross": False,
        "init": 2500
    },
    2: {
        "type": "bounded", 
        "r_min": 1715, "r_max": 4096 + 100,   "r_cross": True,
        "f_min": 942,  "f_max": 3270,        "f_cross": False,
        "init": 973
    },
    3: {
        "type": "bounded", 
        "r_min": 900,  "r_max": 3100,        "r_cross": False,
        "f_min": 834,  "f_max": 3061,        "f_cross": False,
        "init": 3061
    },
    4: {
        "type": "bounded", 
        "r_min": 1650, "r_max": 4015,        "r_cross": False,
        "f_min": 735,  "f_max": 3214,        "f_cross": False,
        "init": 735
    },
    5: {
        "type": "infinite", 
        "init": 1023
    },
    6: {
        "type": "bounded", 
        "r_min": 1990, "r_max": 3000,        "r_cross": False,
        "f_min": 1890, "f_max": 3152,        "f_cross": False,
        "init": 1837
    },
}

# ==========================================
# 3. 通信ヘルパー関数
# ==========================================
def read_servo_position(ser, servo_id):
    """リーダーサーボから現在位置の生データ（0〜4095）を読み取る"""
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


def write_follower_position(ser, servo_id, pos):
    """フォロワー側へ目標位置を書き込む"""
    pos = int(max(0, min(4095, pos)))
    addr = 0x2A
    length = 5
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    checksum = ~(servo_id + length + 0x03 + addr + pos_l + pos_h) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, pos_l, pos_h, checksum])
    ser.write(packet)
    time.sleep(0.001)


def set_torque(ser, servo_id, enable):
    """フォロワーサーボのトルクON/OFF"""
    addr = 0x28
    length = 4
    en_val = 1 if enable else 0
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
    ser.write(packet)
    time.sleep(0.002)


# ==========================================
# 4. 1:1 等倍マッピング変換ロジック
# ==========================================
def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    config = JOINT_CONFIG[sid]
    direction = DIRECTION[sid]
    
    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        r_cross = config["r_cross"]
        f_min = config["f_min"]
        f_max = config["f_max"]
        f_cross = config["f_cross"]
        
        raw_l = raw_leader
        if r_cross and raw_l < (r_max - 4096):
            raw_l += 4096
            
        ratio = 0.0 if r_max == r_min else (raw_l - r_min) / (r_max - r_min)
        ratio = max(0.0, min(1.0, ratio))
        
        f_max_linear = f_max
        if f_cross and f_max < f_min:
            f_max_linear += 4096
            
        target_linear = f_min + ratio * (f_max_linear - f_min)
        if direction == -1:
            target_linear = f_min + (f_max_linear - target_linear)
            
        return int(max(200, min(3900, target_linear)))

    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096
            
        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)
        return new_target  # ⭕ teleop_sync.py と同様に剰余を外して連続値を返す


# ==========================================
# 5. 同期＆記録メイン処理
# ==========================================
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

    try:
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.01)
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)

        ser_leader.reset_input_buffer()
        ser_follower.reset_input_buffer()

        # 指定IDのフォロワートルクをON
        for sid in SERVO_IDS:
            set_torque(ser_follower, sid, sid in TEST_IDS)

        prev_leader_cache = {}
        follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}
        last_valid_positions = {sid: 2048 for sid in SERVO_IDS}

        # 記録モード時のみCSVファイルをオープン
        if RECORD_MODE:
            f = open(OUTPUT_FILEPATH, mode='w', newline='', encoding='utf-8')
            writer = csv.writer(f)
            header = ["timestamp_sec"] + [f"id_{sid}" for sid in SERVO_IDS]
            writer.writerow(header)

        start_time = time.time()
        interval = 1.0 / SAMPLING_RATE_HZ
        frame_count = 0
        first_draw = True

        # 画面のちらつき防止のためカーソルを非表示
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        while True:
            loop_start = time.time()
            current_timestamp = loop_start - start_time

            current_csv_row = [f"{current_timestamp:.4f}"] if RECORD_MODE else []
            mode_label = "記録中" if RECORD_MODE else "同期中"
            
            lines = []
            lines.append("========================================")
            lines.append(f" ⏱️ 状態: {mode_label}  {current_timestamp:5.2f}s [{frame_count:5d} frames]")
            if RECORD_MODE:
                lines.append(f" 📁 {os.path.basename(OUTPUT_FILEPATH)}")
            lines.append("========================================")

            for sid in SERVO_IDS:
                raw_pos = read_servo_position(ser_leader, sid)
                
                if raw_pos is not None:
                    last_valid_positions[sid] = raw_pos

                current_raw = last_valid_positions[sid]
                if RECORD_MODE:
                    current_csv_row.append(current_raw)

                # フォロワー同期制御
                if sid in TEST_IDS and raw_pos is not None:
                    target_pos = calculate_target(sid, current_raw, prev_leader_cache, follower_current_cache)
                    write_follower_position(ser_follower, sid, target_pos)

                    prev_leader_cache[sid] = current_raw
                    follower_current_cache[sid] = target_pos
                    lines.append(f" [ID {sid}]  Raw: {current_raw:4d}  |  Target: {target_pos:4d}  |  (Sync)")
                else:
                    lines.append(f" [ID {sid}]  Raw: {current_raw:4d}  |  (OFF)")

            lines.append("========================================")
            lines.append(" [Ctrl+C] で停止して終了")

            # 2回目以降のループでは、前回出力した行数分だけカーソルを上に戻して上書き
            if not first_draw:
                sys.stdout.write(f"\033[{len(lines)}A")
            else:
                first_draw = False

            # 各行を行末クリアしながら出力
            sys.stdout.write("\n".join(line + "\033[K" for line in lines) + "\n")
            sys.stdout.flush()

            # 記録モード時のみCSVへ書き込み
            if RECORD_MODE and writer:
                writer.writerow(current_csv_row)
            frame_count += 1

            # 50Hz周期を維持
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
        sys.stdout.write("\033[?25h")  # カーソルを確実に再表示
        if f is not None and not f.closed:
            f.close()
        if 'ser_follower' in locals() and ser_follower.is_open:
            for sid in SERVO_IDS:
                set_torque(ser_follower, sid, False)
            ser_follower.close()
        if 'ser_leader' in locals() and ser_leader.is_open:
            ser_leader.close()
        print("✅ フォロワーのトルクをOFFにし、全ポートをクローズしました。")


if __name__ == "__main__":
    main()