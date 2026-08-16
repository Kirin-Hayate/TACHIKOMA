import sys
import time
import serial
import csv
import os

# ==========================================
# 1. 再生設定・通信設定
# ==========================================
PLAYBACK_FILENAME = "arm_nobasu.csv"  # ファイル名
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYBACK_FILE = os.path.join(BASE_DIR, "motions", PLAYBACK_FILENAME)  # 👈 ここに再生したいCSVファイル名を指定
REPEAT_COUNT = 3                   # 👈 再生回数（何回繰り返すか）
SLOW_START_TIME = 1.5              # 初回開始時に初期位置へゆっくり移動させる秒数（安全対策）

FOLLOWER_PORT = 'COM4'             # フォロワーアームのポート 
BAUDRATE = 1000000 
SERVO_IDS = [1, 2, 3, 4, 5, 6] 
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1} 

# ==========================================
# 2. リーダー・フォロワー可動範囲マッピングプロファイル
# ==========================================
JOINT_CONFIG = {
    1: {"type": "bounded", "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,  "f_min": 850,  "f_max": 3400, "f_cross": False},
    2: {"type": "bounded", "r_min": 1715, "r_max": 4096 + 100,  "r_cross": True,  "f_min": 942,  "f_max": 3270, "f_cross": False},
    3: {"type": "bounded", "r_min": 900,  "r_max": 3100,        "r_cross": False, "f_min": 834,  "f_max": 3061, "f_cross": False},
    4: {"type": "bounded", "r_min": 1650, "r_max": 4015,        "r_cross": False, "f_min": 735,  "f_max": 3214, "f_cross": False},
    5: {"type": "infinite", "init": 1023},
    6: {"type": "bounded", "r_min": 1990, "r_max": 3000,        "r_cross": False, "f_min": 1890, "f_max": 3152, "f_cross": False},
}

# ==========================================
# 3. 通信・制御ヘルパー関数
# ==========================================
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
    """サーボのトルクON/OFF""" 
    addr = 0x28 
    length = 4 
    en_val = 1 if enable else 0 
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF 
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum]) 
    ser.write(packet) 
    time.sleep(0.002) 


def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    """リーダー生値からフォロワーのTarget位置を計算""" 
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
            
        return int(max(0, min(4095, target_linear)))

    elif config["type"] == "infinite": 
        prev_raw = prev_raw_cache.get(sid, raw_leader) 
        diff = raw_leader - prev_raw 
        if diff > 2048: 
            diff -= 4096 
        elif diff < -2048: 
            diff += 4096 
            
        current_target = follower_current_cache.get(sid, config["init"]) 
        new_target = current_target + (diff * direction) 
        return int(new_target) % 4096


# ==========================================
# 4. 再生メイン処理
# ==========================================
def load_motion_data(filepath):
    """CSVからモーションデータを読み込む"""
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


def main():
    frames = load_motion_data(PLAYBACK_FILE)
    if not frames:
        return

    print(f"📖 ファイル '{PLAYBACK_FILE}' をロードしました（計 {len(frames)} フレーム）")

    try:
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01) 
        ser_follower.reset_input_buffer() 

        # トルクON 
        for sid in SERVO_IDS: 
            set_torque(ser_follower, sid, True) 

        for loop_idx in range(REPEAT_COUNT):
            print(f"\n▶️ モーション再生開始 [{loop_idx + 1} / {REPEAT_COUNT} 回目]")

            prev_leader_cache = {} 
            follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS} 

            # 最初の目標値を事前計算
            first_raw_positions = frames[0][1]
            first_targets = {}
            for sid in SERVO_IDS:
                raw_val = first_raw_positions[sid]
                t_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                first_targets[sid] = t_val
                prev_leader_cache[sid] = raw_val
                follower_current_cache[sid] = t_val

            # 安全対策：初回はアームをゆっくり初期位置に移動させる
            if loop_idx == 0:
                print(f"⏳ 初期位置へゆっくり移動中 ({SLOW_START_TIME}秒)...")
                for sid in SERVO_IDS:
                    write_follower_position(ser_follower, sid, first_targets[sid]) 
                time.sleep(SLOW_START_TIME)

            playback_start_time = time.time()

            for t_target, raw_positions in frames:
                # 記録されたタイムスタンプと同期をとる
                while (time.time() - playback_start_time) < t_target:
                    time.sleep(0.001)

                for sid in SERVO_IDS:
                    raw_val = raw_positions[sid]
                    target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                    write_follower_position(ser_follower, sid, target_val) 
                    
                    prev_leader_cache[sid] = raw_val
                    follower_current_cache[sid] = target_val

                sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {frames[-1][0]:6.2f}s")
                sys.stdout.flush()

        print("\n\n✅ すべての再生が完了しました。")

    except KeyboardInterrupt:
        print("\n\n🛑 再生を中断しました。")
    except Exception as e:
        print(f"\n❌ エラー: {e}") 
    finally:
        if 'ser_follower' in locals() and ser_follower.is_open: 
            for sid in SERVO_IDS: 
                set_torque(ser_follower, sid, False) 
            ser_follower.close() 
            print("✅ フォロワーのトルクをOFFにし、ポートをクローズしました。")


if __name__ == "__main__":
    main()