import sys
import time
import serial

# ==========================================
# 1. 通信設定・ポート設定
# ==========================================
LEADER_PORT = 'COM3'      # 操作側（リーダー）が接続されているシリアルポート
FOLLOWER_PORT = 'COM4'    # 追従側（フォロワー）が接続されているシリアルポート
BAUDRATE = 1000000        # サーボモーターの通信速度（1Mbps）

# 🧪 【試験用フィルタ】テストしたい関節のIDのみをリストに指定します
TEST_IDS = [2]

SERVO_IDS = [1, 2, 3, 4, 5, 6]
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# ==========================================
# 2. リーダー・フォロワーの可動範囲プロファイル
# ==========================================
JOINT_CONFIG = {
    1: {
        "type": "bounded", 
        "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,
        "f_min": 941,  "f_max": 3560,        "f_cross": False
    },
    2: {
        "type": "bounded", 
        "r_min": 1715, "r_max": 4096 + 10,   "r_cross": True,
        "f_min": 1929, "f_max": 4096 + 200,  "f_cross": True
    },
    3: {
        "type": "bounded", 
        "r_min": 900,  "r_max": 3100,       "r_cross": False,
        "f_min": 3929, "f_max": 4096 + 2046, "f_cross": True
    },
    4: {
        "type": "bounded", 
        "r_min": 1650, "r_max": 4015,       "r_cross": False,
        "f_min": 1780, "f_max": 4096 + 132,  "f_cross": True
    },
    5: {
        "type": "infinite", "init": 1930
    },
    6: {
        "type": "bounded", 
        "r_min": 1990, "r_max": 3000,       "r_cross": False,
        "f_min": 1837, "f_max": 3189,        "f_cross": False
    },
}

# ==========================================
# 3. 通信ヘルパー関数
# ==========================================
def read_servo_position(ser, servo_id):
    """指定サーボから現在位置の生データを読み取る（リーダー用）"""
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
    """サーボのトルクON/OFF"""
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
            
        if r_max == r_min:
            ratio = 0.0
        else:
            ratio = (raw_l - r_min) / (r_max - r_min)
        ratio = max(0.0, min(1.0, ratio))
        
        f_max_linear = f_max
        if f_cross and f_max < f_min:
            f_max_linear += 4096
            
        target_linear = f_min + ratio * (f_max_linear - f_min)
        
        if direction == -1:
            target_linear = f_min + (f_max_linear - target_linear)
            
        target = int(target_linear) % 4096
        return target

    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw
        
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096
            
        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)
        return int(new_target)


# ==========================================
# 5. メインループ（ちらつき防止版）
# ==========================================
def main():
    try:
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.01)
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)
        
        ser_leader.reset_input_buffer()
        ser_follower.reset_input_buffer()

        for sid in SERVO_IDS:
            if sid in TEST_IDS:
                set_torque(ser_follower, sid, True)
            else:
                set_torque(ser_follower, sid, False)
            
        prev_leader_cache = {}
        follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}

        # 画面クリア
        sys.stdout.write("\033[2J")
        sys.stdout.flush()

        while True:
            # カーソルを左上に固定して上書き描画
            output_buffer = "\033[H"
            output_buffer += "========================================\n"
            output_buffer += " 🚀 テレアーム同期モニター (Ctrl+Cで終了)\n"
            output_buffer += "========================================\n"

            for sid in SERVO_IDS:
                if sid in TEST_IDS:
                    # リーダーから生値を読み取る
                    raw_pos = read_servo_position(ser_leader, sid)
                    
                    if raw_pos is not None:
                        # 目標値計算
                        target_pos = calculate_target(sid, raw_pos, prev_leader_cache, follower_current_cache)
                        # フォロワーへ書き込み
                        write_follower_position(ser_follower, sid, target_pos)
                        
                        prev_leader_cache[sid] = raw_pos
                        follower_current_cache[sid] = target_pos
                        
                        target_val = follower_current_cache[sid]
                        # 💡 フォロワーの読取を廃止し、指令した Target 値を表示することで通信の競合・ちらつきを完全に排除
                        output_buffer += f" [ID {sid}]  Raw: {raw_pos:4d}  |  Target: {target_val:4d}  |  Foll: (Sync)\n"
                    else:
                        output_buffer += f" [ID {sid}]  Raw:  N/A   |  Target:  N/A   |  Foll: N/A\n"
                else:
                    output_buffer += f" [ID {sid}]  (OFF)\n"

            output_buffer += "========================================\n"

            sys.stdout.write(output_buffer)
            sys.stdout.flush()
            
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n🛑 停止しました。Follower のトルクをオフにします。")
        if 'ser_follower' in locals() and ser_follower.is_open:
            for sid in SERVO_IDS:
                set_torque(ser_follower, sid, False)
    except Exception as e:
        print(f"\n❌ エラー: {e}")
    finally:
        if 'ser_leader' in locals() and ser_leader.is_open:
            ser_leader.close()
        if 'ser_follower' in locals() and ser_follower.is_open:
            ser_follower.close()
        print("✅ ポートをクローズしました。")

if __name__ == "__main__":
    main()