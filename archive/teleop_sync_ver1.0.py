import sys
import time
import serial

# ==========================================
# 1. 通信設定・ポート設定
# ==========================================
LEADER_PORT = 'COM3'
FOLLOWER_PORT = 'COM4'
BAUDRATE = 1000000

SERVO_IDS = [1, 2, 3, 4, 5, 6]

# ==========================================
# 2. 基準オフセット値 & 設定
# ==========================================
OFFSET_LEADER = {1: 0, 2: 1750, 3: 3104, 4: 1643, 5: 1952, 6: 1980}
OFFSET_FOLLOWER = {1: 2166, 2: 1967, 3: 2022, 4: 1754, 5: 1021, 6: 1843}

DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

LIMIT_MIN = 50
LIMIT_MAX = 4045


# ==========================================
# 3. 通信ヘルパー関数
# ==========================================
def read_leader_position(ser, servo_id):
    """Leader から確実に位置を読み取るパケット送信"""
    addr = 0x38
    read_len = 2
    length = 4
    checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])
    
    ser.write(packet)
    time.sleep(0.001)
    response = ser.read(8)
    
    if len(response) >= 7 and response[2] == servo_id:
        return response[5] | (response[6] << 8)
    return None


def write_follower_position(ser, servo_id, pos):
    """Follower へ目標位置を書き込む"""
    addr = 0x2A
    length = 5
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    checksum = ~(servo_id + length + 0x03 + addr + pos_l + pos_h) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, pos_l, pos_h, checksum])
    
    ser.write(packet)
    time.sleep(0.001)


def set_torque(ser, servo_id, enable):
    """トルクのON/OFF"""
    addr = 0x28
    length = 4
    en_val = 1 if enable else 0
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
    
    ser.write(packet)
    time.sleep(0.002)


def calculate_follower_target(servo_id, raw_leader):
    offset_l = OFFSET_LEADER[servo_id]
    offset_f = OFFSET_FOLLOWER[servo_id]
    direction = DIRECTION[servo_id]

    diff = raw_leader - offset_l
    if diff > 2048:
        diff -= 4096
    elif diff < -2048:
        diff += 4096

    target = offset_f + (diff * direction)

    if target < LIMIT_MIN:
        target = LIMIT_MIN
    elif target > LIMIT_MAX:
        target = LIMIT_MAX

    return int(target)


# ==========================================
# 4. メインループ
# ==========================================
def main():
    try:
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.01)
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)
        
        print(f"✅ Leader ({LEADER_PORT}) & Follower ({FOLLOWER_PORT}) 接続成功")
        
        # 起動時にバッファをクリア
        ser_leader.reset_input_buffer()
        ser_follower.reset_input_buffer()

        # Follower トルクON
        for sid in SERVO_IDS:
            set_torque(ser_follower, sid, True)
            
        print("🚀 同期制御を開始します。[Ctrl + C] で停止\n")

        while True:
            log_msgs = []
            for sid in SERVO_IDS:
                # 1. Leader から現在位置を取得
                raw_pos = read_leader_position(ser_leader, sid)
                
                if raw_pos is not None:
                    # 2. 目標値計算 & Follower へ送信
                    target_pos = calculate_follower_target(sid, raw_pos)
                    write_follower_position(ser_follower, sid, target_pos)
                    log_msgs.append(f"ID{sid}:{target_pos}")
                else:
                    log_msgs.append(f"ID{sid}: N/A")

            print("\r" + " | ".join(log_msgs), end="", flush=True)
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