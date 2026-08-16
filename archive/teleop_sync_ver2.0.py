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
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# ==========================================
# 2. 各関節の設定プロファイル
# ==========================================
# 【有限軸設定】
# リーダーの [min, max] を、フォロワーの [0, 4095] に線形マッピングする
# 境界跨ぎ「有」の軸は、4096を足して連続空間に拡張してある前提の数値
JOINT_CONFIG = {
    1: {"type": "bounded", "r_min": 2850, "r_max": 4096 + 920, "cross": True},   # 2850 ~ 5016
    2: {"type": "bounded", "r_min": 1715, "r_max": 4096 + 10,  "cross": True},   # 1715 ~ 4106
    3: {"type": "bounded", "r_min": 900,  "r_max": 3100,      "cross": False},  # 900 ~ 3100
    4: {"type": "bounded", "r_min": 1650, "r_max": 4015,      "cross": False},  # 1650 ~ 4015
    # ID5 は無限回転 (bounded ではなく infinite)
    5: {"type": "infinite"},
    6: {"type": "bounded", "r_min": 1990, "r_max": 2950,      "cross": False},  # 1990 ~ 2950
}

# フォロワー側の初期オフセット（ID5は無限なので現在位置ベース、他は中央など基準用）
OFFSET_FOLLOWER = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 1940, 6: 2048}


# ==========================================
# 3. 通信ヘルパー関数
# ==========================================
def read_leader_position(ser, servo_id):
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
    # 安全のため 0 ~ 4095 の範囲に完全にクランプ
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
    addr = 0x28
    length = 4
    en_val = 1 if enable else 0
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
    
    ser.write(packet)
    time.sleep(0.002)


# ==========================================
# 4. 変換ロジック（有限軸 & 無限軸）
# ==========================================
def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    config = JOINT_CONFIG[sid]
    direction = DIRECTION[sid]
    
    # --- パターンA: 有限軸（線形連続マッピング） ---
    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        cross = config["cross"]
        
        raw = raw_leader
        # 境界跨ぎがある軸で、値が小さい側（0付近）にいる場合は 4096 を足して連続空間に乗せる
        if cross and raw < (r_max - 4096):
            raw += 4096
            
        # 0.0 ~ 1.0 に正規化
        if r_max == r_min:
            ratio = 0.0
        else:
            ratio = (raw - r_min) / (r_max - r_min)
        ratio = max(0.0, min(1.0, ratio)) # はみ出し防止
        
        # フォロワー側の 0 ~ 4095 に変換
        target = ratio * 4095
        if direction == -1:
            target = 4095 - target
            
        return int(target)

    # --- パターンB: 無限回転軸（差分・相対追従） ---
    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw
        
        # 境界を跨いだ差分の補正
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096
            
        current_target = follower_current_cache.get(sid, 1940)
        new_target = current_target + (diff * direction)
        
        return int(new_target)


# ==========================================
# 5. メインループ
# ==========================================
def main():
    try:
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.01)
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)
        
        print(f"✅ Leader ({LEADER_PORT}) & Follower ({FOLLOWER_PORT}) 接続成功")
        
        ser_leader.reset_input_buffer()
        ser_follower.reset_input_buffer()

        # Follower トルクON & 初期位置セット
        for sid in SERVO_IDS:
            set_torque(ser_follower, sid, True)
            
        print("🚀 ハイブリッド同期制御を開始します。[Ctrl + C] で停止\n")

        prev_leader_cache = {}
        follower_current_cache = {5: 1940} # ID5の初期位置

        while True:
            log_msgs = []
            for sid in SERVO_IDS:
                raw_pos = read_leader_position(ser_leader, sid)
                
                if raw_pos is not None:
                    # 目標値計算
                    target_pos = calculate_target(sid, raw_pos, prev_leader_cache, follower_current_cache)
                    
                    # フォロワーへ書き込み
                    write_follower_position(ser_follower, sid, target_pos)
                    
                    # キャッシュ更新
                    prev_leader_cache[sid] = raw_pos
                    follower_current_cache[sid] = target_pos
                    
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