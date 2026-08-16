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
TEST_IDS = [1, 2, 3, 4, 5, 6]

SERVO_IDS = [1, 2, 3, 4, 5, 6]
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# ==========================================
# 2. リーダー・フォロワーの可動範囲プロファイル
# ==========================================
# 物理調整により、フォロワー側の有限軸（ID1〜4, 6）はすべて 0/4095 境界跨ぎ無（f_cross: False）に統一
JOINT_CONFIG = {
    # ID1: リーダーは境界跨ぎ有（2850 -> 4095 -> 0 -> 1250: 連続値 2850〜5346）
    1: {
        "type": "bounded", 
        "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,
        "f_min": 850,  "f_max": 3400,        "f_cross": False,
        "init": 2048
    },
    # ID2: リーダーは境界跨ぎ有（1715 -> 4095 -> 0 -> 10: 連続値 1715〜4106）
    2: {
        "type": "bounded", 
        "r_min": 1715, "r_max": 4096 + 100,   "r_cross": True,
        "f_min": 942,  "f_max": 3270,        "f_cross": False,
        "init": 973
    },
    # ID3: リーダー・フォロワーともに境界跨ぎ無
    3: {
        "type": "bounded", 
        "r_min": 900,  "r_max": 3100,       "r_cross": False,
        "f_min": 834,  "f_max": 3061,        "f_cross": False,
        "init": 3061
    },
    # ID4: リーダー・フォロワーともに境界跨ぎ無
    4: {
        "type": "bounded", 
        "r_min": 1650, "r_max": 4015,       "r_cross": False,
        "f_min": 735,  "f_max": 3214,        "f_cross": False,
        "init": 735
    },
    # ID5: 無限回転軸（差分・相対追従）
    5: {
        "type": "infinite", 
        "init": 1023
    },
    # ID6: リーダー・フォロワーともに境界跨ぎ無
    6: {
        "type": "bounded", 
        "r_min": 1990, "r_max": 3000,       "r_cross": False,
        "f_min": 1890, "f_max": 3152,        "f_cross": False,
        "init": 1837
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

def write_follower_speed(ser, servo_id, speed):
    """
    フォロワー側へ速度（Speed / Wheel Mode用）を書き込む
    ※ Feetechサーボの速度制御レジスタ（通常アドレス 0x2E や 0x30 付近、または型番の仕様に合わせます）
    ここでは安全に符号付き速度をパケット化して送信します。
    """
    # 速度の上下限クランプ（例: -1000 〜 1000）
    speed = int(max(-1000, min(1000, speed)))
    
    addr = 0x2E       # 速度指令用レジスタアドレス（※STS3215等の仕様に合致させる）
    length = 5
    # 2バイトの符号付き整数（負数は2の補数表現）として分解
    if speed < 0:
        speed = 65536 + speed
    speed_l = speed & 0xFF
    speed_h = (speed >> 8) & 0xFF
    
    checksum = ~(servo_id + length + 0x03 + addr + speed_l + speed_h) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, speed_l, speed_h, checksum])
    
    ser.write(packet)
    time.sleep(0.001)

def set_servo_wheel_mode(ser, servo_id):
    """
    ID5を無限回転（速度制御モード / Wheel Mode）に設定する関数
    ※ Feetechサーボの動作モードレジスタ（通常 0x21 あたり、またはメーカー仕様に準拠）
    ※ RAM領域への書き込みなので電源OFFで元に戻る安全設計
    """
    addr = 0x21       # 動作モード設定レジスタの番地（※型番によって異なる場合がありますが一般的に0x21）
    length = 4
    mode_val = 3      # 3 または 1 が速度制御モード（ホイールモード）に相当
    
    checksum = ~(servo_id + length + 0x03 + addr + mode_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, mode_val, checksum])
    
    ser.write(packet)
    time.sleep(0.005)

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
            
        target = int(max(200,min(3900, target_linear)))
        return target

    elif config["type"] == "infinite":
            prev_raw = prev_raw_cache.get(sid, raw_leader)
            diff = raw_leader - prev_raw
            
            # 境界跨ぎの差分補正
            if diff > 2048:
                diff -= 4096
            elif diff < -2048:
                diff += 4096
                
            # 差分をそのまま速度指令値にスケーリング（ゲイン調整可能）
            gain = 5.0  # 追従の敏感さ（お好みで調整）
            speed_command = int(diff * direction * gain)
            
            return speed_command


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

        # ポートオープン＆バッファクリアの後...
        ser_leader.reset_input_buffer()
        ser_follower.reset_input_buffer()

        # 💡 【追加】ID5を速度制御モードに切り替えるパケットを送信
        print("⚙️ ID5を速度制御モード（Wheel Mode）に設定中...")
        set_servo_wheel_mode(ser_follower, 5)
        time.sleep(0.1)

        for sid in SERVO_IDS:
            if sid in TEST_IDS:
                set_torque(ser_follower, sid, True)
            else:
                set_torque(ser_follower, sid, False)        

        # 画面クリア
        sys.stdout.write("\033[2J")
        sys.stdout.flush()

        while True:
            # カーソルを左上に固定して上書き描画
            output_buffer = "\033[H"
            output_buffer += "========================================\n"
            output_buffer += " Tele Operation Monitor (Ctrl+Cで終了)\n"
            output_buffer += "========================================\n"
            for sid in SERVO_IDS:
                if sid in TEST_IDS:
                    # リーダーから生値を読み取る
                    raw_pos = read_servo_position(ser_leader, sid)
                    
                    if raw_pos is not None:
                        # 目標値（または速度値）の計算
                        command_val = calculate_target(sid, raw_pos, prev_leader_cache, follower_current_cache)
                        
                        # 💡 ID5 とそれ以外の軸（ID1〜4, 6）で書き込み処理を分岐
                        if sid == 5:
                            write_follower_speed(ser_follower, sid, command_val)
                            output_buffer += f" [ID {sid}]  Raw: {raw_pos:4d}  |  Speed: {command_val:4d}  |  Foll: (Velocity)\n"
                        else:
                            write_follower_position(ser_follower, sid, command_val)
                            output_buffer += f" [ID {sid}]  Raw: {raw_pos:4d}  |  Target: {command_val:4d}  |  Foll: (Sync)\n"
                        
                        prev_leader_cache[sid] = raw_pos
                        follower_current_cache[sid] = command_val
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