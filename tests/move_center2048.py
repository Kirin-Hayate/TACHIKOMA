import time
import serial

# ==========================================
# 通信ポート・設定
# ==========================================
FOLLOWER_PORT = 'COM4'    # フォロワー側が接続されているシリアルポート
BAUDRATE = 1000000        # サーボモーターの通信速度 (1Mbps)
TARGET_ID = 3             # 動かすサーボのID
GOAL_POS = 2048           # 目標値 (中央: 2048)

def set_torque(ser, servo_id, enable):
    """サーボのトルクON/OFFを切り替える関数"""
    addr = 0x28
    length = 4
    en_val = 1 if enable else 0
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
    
    ser.write(packet)
    time.sleep(0.002)

def write_follower_position(ser, servo_id, pos):
    """フォロワー側へ目標位置（Goal Position）を書き込む関数"""
    # 安全のため 0 〜 4095 の範囲にクランプ
    pos = int(max(0, min(4095, pos)))
    
    addr = 0x2A       # 目標位置レジスタアドレス
    length = 5
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    checksum = ~(servo_id + length + 0x03 + addr + pos_l + pos_h) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, pos_l, pos_h, checksum])
    
    ser.write(packet)
    time.sleep(0.001)

def main():
    try:
        # フォロワー側のポートを開く
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)
        print(f"✅ フォロワーポート ({FOLLOWER_PORT}) に接続しました。")
        
        ser_follower.reset_input_buffer()

        # 1. トルクをONにしてモーターを有効化
        print(f"🔓 ID {TARGET_ID} のトルクをONにします...")
        set_torque(ser_follower, TARGET_ID, True)
        time.sleep(0.1)

        # 2. 目標値 2048 を送信
        print(f"🎯 ID {TARGET_ID} に目標値 [{GOAL_POS}] を送信します...")
        write_follower_position(ser_follower, TARGET_ID, GOAL_POS)
        
        # モーターが位置に到達するまで少し待機
        time.sleep(1.0)
        print("✨ 指令の送信が完了しました。")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    finally:
        if 'ser_follower' in locals() and ser_follower.is_open:
            ser_follower.close()
            print("✅ ポートをクローズしました。")

if __name__ == "__main__":
    main()