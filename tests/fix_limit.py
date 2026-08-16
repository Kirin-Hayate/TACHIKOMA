import time
import serial

FOLLOWER_PORT = "COM4"
BAUDRATE = 1000000
TARGET_ID = 4  # 制限がかかっているID4を指定

# STS3215 のリミットに関するレジスタアドレス
# Min Angle Limit: 0x06 (6), Max Angle Limit: 0x08 (8) （各2バイト）
ADDR_MIN_LIMIT = 6
ADDR_MAX_LIMIT = 8

def set_servo_register(ser, servo_id, addr, value, length=2):
    """サーボのレジスタに値を書き込む汎用関数"""
    if length == 2:
        val_l = value & 0xFF
        val_h = (value >> 8) & 0xFF
        checksum = ~(servo_id + 5 + 0x03 + addr + val_l + val_h) & 0xFF
        packet = bytes([0xFF, 0xFF, servo_id, 5, 0x03, addr, val_l, val_h, checksum])
    else:
        checksum = ~(servo_id + 4 + 0x03 + addr + value) & 0xFF
        packet = bytes([0xFF, 0xFF, servo_id, 4, 0x03, addr, value, checksum])
    
    ser.write(packet)
    time.sleep(0.01)

def main():
    try:
        ser = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.1)
        print(f"✅ Follower ({FOLLOWER_PORT}) を開きました。ID {TARGET_ID} のリミットを解除します。")
        
        # トルクを一度オフにする（書き込みのため）
        # トルクアドレス: 0x28 (40)
        torque_packet = bytes([0xFF, 0xFF, TARGET_ID, 4, 0x03, 40, 0, ~(TARGET_ID + 4 + 0x03 + 40 + 0) & 0xFF])
        ser.write(torque_packet)
        time.sleep(0.05)
        
        # 最小リミットを 0 に設定
        set_servo_register(ser, TARGET_ID, ADDR_MIN_LIMIT, 0, length=2)
        # 最大リミットを 4095 に設定
        set_servo_register(ser, TARGET_ID, ADDR_MAX_LIMIT, 4095, length=2)
        
        print(f"✨ ID {TARGET_ID} のリミットを Min: 0, Max: 4095 に書き換えました！")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()