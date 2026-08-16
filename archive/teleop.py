import time
import serial

# COMポートの設定
LEADER_PORT = 'COM3'   # 黒（操作用）
FOLLOWER_PORT = 'COM4' # 白（作業用）
BAUDRATE = 1000000

def sync_arm_positions():
    try:
        # 両方のポートをオープン
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.05)
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.05)
        
        print(f"✅ テレオペレーションを開始します！")
        print(f"  🎮 リーダー (黒): {LEADER_PORT}")
        print(f"  🤖 フォロワー(白): {FOLLOWER_PORT}")
        print("黒アームを手で動かすと、白アームが追従します。")
        print("停止するには [Ctrl + C] を押してください。\n")
        
        while True:
            for servo_id in range(1, 7):
                # --- 1. リーダー（黒）から現在の位置を読み取る ---
                addr = 0x38
                read_len = 2
                length = 4
                checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
                packet_read = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])
                
                ser_leader.write(packet_read)
                time.sleep(0.001)
                
                resp = ser_leader.read(8)
                if len(resp) >= 7 and resp[2] == servo_id:
                    pos = resp[5] | (resp[6] << 8)
                    
                    # --- 2. 読み取った位置をフォロワー（白）のサーボへ書き込む ---
                    # Write Instruction (0x03), アドレス 0x2A (Goal Position)
                    pos_low = pos & 0xFF
                    pos_high = (pos >> 8) & 0xFF
                    write_len = 7
                    # パケット: ID, Length, Instruction(0x03), Addr(0x2A), Pos_L, Pos_H, Speed_L(0), Speed_H(0)
                    w_checksum = ~(servo_id + write_len + 0x03 + 0x2A + pos_low + pos_high + 0 + 0) & 0xFF
                    packet_write = bytes([0xFF, 0xFF, servo_id, write_len, 0x03, 0x2A, pos_low, pos_high, 0x00, 0x00, w_checksum])
                    
                    ser_follower.write(packet_write)
                    time.sleep(0.001)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n🛑 テレオペレーションを停止しました。")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
    finally:
        if 'ser_leader' in locals() and ser_leader.is_open:
            ser_leader.close()
        if 'ser_follower' in locals() and ser_follower.is_open:
            ser_follower.close()

if __name__ == "__main__":
    sync_arm_positions()