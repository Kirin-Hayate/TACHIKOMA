import time
import serial

PORT = 'COM4'
BAUDRATE = 1000000

def read_servo_positions():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        print(f"✅ {PORT} を開きました。アームの手先や関節を手で軽く動かしてみてください。")
        print("停止するには [Ctrl + C] を押してください。\n")
        
        # STS3215等のシリアルバスサーボの現在位置読み取りコマンド（Ping/Read）
        # 各関節(ID 1〜6)に対して位置取得をリクエスト
        while True:
            positions = []
            for servo_id in range(1, 7):
                # STS3215 位置読み取りパケット構造 (ID, Length, Instruction, Address, Read_Len, Checksum)
                # 位置アドレス 0x38 (56), 長さ 2バイト
                addr = 0x38
                read_len = 2
                length = 4
                checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
                packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])
                
                ser.write(packet)
                time.sleep(0.002) # 微小ウェイト
                
                response = ser.read(8)
                if len(response) >= 7 and response[2] == servo_id:
                    # 2バイトの位置データを数値に変換 (0〜4095)
                    pos = response[5] | (response[6] << 8)
                    positions.append(f"ID{servo_id}:{pos:4d}")
                else:
                    positions.append(f"ID{servo_id}: N/A")
            
            print("\r" + " | ".join(positions), end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n🛑 読み取りを停止しました。")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    read_servo_positions()