import time
import serial

PORT = "COM3"
BAUDRATE = 1000000

def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        print(f"✅ Leader ({PORT}) を開きました。手で動かしてみてください。([Ctrl+C]で終了)")
        
        while True:
            positions = []
            for servo_id in range(1, 7):
                addr = 0x38
                read_len = 2
                length = 4
                checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
                packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])
                
                ser.write(packet)
                time.sleep(0.002)
                
                response = ser.read(8)
                if len(response) >= 7 and response[2] == servo_id:
                    pos = response[5] | (response[6] << 8)
                    positions.append(f"ID{servo_id}:{pos:4d}")
                else:
                    positions.append(f"ID{servo_id}: N/A")
            
            print("\r" + " | ".join(positions), end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n終了しました。")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()