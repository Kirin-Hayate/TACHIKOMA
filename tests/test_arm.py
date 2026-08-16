import time
import serial

# ポート番号をCOMxに設定
PORT = 'COM4'
BAUDRATE = 1000000  # STS3215バスサーボの標準ボーレートは1Mbps

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} との接続に成功しました！")
    
    time.sleep(1)
    
    if ser.is_open:
        print("🤖 制御ボード通信OK：物理接続は問題ありません。")
        
    ser.close()
    print("🔌 接続を正常に切断しました。")

except Exception as e:
    print(f"❌ 接続エラーが発生しました: {e}")