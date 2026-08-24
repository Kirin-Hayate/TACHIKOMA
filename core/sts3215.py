"""
==============================================================================
STS3215 シリアルサーボモータ通信ドライバ (core/sts3215.py)
==============================================================================
【役割】
Feetech STS3215 サーボモータと USB シリアル通信を行い、
「角度の読み取り」「目標角度の書き込み」「トルクON/OFF」を実行する共通クラスです。
==============================================================================
"""

import time
import serial


class STS3215Driver:
    def __init__(self, port, baudrate=1000000, timeout=0.01):
        """
        シリアルポートを開いて初期化する
        - port     : 'COM3', 'COM4' などのポート名
        - baudrate : 通信速度 (STS3215 の標準は 1Mbps = 1000000)
        - timeout  : タイムアウト時間 (秒)
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def read_position(self, servo_id):
        """
        指定したサーボ ID から現在の角度生値 (0〜4095) を読み取る
        - 戻り値: 正常時は 0〜4095 の整数値、失敗時は None
        """
        addr = 0x38     # 現在位置レジスタの先頭アドレス
        read_len = 2     # 読み取るバイト数 (位置は2バイト: Low, High)
        length = 4       # パケット長 (命令コード + アドレス + 読取長 + チェックサム = 4)
        
        # チェックサム計算 (プロトコルの仕様: ~(ID + Length + Instruction + Param...) & 0xFF)
        checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
        
        # 送信パケットの作成 (0xFF 0xFF はヘッダー, 0x02 は READ 命令)
        packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])
        
        self.ser.write(packet)
        time.sleep(0.001)
        response = self.ser.read(8)
        
        # 受信データの解析 (8バイト中、5番目と6番目が位置データ)
        if len(response) >= 7 and response[2] == servo_id:
            pos = response[5] | (response[6] << 8)
            if 0 <= pos <= 4095:
                return pos
        return None

    def write_position(self, servo_id, pos):
        """
        指定したサーボ ID へ目標角度 (0〜4095) を書き込む
        - pos: 指示したい位置の数値 (範囲外の値は 0〜4095 に自動クランプ)
        """
        # 安全のため 0〜4095 の範囲内に数値を丸める
        pos = int(max(0, min(4095, pos)))
        
        addr = 0x2A     # 目標位置レジスタの先頭アドレス
        length = 5       # パケット長
        pos_l = pos & 0xFF         # 下位8ビット
        pos_h = (pos >> 8) & 0xFF  # 上位8ビット
        
        # チェックサム計算 (0x03 は WRITE 命令)
        checksum = ~(servo_id + length + 0x03 + addr + pos_l + pos_h) & 0xFF
        packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, pos_l, pos_h, checksum])
        
        self.ser.write(packet)
        time.sleep(0.001)

    def set_torque(self, servo_id, enable):
        """
        サーボのトルクを ON / OFF する
        - enable: True (トルクON = モーター固定), False (トルクOFF = 手で回せる状態)
        """
        addr = 0x28     # トルクイネーブルレジスタのアドレス
        length = 4
        en_val = 1 if enable else 0
        
        checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
        packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
        
        self.ser.write(packet)
        time.sleep(0.002)

    def close(self):
        """シリアルポートを安全に閉じる"""
        if self.ser.is_open:
            self.ser.close()