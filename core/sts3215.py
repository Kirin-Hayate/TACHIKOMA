"""
==============================================================================
STS3215 シリアルサーボモータ通信ドライバ (core/sts3215.py)
==============================================================================
【役割】
Feetech STS3215 サーボモータと USB シリアル通信を行い、
「角度の読み取り」「目標角度の書き込み」「トルクON/OFF」を実行する共通クラスです。

【安全機能】
- ソフトウェア・スルーレートリミッター（最大変化量制限）:
  1回（1ステップ）の送信あたりに変化して良い最大角度幅（max_step_limit）を制限します。
  プログラムのバグや急激な目標値変更があっても、モーターが最高速度で急旋回して破損するのを防ぎます。
==============================================================================
"""

import time
import serial


class STS3215Driver:
    def __init__(self, port, baudrate=1000000, timeout=0.01, max_step_limit=50):
        """
        シリアルポートを開いて初期化する
        - port           : 'COM3', 'COM4' などのポート名
        - baudrate       : 通信速度 (STS3215 の標準は 1Mbps = 1000000)
        - timeout        : タイムアウト時間 (秒)
        - max_step_limit : 1回の送信あたりの最大変位量 (0〜4095 scale)
                           50Hz制御時、50 count/step ≒ 約220 deg/sec の最高速度制限。
                           None または 0 を渡すとリミッター無効。
        """
        self.port = port
        self.baudrate = baudrate
        self.max_step_limit = max_step_limit
        
        # 直前にサーボへ送信した（または読み取った）位置を ID ごとに記録する辞書
        self.last_positions = {}

        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def read_position(self, servo_id, retries=3):
        """
        指定したサーボ ID から現在の角度生値 (0〜4095) を読み取る (リトライ・バッファクリア対応)
        """
        addr = 0x38     # 現在位置レジスタの先頭アドレス
        read_len = 2     # 読み取るバイト数
        length = 4       # パケット長
        
        checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
        packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])

        for _ in range(retries):
            # 送信前に受信バッファに残っているゴミを必ず破棄
            self.ser.reset_input_buffer()
            self.ser.write(packet)
            
            # サーボ側の応答生成待ち (2ms〜3ms)
            time.sleep(0.003)
            response = self.ser.read(8)

            if len(response) >= 7 and response[0] == 0xFF and response[1] == 0xFF and response[2] == servo_id:
                pos = response[5] | (response[6] << 8)
                if 0 <= pos <= 4095:
                    self.last_positions[servo_id] = pos
                    return pos
            time.sleep(0.005)

        return None
    def write_position(self, servo_id, pos):
        """
        指定したサーボ ID へ目標角度 (0〜4095) を書き込む
        ★ スルーレートリミッターにより、急激な変位量を自動制限します。
        - pos: 指示したい位置の数値 (範囲外の値は 0〜4095 に自動クランプ)
        """
        # 安全のため 0〜4095 の範囲内に数値を丸める
        target_pos = int(max(0, min(4095, pos)))

        # --- 🛡️ スルーレートリミッター（最大変化量制限） ---
        if self.max_step_limit is not None and self.max_step_limit > 0:
            if servo_id in self.last_positions:
                prev_pos = self.last_positions[servo_id]
                diff = target_pos - prev_pos

                # 変化量が制限値を超えている場合は最大変化量に抑え込む
                if abs(diff) > self.max_step_limit:
                    target_pos = prev_pos + (self.max_step_limit if diff > 0 else -self.max_step_limit)

        # 送信した値を次回のために記録
        self.last_positions[servo_id] = target_pos

        addr = 0x2A     # 目標位置レジスタの先頭アドレス
        length = 5       # パケット長
        pos_l = target_pos & 0xFF         # 下位8ビット
        pos_h = (target_pos >> 8) & 0xFF  # 上位8ビット
        
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