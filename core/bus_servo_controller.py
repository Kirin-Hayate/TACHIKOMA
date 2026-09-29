"""
==============================================================================
Feetech STS3215 バスサーボ一括制御マネージャー
(core/bus_servo_controller.py)
==============================================================================
"""

import time
import serial.tools.list_ports
from typing import Dict, Optional, List

from core.sts3215 import STS3215Driver


class BusServoController:
    def __init__(self, port: Optional[str] = None, baudrate: int = 1000000):
        self.port = port
        self.baudrate = baudrate
        self.driver: Optional[STS3215Driver] = None
        self.is_open = False
        self.servo_ids: List[int] = [1, 2, 3, 4, 5, 6]

    def auto_detect_port(self) -> Optional[str]:
        """利用可能な USB シリアルポート (CH340 / CP210x / FTDI 等) を自動検出"""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = p.description.lower()
            if any(kw in desc for kw in ["ch340", "cp210", "ftdi", "serial", "usb"]):
                return p.device
        if ports:
            return ports[0].device
        return None

    def connect(self) -> bool:
        if self.port is None:
            self.port = self.auto_detect_port()

        if self.port is None:
            print("❌ エラー: 接続可能なシリアルポートが見つかりません。")
            return False

        try:
            # 50 count/step (約 220 deg/s) のスルーレート制限で安全初期化
            self.driver = STS3215Driver(self.port, baudrate=self.baudrate, max_step_limit=60)
            self.is_open = True
            print(f"✅ STS3215 バスサーボ接続成功 ({self.port}, {self.baudrate} bps)")

            # 全軸の現在位置を一度読み取って初期同期
            for sid in self.servo_ids:
                cur_pos = self.driver.read_position(sid)
                if cur_pos is not None:
                    self.driver.last_positions[sid] = cur_pos
            return True
        except Exception as e:
            print(f"❌ シリアルポート接続失敗 ({self.port}): {e}")
            self.is_open = False
            return False

    def move_servos(self, targets_raw: Dict[int, int], move_time_ms: int = 1000):
        """
        指定した ID と生目標値 (0〜4095) へサーボを駆動
        """
        if not self.is_open or self.driver is None:
            return

        for sid, pos in targets_raw.items():
            if sid in self.servo_ids:
                self.driver.write_position(sid, pos)

    def set_all_torque(self, enable: bool):
        """全サーボのトルクを一括 ON / OFF"""
        if not self.is_open or self.driver is None:
            return
        for sid in self.servo_ids:
            self.driver.set_torque(sid, enable)

    def close(self):
        if self.driver is not None:
            self.driver.close()
        self.is_open = False