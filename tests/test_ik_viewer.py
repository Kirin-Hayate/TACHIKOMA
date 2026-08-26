"""
tests/test_ik_viewer.py (机面把持エリア検証版)
"""
import sys
import os
import time
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sim_viewer import MujocoSimViewer
from core.kinematics import solve_ik_polar


def main():
    viewer = MujocoSimViewer()
    with viewer.launch():
        print("=== MuJoCo 机面扇形スキャン検証開始 ===")
        print("手前・奥・左右へ机面すれすれをスキャンします。[Ctrl+C] または画面を閉じて終了。")

        t = 0.0
        while viewer.is_running():
            if viewer.paused:
                time.sleep(0.05)
                continue

            # 机面上の扇形領域テストパラメータ (指先基準)
            # 距離 r: 180mm 〜 260mm をゆっくり前後
            r = 0.22 + 0.04 * math.sin(t * 1.2)
            # 旋回 theta: -35度(左) 〜 +35度(右) をスイング
            theta = 35.0 * math.sin(t * 0.8)
            # 高さ z: 15mm 〜 40mm (机面すれすれをホバリング)
            z = 0.025 + 0.015 * math.cos(t * 1.5)

            # 解析的IK計算 (真下に近い -75度アプローチ)
            targets = solve_ik_polar(r=r, theta_deg=theta, z=z, pitch_deg=-75.0)

            if targets is not None:
                viewer.update_joints(targets)

            dt = 0.02 / viewer.playback_speed
            time.sleep(dt)
            t += 0.02


if __name__ == "__main__":
    main()