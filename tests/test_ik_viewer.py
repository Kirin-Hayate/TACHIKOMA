"""
tests/test_ik_viewer.py (リアルタイム数値モニタ付き)
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
        print("=== MuJoCo 机面扇形スキャン検証 ===")
        print("目標座標と計算結果をリアルタイム表示します。[Ctrl+C] で終了。\n")

        t = 0.0
        frame = 0

        while viewer.is_running():
            if viewer.paused:
                time.sleep(0.05)
                continue

            # 爪先端基準の目標パラメータ
            # 距離 r: 180mm 〜 250mm
            r = 0.215 + 0.035 * math.sin(t * 1.0)
            # 旋回 theta: -40度(左) 〜 +40度(右)
            theta = 40.0 * math.sin(t * 0.7)
            # 高さ z: 10mm 〜 40mm (机面上 1cm〜4cm のホバリング)
            z = 0.025 + 0.015 * math.cos(t * 1.3)

            targets = solve_ik_polar(r=r, theta_deg=theta, z=z)

            status = "✅ 追従中" if targets is not None else "⚠️ 範囲外/未収束"
            
            # ターミナルにリアルタイム表示
            out_str = f"\r[{frame:04d}] Target: r={r*1000:5.1f}mm, θ={theta:+5.1f}°, z={z*1000:4.1f}mm | IK: {status}"
            if targets:
                out_str += f" | J1:{targets[1]} J2:{targets[2]} J3:{targets[3]} J4:{targets[4]}"
            sys.stdout.write(out_str)
            sys.stdout.flush()

            if targets is not None:
                viewer.update_joints(targets)

            dt = 0.02 / viewer.playback_speed
            time.sleep(dt)
            t += 0.02
            frame += 1

        print("\n")


if __name__ == "__main__":
    main()