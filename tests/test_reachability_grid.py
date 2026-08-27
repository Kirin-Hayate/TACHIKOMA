"""
==============================================================================
広域可動範囲・40点グリッド網羅性テスト (tests/test_reachability_grid.py)
==============================================================================
【役割】
手前限界 (140mm) 〜 伸長限界 (320mm) の扇形空間 40 点に対して、
IK の収束可否、サーボ限界、床面めり込み防止の判定結果を表形式で出力します。
==============================================================================
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.kinematics import solve_ik_polar


def run_reachability_test():
    # 距離 r: 手前(140mm) 〜 最遠(320mm)
    r_list = [0.14, 0.18, 0.22, 0.26, 0.30]  # 5段階
    # 旋回 theta: 左(-60°) 〜 右(+60°)
    theta_list = [-60.0, -30.0, 0.0, 30.0, 60.0]  # 5段階
    # 高さ z: 把持高さ(15mm) 〜 持ち上げ(120mm)
    z_list = [0.015, 0.050]  # 2段階 (5 x 5 x 2 = 40点テスト)

    print("=======================================================================")
    print(" 🚀 SO-ARM100 広域 40点 可動範囲・めり込み防止網羅テスト")
    print("=======================================================================")

    total = len(r_list) * len(theta_list) * len(z_list)
    success = 0

    for z in z_list:
        print(f"\n【高さ z = {z*1000:4.0f} mm (把持・スキャン面)】")
        header = "  距離 r \\ 旋回 θ | " + "  ".join([f"{th:+5.0f}°" for th in theta_list])
        print(header)
        print("-" * len(header))

        for r in r_list:
            row_str = f"  r = {r*1000:3.0f} mm     | "
            for th in theta_list:
                res = solve_ik_polar(r=r, theta_deg=th, z=z, prevent_penetration=True)
                if res is not None:
                    success += 1
                    row_str += "  [OK] "
                else:
                    row_str += "  [--] "
            print(row_str)

    print("\n=======================================================================")
    print(f"📊 検証結果: {success}/{total} 点 到達可能 (到達率: {success/total*100:.1f}%)")
    print("=======================================================================")


if __name__ == "__main__":
    run_reachability_test()