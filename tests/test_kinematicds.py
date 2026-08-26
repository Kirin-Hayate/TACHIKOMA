"""
==============================================================================
逆運動学 (IK) 単体テスト & 到達可能エリア検証スクリプト (tests/test_kinematics.py)
==============================================================================
【役割】
扇形ワークスペース (r, theta, z) のグリッド点に対して solve_ik_polar を実行し、
到達可能領域、特異点、可動限界エラーを可視化・テストします。
==============================================================================
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.kinematics import solve_ik_polar


def test_ik_grid():
    # テスト対象のグリッド設定
    r_list = [0.15, 0.20, 0.25, 0.30]           # 距離 [m]
    theta_list = [-45.0, -30.0, 0.0, 30.0, 45.0] # 旋回角 [度]
    z_list = [0.03, 0.08, 0.15]                  # 高さ [m]
    pitch = -45.0                                 # ピッチ角 [度]

    total_points = len(r_list) * len(theta_list) * len(z_list)
    success_count = 0

    print(f"=== SO-ARM100 解析的IK グリッドテスト開始 (全 {total_points} 点) ===")

    for z in z_list:
        print(f"\n--- Height z = {z*1000:.0f} mm ---")
        for r in r_list:
            row_results = []
            for th in theta_list:
                res = solve_ik_polar(r=r, theta_deg=th, z=z, pitch_deg=pitch)
                if res is not None:
                    success_count += 1
                    row_results.append(f"O (θ={th:+3.0f}°)")
                else:
                    row_results.append(f"X (θ={th:+3.0f}°)")
            print(f"r = {r*1000:3.0f} mm | " + "  ".join(row_results))

    print("\n=======================================================")
    print(f"テスト結果: {success_count}/{total_points} 点が到達可能 (成功率: {success_count/total_points*100:.1f}%)")
    print("=======================================================")


if __name__ == "__main__":
    test_ik_grid()