"""
==============================================================================
ワークスペース到達可能性マッピングツール (tools/test_workspace_reachability.py)
==============================================================================
【目的】
1. 40cm x 40cm の机上面を 6x6 (計36点) のグリッドに分割。
2. 各座標において、ジェンガブロックの 4 種類の向き (0°, 45°, 90°, -45°) で
   core/kinematics.py の solve_ik_tabletop_grasp を実行。
3. 全 144 パターンの IK 可解性をテストし、成功率ヒートマップをコンソールに出力。
4. MuJoCo 3D 空間上にも到達可能点 (緑球) / 到達不能点 (赤球) をマーカー表示。

【操作】
  - 起動時に 36 点 x 4 姿勢の全自動テストを実行
  - [SPACE] : 再テスト実行
  - [Q/ESC] : 終了
==============================================================================
"""

import sys
import os
import math
import numpy as np
import mujoco
import mujoco.viewer
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.sim_viewer import MujocoSimViewer
from core.kinematics import (
    get_home_radians,
    solve_ik_tabletop_grasp,
    GRIPPER_OPEN_RAD
)

# ワークスペース範囲 [mm]
# ロボット台座からの奥行き X: 150mm 〜 350mm (スパン 200mm)
# 左右方向 Y: -150mm 〜 +150mm (スパン 300mm)
# 全体として約 40cm x 40cm の領域をカバー
X_RANGE = np.linspace(150.0, 380.0, 6)
Y_RANGE = np.linspace(-180.0, 180.0, 6)

TEST_ANGLES_DEG = [0.0, 45.0, 90.0, -45.0]

# ジェンガ標準寸法 (厚み/把持幅: 約 25mm, 把持中心高さ: 7.5mm)
JENGA_THICKNESS_MM = 25.0
JENGA_GRASP_Z_MM = 7.5


def run_reachability_benchmark():
    """36点 x 4姿勢の IK 可解性ベンチマークを実行 (リアルタイム進捗表示付き)"""
    print("\n" + "=" * 65)
    print(" 📊 ワークスペース到達可能性テスト (36点 x 4方向 = 144試行)")
    print("=" * 65)

    results = {}
    total_trials = len(X_RANGE) * len(Y_RANGE) * len(TEST_ANGLES_DEG)
    success_count = 0
    current_trial = 0
    start_time = time.time()

    # グリッド計算
    for r_idx, x_mm in enumerate(reversed(X_RANGE)):  # 奥 (X大) から手前 (X小)
        for c_idx, y_mm in enumerate(Y_RANGE):        # 左 (Y負) から右 (Y正)
            cell_results = []
            for angle in TEST_ANGLES_DEG:
                current_trial += 1
                elapsed = time.time() - start_time
                progress_pct = (current_trial / total_trials) * 100

                # リアルタイム進捗の1行上書き表示
                status_msg = (
                    f"\r⏳ [{current_trial:3d}/{total_trials}] ({progress_pct:4.1f}%) "
                    f"X={x_mm:5.1f}mm, Y={y_mm:+5.1f}mm, θ={angle:+4.0f}° 試行中... "
                    f"[成功: {success_count:3d}件 | 経過: {elapsed:4.1f}秒]"
                )
                sys.stdout.write(status_msg)
                sys.stdout.flush()

                # IK計算
                ik_grasp, ik_wp, pitch = solve_ik_tabletop_grasp(
                    x_phys_mm=x_mm,
                    y_phys_mm=y_mm,
                    z_phys_mm=JENGA_GRASP_Z_MM,
                    angle_deg=angle,
                    obj_thickness_mm=JENGA_THICKNESS_MM,
                    gripper_open_rad=GRIPPER_OPEN_RAD,
                    enable_sag_compensation=True
                )

                ok = (ik_grasp is not None and ik_wp is not None)
                if ok:
                    success_count += 1
                cell_results.append((angle, ok, pitch))

            results[(r_idx, c_idx)] = {
                "x_mm": x_mm,
                "y_mm": y_mm,
                "tests": cell_results
            }

    total_elapsed = time.time() - start_time
    # 進捗行をクリアして完了メッセージを表示
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()
    print(f"✅ 全 {total_trials} 件のテスト完了 (総所要時間: {total_elapsed:.2f}秒)")

    # --- 1. アスキーヒートマップの出力 ---
    print("\n【到達成功率マップ】(4方向中の成功数: 4=全方向可解, 0=到達不可)")
    print("      " + " ".join([f" Y={y:+4.0f}" for y in Y_RANGE]))
    print("    +" + "-------" * len(Y_RANGE) + "-+")

    for r_idx, x_mm in enumerate(reversed(X_RANGE)):
        row_str = f"X={x_mm:3.0f}|"
        for c_idx, y_mm in enumerate(Y_RANGE):
            tests = results[(r_idx, c_idx)]["tests"]
            oks = sum(1 for _, ok, _ in tests)

            if oks == 4:
                symbol = "  [4]  "  # 完全可解
            elif oks == 0:
                symbol = "   .   "  # 到達不能
            else:
                symbol = f"  ({oks})  "  # 一部角度のみ可解
            row_str += symbol
        row_str += "|"
        print(row_str)
    print("    +" + "-------" * len(Y_RANGE) + "-+")
    actual_success = sum(
        sum(1 for _, ok, _ in data["tests"])
        for data in results.values()
    )
    print(f"\n総成功率: {actual_success}/{total_trials} ({actual_success / total_trials * 100:.1f}%)")
    
    # --- 2. 不可解原因の傾向分析 ---
    unreachable_points = [v for v in results.values() if sum(1 for _, ok, _ in v["tests"]) == 0]
    partial_points = [v for v in results.values() if 0 < sum(1 for _, ok, _ in v["tests"]) < 4]

    print(f"・完全到達不能グリッド : {len(unreachable_points)} / 36 点")
    print(f"・角度依存で失敗する点 : {len(partial_points)} / 36 点")

    return results


def draw_reachability_in_mujoco(sim, results):
    """MuJoCo ビューアの机面上に結果球マーカーを描画"""
    if sim.viewer is None:
        return

    sim.viewer.user_scn.ngeom = 0

    for (r_idx, c_idx), data in results.items():
        if sim.viewer.user_scn.ngeom >= sim.viewer.user_scn.maxgeom:
            break

        x_mm = data["x_mm"]
        y_mm = data["y_mm"]
        oks = sum(1 for _, ok, _ in data["tests"])

        # MuJoCo 座標換算 (正面: -Y, 右: +X, 机面上: Z=5mm)
        mj_x = -y_mm / 1000.0
        mj_y = -x_mm / 1000.0
        mj_z = 0.005

        # 色分け: 全可解=緑, 一部可解=黄/橙, 全滅=赤
        if oks == 4:
            rgba = np.array([0.1, 0.9, 0.1, 0.8], dtype=np.float32)
            radius = 0.012
        elif oks > 0:
            rgba = np.array([0.9, 0.7, 0.0, 0.8], dtype=np.float32)
            radius = 0.009
        else:
            rgba = np.array([0.9, 0.1, 0.1, 0.4], dtype=np.float32)
            radius = 0.006

        ng = sim.viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            sim.viewer.user_scn.geoms[ng],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0, 0], dtype=np.float64),
            pos=np.array([mj_x, mj_y, mj_z], dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        sim.viewer.user_scn.ngeom += 1


def main():
    sim = MujocoSimViewer()
    home_rad = get_home_radians()
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass

    sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    # ベンチマーク実行
    results = run_reachability_benchmark()

    print("\nMuJoCo 空間上に可達性マーカーを表示しています。")
    print("  🟢 緑色: 4方向すべて把持可能")
    print("  🟡 黄色: 一部の向きのみ把持可能")
    print("  🔴 赤色: 到達不能 (解なし)")
    print("  [SPACE] で再計算、[ESC]/[Q] で終了")

    while sim.is_running():
        draw_reachability_in_mujoco(sim, results)
        mujoco.mj_forward(sim.model, sim.data)
        if sim.viewer is not None:
            sim.viewer.sync()


if __name__ == "__main__":
    main()