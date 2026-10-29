"""
==============================================================================
正方形作業空間幾何ソルバー (core/square_workspace_solver.py)
==============================================================================
【役割】
手前の2点 (ID 2: 手前左, ID 3: 手前右) の実測直交座標 [X, Y] mm と、
既知の正方形一辺長 L [mm] から、アームの手が届かない奥の2点 (ID 0, ID 1) の
位置を幾何拘束 (直交・等長ベクトル) によって一意に復元し、極座標系 (r, theta) を算出します。

【座標系の定義】
  - アーム台座旋回中心を原点 (0, 0)
  - X軸: アームの前方方向 (正) [mm]
  - Y軸: アームの側方方向 (右が正、左が負) [mm]
  - 極座標:
      r = sqrt(X^2 + Y^2) [cm]
      theta = atan2(Y, X) [deg]
==============================================================================
"""

import math
import json
import os
from typing import Dict, Tuple, Optional
import numpy as np


class SquareWorkspaceSolver:
    def __init__(self, side_length_mm: float = 400.0, tolerance_ratio: float = 0.10):
        """
        Args:
            side_length_mm: 正方形の公称一辺長 [mm] (デフォルト: 400mm)
            tolerance_ratio: 実測した手前2点間距離と公称値の許容誤差率 (デフォルト: ±10%)
        """
        self.L = float(side_length_mm)
        self.tolerance_ratio = tolerance_ratio

    def solve_from_front_points(
        self,
        p2_xy: Tuple[float, float],
        p3_xy: Tuple[float, float],
        enforce_nominal_length: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """
        手前2点 (ID 2, ID 3) から奥2点 (ID 0, ID 1) を計算し、全4マーカーの極座標辞書を返す。

        Args:
            p2_xy: ID 2 (手前左) の直交座標 (x_mm, y_mm)
            p3_xy: ID 3 (手前右) の直交座標 (x_mm, y_mm)
            enforce_nominal_length: Trueの場合、奥方向の辺長として公称値 L を使用。
                                   Falseの場合、実測底辺長を使用。

        Returns:
            Dict[int, Dict[str, float]]: 各マーカーの {r_cm, theta_deg, x_mm, y_mm}
        """
        p2 = np.array([float(p2_xy[0]), float(p2_xy[1])], dtype=np.float64)
        p3 = np.array([float(p3_xy[0]), float(p3_xy[1])], dtype=np.float64)

        # 底辺ベクトル (ID 2 -> ID 3: 左から右へ向かうベクトル)
        v_base = p3 - p2
        measured_width = float(np.linalg.norm(v_base))

        if measured_width < 1e-3:
            raise ValueError("ID 2 と ID 3 の座標が近すぎるため、底辺を定義できません。")

        # 実測距離のバリデーションチェック
        diff_ratio = abs(measured_width - self.L) / self.L
        if diff_ratio > self.tolerance_ratio:
            print(f"⚠️ [警告] 手前2点間の実測距離 ({measured_width:.1f} mm) が公称値 ({self.L:.1f} mm) と {diff_ratio*100:.1f}% 乖離しています。")

        # 底辺の単位ベクトル u_base (右方向)
        u_base = v_base / measured_width

        # 直交する奥方向の単位ベクトル n_fwd (2次元平面での左回り90度回転: [ux, uy] -> [-uy, ux] または [uy, -ux])
        # Xが前方、Yが右の場合:
        # 左(p2)から右(p3)へ向かうベクトルの「前進方向 (奥)」は、進行方向に対して左手側。
        # したがって: n_fwd = np.array([u_base[1], -u_base[0]]) またはアーム前方 (+X方向) を向くよう符号選択
        n_candidate1 = np.array([-u_base[1], u_base[0]])
        n_candidate2 = np.array([u_base[1], -u_base[0]])

        # 奥方向は前方 (+X) を向いているはずなので、X成分が大きい候補を採用
        n_fwd = n_candidate1 if n_candidate1[0] > n_candidate2[0] else n_candidate2

        depth_len = self.L if enforce_nominal_length else measured_width

        # 奥の2点を算出
        # ID 0: ID 2 から奥方向へ進んだ地点 (奥左)
        # ID 1: ID 3 から奥方向へ進んだ地点 (奥右)
        p0 = p2 + depth_len * n_fwd
        p1 = p3 + depth_len * n_fwd

        points_xy = {
            0: p0,
            1: p1,
            2: p2,
            3: p3
        }

        descriptions = {
            0: "奥・左",
            1: "奥・右",
            2: "手前・左",
            3: "手前・右"
        }

        result = {}
        for mid, pt in points_xy.items():
            x_mm, y_mm = float(pt[0]), float(pt[1])
            r_mm = math.sqrt(x_mm**2 + y_mm**2)
            th_deg = math.degrees(math.atan2(y_mm, x_mm))
            r_cm = r_mm / 10.0

            result[mid] = {
                "r_cm": round(r_cm, 2),
                "theta_deg": round(th_deg, 2),
                "x_mm": round(x_mm, 2),
                "y_mm": round(y_mm, 2),
                "description": descriptions[mid]
            }

        return result

    def export_to_config_json(
        self,
        calibrated_result: Dict[int, Dict[str, float]],
        output_path: str
    ) -> bool:
        """算出したマーカー極座標を設定ファイル (markers_config.json) に保存"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 既存のwarp_settings等を維持しつつmarkersのみ上書き
            existing_data = {}
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                    except Exception:
                        existing_data = {}

            markers_dict = {}
            for mid in [0, 1, 2, 3]:
                if mid in calibrated_result:
                    markers_dict[str(mid)] = {
                        "r_cm": calibrated_result[mid]["r_cm"],
                        "theta_deg": calibrated_result[mid]["theta_deg"],
                        "description": calibrated_result[mid]["description"]
                    }

            existing_data["markers"] = markers_dict
            if "warp_settings" not in existing_data:
                existing_data["warp_settings"] = {
                    "output_width_px": 500,
                    "output_height_px": 500,
                    "pixels_per_mm": 1.0
                }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"❌ 設定ファイルの保存に失敗しました: {e}")
            return False


# --- 単体動作テストコード ---
if __name__ == "__main__":
    print("=== SquareWorkspaceSolver 幾何計算テスト ===")

    # テスト入力: 現在実測されている手前2点 (ID2, ID3)
    # ID2: r = 27.0cm, theta = -60.0 deg -> X=135.0mm, Y=-233.8mm
    # ID3: r = 26.0cm, theta = +62.0 deg -> X=122.1mm, Y=+229.6mm
    th2_rad = math.radians(-60.0)
    p2_test = (270.0 * math.cos(th2_rad), 270.0 * math.sin(th2_rad))

    th3_rad = math.radians(62.0)
    p3_test = (260.0 * math.cos(th3_rad), 260.0 * math.sin(th3_rad))

    solver = SquareWorkspaceSolver(side_length_mm=400.0)
    solved = solver.solve_from_front_points(p2_test, p3_test)

    for mid in sorted(solved.keys()):
        d = solved[mid]
        print(f"ID {mid} ({d['description']}): r={d['r_cm']}cm, θ={d['theta_deg']}° (X={d['x_mm']}mm, Y={d['y_mm']}mm)")