"""
==============================================================================
机面正射影・座標変換モジュール (core/vision_projector.py)
==============================================================================
【役割】
1. カメラ画像から ArUco マーカー (ID 0〜3) の中心画素座標 (u, v) を抽出
2. 各マーカーの物理極座標 (r, theta) から机上直交座標 (X_fwd, Y_lat) [mm] を算出
3. ホモグラフィ行列 H を計算し、生画像を「真上視点の正射影画像」へ透視変換
4. 正射影画像上のピクセル座標 ⇄ アーム極座標 (r, theta) の相互変換関数を提供
==============================================================================
"""

import os
import json
import math
from typing import Dict, Tuple, Optional
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "markers_config.json")


class VisionProjector:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.marker_polar: Dict[int, Tuple[float, float]] = {}  # {id: (r_mm, theta_deg)}
        self.marker_phys_xy: Dict[int, np.ndarray] = {}         # {id: np.array([x_mm, y_mm])}
        self.homography_mat: Optional[np.ndarray] = None
        self.inv_homography_mat: Optional[np.ndarray] = None
        
        # 検出器初期化
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            self.detector = None

        self.load_config()

    def load_config(self):
        """設定ファイルからマーカーの極座標を読み込み、直交座標 (mm) を算出"""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            markers_data = data.get("markers", {})
        else:
            # デフォルト値フォールバック
            markers_data = {
                "0": {"r_cm": 60.0, "theta_deg": -20.0},
                "1": {"r_cm": 60.0, "theta_deg":  20.0},
                "2": {"r_cm": 27.0, "theta_deg": -60.0},
                "3": {"r_cm": 26.0, "theta_deg":  62.0},
            }

        self.marker_polar.clear()
        self.marker_phys_xy.clear()

        for str_id, val in markers_data.items():
            mid = int(str_id)
            r_mm = float(val["r_cm"]) * 10.0
            th_deg = float(val["theta_deg"])
            self.marker_polar[mid] = (r_mm, th_deg)

            # アーム基準直交座標: 前方 = X_fwd, 側方 = Y_lat (右: +, 左: -)
            th_rad = math.radians(th_deg)
            x_mm = r_mm * math.cos(th_rad)
            y_mm = r_mm * math.sin(th_rad)
            self.marker_phys_xy[mid] = np.array([x_mm, y_mm], dtype=np.float32)

    def set_marker_polar(self, marker_id: int, r_cm: float, theta_deg: float):
        """CLIや外部コードから動的に座標を上書き更新"""
        r_mm = r_cm * 10.0
        self.marker_polar[marker_id] = (r_mm, theta_deg)
        th_rad = math.radians(theta_deg)
        x_mm = r_mm * math.cos(th_rad)
        y_mm = r_mm * math.sin(th_rad)
        self.marker_phys_xy[marker_id] = np.array([x_mm, y_mm], dtype=np.float32)

    def detect_markers(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """画像中からマーカーを検出し、{ID: 中心ピクセル(u, v)} を取得"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        centers = {}
        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten()):
                c = corners[i][0]
                centers[int(mid)] = np.array([c[:, 0].mean(), c[:, 1].mean()], dtype=np.float32)
        return centers

    def update_homography(self, image: np.ndarray) -> bool:
        """4枚のマーカーからホモグラフィ行列 H を更新"""
        centers = self.detect_markers(image)
        # ID 0〜3 がすべて揃っているか確認
        if not all(mid in centers for mid in [0, 1, 2, 3]):
            return False

        # 画像上のピクセル座標 (src)
        src_pts = np.array([centers[0], centers[1], centers[3], centers[2]], dtype=np.float32)

        # 投影先の正射影キャンバス上でのピクセル配置 (dst)
        # 1辺 400mm の領域を 500x500 ピクセルの直交グリッドに配置
        dst_pts = np.array([
            [50.0, 50.0],    # ID 0 (奥・左)
            [450.0, 50.0],   # ID 1 (奥・右)
            [450.0, 450.0],  # ID 3 (手前・右)
            [50.0, 450.0],   # ID 2 (手前・左)
        ], dtype=np.float32)

        self.homography_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.inv_homography_mat = np.linalg.inv(self.homography_mat)
        return True

    def warp_to_topdown(self, image: np.ndarray, out_w: int = 500, out_h: int = 500) -> Optional[np.ndarray]:
        """生画像を真上視点の正射影画像に変換"""
        if self.homography_mat is None:
            if not self.update_homography(image):
                return None
        return cv2.warpPerspective(image, self.homography_mat, (out_w, out_h))