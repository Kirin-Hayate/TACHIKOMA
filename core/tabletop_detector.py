"""
==============================================================================
TACHIKOMA 机上物体検出・幾何計測モジュール (core/tabletop_detector.py)
==============================================================================
【モジュールの役割と構成】
本モジュールは、Webカメラの真上正射影画像（Top-down Warped Image）から、
机上に存在するすべての物理物体を高精度・30fps で検出し、物理座標系へと変換します。

1. マスク・背景差分管理レイヤー
   - create_marker_mask: 4隅の ArUco マーカー領域を除外する中央作業領域マスクの生成
   - update_background: 机面の背景画像を記憶し、木目や照明変動に強い差分画像を生成

2. 輪郭抽出・幾何計測レイヤー
   - detect_objects: 輪郭検出 (cv2.findContours) から最小外接回転矩形 (cv2.minAreaRect) を算出し、
     物体の重心ピクセル座標 (u, v)、物理座標 (X_mm, Y_mm)、長軸・短軸寸法 (mm)、
     把持角度 (angle_deg)、および VLM 送信用サムネイルクロップ画像を抽出

3. 画面描画・デバッグ重畳レイヤー
   - draw_annotations: 検出矩形、中心点、ミリ寸法、および同定ラベルを画像上に可視化描画
==============================================================================
"""

import math
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.vision_projector import VisionProjector


class TabletopDetector:
    def __init__(
        self,
        projector: VisionProjector,
        canvas_size: int = 500,
        margin: int = 50,
        inner_span_px: int = 400,
        min_area_px: int = 600,
        max_area_px: int = 30000,
    ):
        """
        引数:
            projector: 幾何射影および物理座標変換を担う VisionProjector インスタンス
            canvas_size: 正射影キャンバスの解像度 (デフォルト 500x500 px)
            margin: 作業外周マージン (デフォルト 50 px)
            inner_span_px: 400mm 相当のキャンバス内側スパン (デフォルト 400 px -> 1px ≈ 1mm)
            min_area_px: 検出対象とする最小ピクセル面積 (ノイズ除去)
            max_area_px: 検出対象とする最大ピクセル面積 (机面全体などの誤検出防止)
        """
        self.projector = projector
        self.canvas_size = canvas_size
        self.margin = margin
        self.inner_span_px = inner_span_px
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

        self.bg_gray: Optional[np.ndarray] = None
        self.valid_mask = self._create_marker_mask()

    def _create_marker_mask(self) -> np.ndarray:
        """4隅の ArUco マーカー領域を除外する中央矩形マスクを生成"""
        mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        pad = self.margin + 15
        mask[pad:self.canvas_size - pad, pad:self.canvas_size - pad] = 255
        return mask

    def update_background(self, warped_img: np.ndarray):
        """現在の机面を背景グレースケール画像として記憶"""
        cur_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        self.bg_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)
        print("📸 [TabletopDetector] 背景差分用画像を更新しました。")

    def pixel_to_robot_phys_xy(self, u: float, v: float) -> Tuple[float, float]:
        """正射影ピクセル (u, v) をロボット直交物理座標 (X_mm, Y_mm) にバイリニア射影変換"""
        p0 = self.projector.marker_phys_xy[0]
        p1 = self.projector.marker_phys_xy[1]
        p2 = self.projector.marker_phys_xy[2]
        p3 = self.projector.marker_phys_xy[3]

        s = (u - self.margin) / float(self.inner_span_px)
        t = (v - self.margin) / float(self.inner_span_px)

        top = (1.0 - s) * p0 + s * p1
        bottom = (1.0 - s) * p2 + s * p3
        phys_xy = (1.0 - t) * top + t * bottom
        return float(phys_xy[0]), float(phys_xy[1])

    def detect_objects(self, warped_img: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        正射影画像から物体を抽出し、物理情報・クロップ画像リストを返す。
        
        戻り値:
            (detected_objects, threshold_binary_image)
        """
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 背景画像がある場合は高精度背景差分、未登録時は適応的二値化
        if self.bg_gray is not None:
            diff = cv2.absdiff(blurred, self.bg_gray)
            _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        else:
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 6
            )

        thresh = cv2.bitwise_and(thresh, thresh, mask=self.valid_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        h_img, w_img = warped_img.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area_px <= area <= self.max_area_px):
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # 長軸・短軸の規格化 (常に w >= h とする)
            if w < h:
                w, h = h, w
                angle += 90.0

            # 角度を [-90°, +90°] の範囲に正規化
            while angle > 90.0:
                angle -= 180.0
            while angle <= -90.0:
                angle += 180.0

            box_pts = np.int32(cv2.boxPoints(rect))
            x_mm, y_mm = self.pixel_to_robot_phys_xy(cx, cy)

            # VLM用サムネイルのクロップ (AABB + パディング)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            pad = 10
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w_img, bx + bw + pad)
            y2 = min(h_img, by + bh + pad)
            crop = warped_img[y1:y2, x1:x2]

            detected.append({
                "u": cx,
                "v": cy,
                "phys_xy": (round(x_mm, 1), round(y_mm, 1)),
                "size_mm": (round(w, 1), round(h, 1)),
                "angle_deg": round(angle, 1),
                "box_pts": box_pts,
                "crop": crop
            })

        return detected, thresh

    def draw_annotations(
        self,
        base_img: np.ndarray,
        detected_objs: List[dict],
        object_profiles: Optional[Dict[int, dict]] = None,
        target_idx: Optional[int] = None
    ) -> np.ndarray:
        """検出矩形、ミリ寸法、VLM同定ラベルを画像上に重畳描画"""
        annotated = base_img.copy()
        if object_profiles is None:
            object_profiles = {}

        for i, obj in enumerate(detected_objs):
            is_target = (target_idx is not None and i == target_idx)
            box_color = (0, 0, 255) if is_target else (0, 255, 0)
            thickness = 3 if is_target else 2

            # 輪郭と重心
            cv2.drawContours(annotated, [obj["box_pts"]], 0, box_color, thickness)
            u_pt, v_pt = int(obj["u"]), int(obj["v"])
            cv2.circle(annotated, (u_pt, v_pt), 4, (0, 0, 255), -1)

            # ラベル名
            prof = object_profiles.get(i, {})
            display_name = prof.get("display_name", f"#{i}")
            if is_target:
                display_name += " [TARGET]"

            tag = f"#{i}: {display_name}"
            geom_text = f"{obj['size_mm'][0]:.0f}x{obj['size_mm'][1]:.0f}mm {obj['angle_deg']:+.0f}deg"

            # 縁取り付きテキスト
            cv2.putText(annotated, tag, (u_pt - 40, v_pt - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, tag, (u_pt - 40, v_pt - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if not is_target else (0, 165, 255), 1, cv2.LINE_AA)

            cv2.putText(annotated, geom_text, (u_pt - 40, v_pt - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, geom_text, (u_pt - 40, v_pt - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1, cv2.LINE_AA)

        return annotated