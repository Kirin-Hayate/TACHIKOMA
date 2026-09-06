"""
==============================================================================
円筒座標パラメータ駆動型 モーションジェネレータ (ラジアン統一版)
(core/motion_generator.py)
==============================================================================
【役割】
基準テンプレート CSV (Pick & Place 動作: ラジアン形式) をベースに、
指定された旋回角度 (theta_pick_rad, theta_place_rad) に応じて
ID 1 (台座旋回) の軌道を動的に再計算したモーションフレーム配列を生成します。

【処理の流れ】
1. ラジアン CSV (q1〜q6) からフレーム配列をロード。
2. 掴み〜持ち上げ区間 (Phase 1): ID 1 を theta_pick_rad で固定。
3. 旋回区間 (Phase 2): コサイン S 字補間で theta_pick_rad から theta_place_rad へ補間。
4. 下降〜離し区間 (Phase 3): ID 1 を theta_place_rad で固定。
5. 復帰区間 (Phase 4): コサイン S 字補間で theta_place_rad から Home 旋回角 (0 rad) へ復帰。
==============================================================================
"""

import os
import sys
import csv
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import SERVO_IDS
from core.kinematics import raw_to_radian

# 境界フレーム定義 (rad_tuned.csv のステップ構成に準拠)
# フェーズ a: 0〜249, フェーズ b: 250〜439, フェーズ c: 440〜589
FRAME_ROTATE_START = 310  # 持ち上げ完了・旋回開始
FRAME_ROTATE_END   = 380  # 旋回終了・下降開始
FRAME_PLACE_END    = 440  # 離し完了・Home復帰開始


class ParametricMotionGenerator:
    def __init__(self, template_csv=None):
        if template_csv is None:
            template_csv = "rad_tuned.csv"
        resolved_path = self._resolve_path(template_csv)
        self.template_frames = self._load_template(resolved_path)

    def _resolve_path(self, filepath):
        # 1. 指定されたパスそのまま
        if os.path.exists(filepath):
            return filepath
        # 2. motions/ フォルダ配下
        candidate1 = os.path.join(BASE_DIR, "motions", os.path.basename(filepath))
        if os.path.exists(candidate1):
            return candidate1
        # 3. プロジェクトルートからの相対パス
        candidate2 = os.path.join(BASE_DIR, filepath)
        if os.path.exists(candidate2):
            return candidate2

        raise FileNotFoundError(
            f"❌ テンプレートCSVが見つかりません: {filepath}\n"
            f"   'python tools/generate_motion_csv.py --output motions/{os.path.basename(filepath)}' を実行して生成してください。"
        )

    def _load_template(self, filepath):
        """CSV からラジアンフレーム配列を読み込む (旧 Raw 形式も自動対応)"""
        frames = []
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            is_radian = "q1" in headers

            for row in reader:
                t = float(row["timestamp_sec"])
                positions = {}
                for sid in SERVO_IDS:
                    if is_radian:
                        positions[sid] = float(row[f"q{sid}"])
                    else:
                        raw_val = int(row[f"id_{sid}"])
                        positions[sid] = raw_to_radian(sid, raw_val)
                frames.append((t, positions))
        return frames

    def generate(self, theta_pick_rad: float, theta_place_rad: float, theta_home_rad: float = 0.0) -> list:
        """
        theta_pick_rad: 掴む位置の台座旋回角 [rad] (正面 0, 右 +, 左 -)
        theta_place_rad: 置く位置の台座旋回角 [rad]
        theta_home_rad: 初期姿勢の旋回角 (通常 0.0 rad)

        ID 1 以外の関節角度はテンプレートの滑らかな昇降・把持軌道を維持し、
        ID 1 のみ指定された角度へ S 字補間で差し替えたフレーム配列を返します。
        """
        new_frames = []
        total_frames = len(self.template_frames)

        # 動的境界調整 (テンプレート長が異なる場合の安全クリップ)
        f_rot_start = min(FRAME_ROTATE_START, int(total_frames * 0.50))
        f_rot_end   = min(FRAME_ROTATE_END, int(total_frames * 0.65))
        f_place_end = min(FRAME_PLACE_END, int(total_frames * 0.75))

        for idx, (t, rad_pos) in enumerate(self.template_frames):
            frame_pos = dict(rad_pos)

            # ID 1（旋回軸）の目標値を指定パラメータで動的に書き換え
            if idx < f_rot_start:
                # Phase 1: 掴み〜持ち上げまでは theta_pick
                frame_pos[1] = theta_pick_rad

            elif f_rot_start <= idx <= f_rot_end:
                # Phase 2: 旋回区間（コサイン S 字補間）
                ratio_linear = (idx - f_rot_start) / max(1, (f_rot_end - f_rot_start))
                s_ratio = (1.0 - math.cos(ratio_linear * math.pi)) / 2.0
                frame_pos[1] = theta_pick_rad + s_ratio * (theta_place_rad - theta_pick_rad)

            elif f_rot_end < idx <= f_place_end:
                # Phase 3: 下降〜離し完了までは theta_place
                frame_pos[1] = theta_place_rad

            else:
                # Phase 4: Home 復帰区間
                ratio_linear = (idx - f_place_end) / max(1, (total_frames - 1 - f_place_end))
                s_ratio = (1.0 - math.cos(ratio_linear * math.pi)) / 2.0
                frame_pos[1] = theta_place_rad + s_ratio * (theta_home_rad - theta_place_rad)

            new_frames.append((t, frame_pos))

        return new_frames