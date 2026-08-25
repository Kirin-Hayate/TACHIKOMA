"""
==============================================================================
円筒座標パラメータ駆動型 モーションジェネレータ (core/motion_generator.py)
==============================================================================
【役割】
基準テンプレートCSV (Pick & Place 動作) をベースに、
指定された旋回角度 (theta_pick, theta_place) に応じて
ID1 の軌道を動的に再計算したモーションフレーム配列を生成します。
==============================================================================
"""

import os
import sys
import csv
import math
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import SERVO_IDS, JOINT_CONFIG
from core.kinematics import calculate_target

# 境界フレーム定義（目次ログより）
FRAME_ROTATE_START = 2176  # 持ち上げ完了・旋回開始
FRAME_ROTATE_END   = 2316  # 旋回終了・下降開始
FRAME_PLACE_END    = 2999  # 離し完了・Home復帰開始


class ParametricMotionGenerator:
    def __init__(self, template_csv):
        resolved_path = self._resolve_path(template_csv)
        self.template_frames = self._load_template(resolved_path)

    def _resolve_path(self, filepath):
        if os.path.exists(filepath):
            return filepath
        candidate = os.path.join(BASE_DIR, "motions", os.path.basename(filepath))
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"❌ テンプレートCSVが見つかりません: {filepath}")

    def _load_template(self, filepath):
        """CSVからフレーム配列を読み込む"""
        frames = []
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row["timestamp_sec"])
                positions = {int(k.replace("id_", "")): int(v) for k, v in row.items() if k.startswith("id_")}
                frames.append((t, positions))
        return frames

    def generate(self, theta_pick_raw: int, theta_place_raw: int, theta_home_raw: int = 2048) -> list:
        """
        theta_pick_raw: 掴む位置のID1目標値 (フォロワー値: 0-4095)
        theta_place_raw: 置く位置のID1目標値 (フォロワー値: 0-4095)
        
        リーダー生値 ➔ フォロワー目標値への kinematics 変換を行い、
        ID1 のみ指定角度へ滑らかに差し替えたフレーム配列を返します。
        """
        home_positions = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}
        prev_leader_cache = {}
        follower_current_cache = dict(home_positions)

        new_frames = []
        total_frames = len(self.template_frames)

        for idx, (t, raw_pos) in enumerate(self.template_frames):
            # 1. まず全軸をリーダー生値 ➔ フォロワー目標値へ座標変換
            converted_pos = {}
            for sid in SERVO_IDS:
                raw_val = raw_pos[sid]
                val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                prev_leader_cache[sid] = raw_val
                follower_current_cache[sid] = val
                converted_pos[sid] = val

            # 2. ID1（旋回軸）の目標値を指定パラメータで動的に書き換え
            if idx < FRAME_ROTATE_START:
                # Phase 1: 掴み〜持ち上げまでは theta_pick
                converted_pos[1] = theta_pick_raw

            elif FRAME_ROTATE_START <= idx <= FRAME_ROTATE_END:
                # Phase 2: 旋回区間（コサインS字補間）
                ratio_linear = (idx - FRAME_ROTATE_START) / max(1, (FRAME_ROTATE_END - FRAME_ROTATE_START))
                s_ratio = (1.0 - math.cos(ratio_linear * math.pi)) / 2.0
                converted_pos[1] = int(theta_pick_raw + s_ratio * (theta_place_raw - theta_pick_raw))

            elif FRAME_ROTATE_END < idx <= FRAME_PLACE_END:
                # Phase 3: 下降〜離し完了までは theta_place
                converted_pos[1] = theta_place_raw

            else:
                # Phase 4: Home復帰区間
                ratio_linear = (idx - FRAME_PLACE_END) / max(1, (total_frames - 1 - FRAME_PLACE_END))
                s_ratio = (1.0 - math.cos(ratio_linear * math.pi)) / 2.0
                converted_pos[1] = int(theta_place_raw + s_ratio * (theta_home_raw - theta_place_raw))

            new_frames.append((t, converted_pos))

        return new_frames


# ==============================================================================
# 単体テスト ＆ 3Dシミュレータ動作確認
# ==============================================================================
if __name__ == "__main__":
    from core.sim_viewer import MujocoSimViewer
    from config.joint_config import JOINT_CONFIG
    import time

    template_file = "motion_20260825_222909.csv"
    generator = ParametricMotionGenerator(template_file)

    # 🎯 正面（中央）で掴んで、別の指定角度へ置くテスト
    pick_center = JOINT_CONFIG[1]["init"]  # 正面 (2048)
    place_left  = 2600                    # 左前方

    generated_frames = generator.generate(theta_pick_raw=pick_center, theta_place_raw=place_left)
    print(f"✅ モーション生成完了: 全 {len(generated_frames)} フレーム")

    print("\n🖥️ MuJoCo 3Dシミュレータでプレビュー再生を開始します...")
    try:
        sim = MujocoSimViewer()
        with sim.launch():
            start_time = time.time()
            for t_target, pos_dict in generated_frames:
                if not sim.is_running():
                    break
                while (time.time() - start_time) < t_target:
                    time.sleep(0.001)
                sim.update_joints(pos_dict)
        print("✅ プレビュー再生が終了しました。")
    except Exception as e:
        print(f"⚠️ プレビューエラー: {e}")