import sys
import time
import os
import csv
import math
import mujoco
import mujoco.viewer

# ==========================================
# 1. 設定・マッピング定義
# ==========================================
CSV_FILENAME = "modelmotion_mometohome.csv"  # 再生したいCSVファイル名

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "motions", CSV_FILENAME)

SERVO_IDS = [1, 2, 3, 4, 5, 6]
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# teleop_replay.py と完全一致させたフォロワー制御プロファイル
JOINT_CONFIG = {
    1: {"type": "bounded", "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,  "f_min": 850,  "f_max": 3400, "f_cross": False, "init": 2130},
    2: {"type": "bounded", "r_min": 1715, "r_max": 4096 + 100,  "r_cross": True,  "f_min": 942,  "f_max": 3270, "f_cross": False, "init": 973},
    3: {"type": "bounded", "r_min": 900,  "r_max": 3100,        "r_cross": False, "f_min": 834,  "f_max": 3061, "f_cross": False, "init": 3061},
    4: {"type": "bounded", "r_min": 1650, "r_max": 4015,        "r_cross": False, "f_min": 735,  "f_max": 3214, "f_cross": False, "init": 735},
    5: {"type": "infinite", "init": 1023},
    6: {"type": "bounded", "r_min": 1990, "r_max": 3000,        "r_cross": False, "f_min": 1890, "f_max": 3152, "f_cross": False, "init": 1837},
}

# ==========================================
# 2. MuJoCoモデル整合用パラメータ
# ==========================================
# MuJoCoのモデル（so100_scene.xml）の 0 rad 姿勢に対応するフォロワーのRawカウント値
# フォロワーの基準初期姿勢（Home）をそのままゼロ点としてマッピング
SIM_OFFSETS = {
    1: 2130,  # Base: 初期中心位置
    2: 973,   # Shoulder: 折りたたみ下限初期位置
    3: 3061,  # Elbow: 屈曲初期位置
    4: (735+4001)/2,   # Wrist Pitch: 初期位置
    5: 1023,  # Wrist Roll: 初期回転角
    6: 1837,  # Gripper: 閉初期位置
}

# MuJoCoの回転軸（Joint Axis）と実機サーボの増減方向の符号合わせ
SIM_DIRECTIONS = {
    1: -1.0,
    2:  1.0,
    3:  1.0,  # 肘関節（値増加で曲がる実機とMuJoCoの正方向の整合）
    4:  1.0,
    5:  1.0,
    6:  1.0,
}

# ==========================================
# 3. 角度計算ヘルパー（実機再現）
# ==========================================
def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    """teleop_replay.py と全く同じ目標値計算[cite: 7]"""
    config = JOINT_CONFIG[sid]
    direction = DIRECTION[sid]

    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        r_cross = config["r_cross"]
        f_min = config["f_min"]
        f_max = config["f_max"]
        f_cross = config["f_cross"]

        raw_l = raw_leader
        if r_cross and raw_l < (r_max - 4096):
            raw_l += 4096

        ratio = 0.0 if r_max == r_min else (raw_l - r_min) / (r_max - r_min)
        ratio = max(0.0, min(1.0, ratio))

        f_max_linear = f_max
        if f_cross and f_max < f_min:
            f_max_linear += 4096

        target_linear = f_min + ratio * (f_max_linear - f_min)
        if direction == -1:
            target_linear = f_min + (f_max_linear - target_linear)

        return int(max(0, min(4095, target_linear)))

    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096

        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)
        return int(new_target) % 4096


def raw_to_radian(sid, target_val):
    """フォロワーTarget目標値（0〜4095）をMuJoCo用のラジアン角に精密変換"""
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    
    # 4096 = 2π rad
    diff = (target_val - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


# ==========================================
# 4. 再生メイン処理
# ==========================================
def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSVファイルが見つかりません: {CSV_PATH}")
        return

    frames = []
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_sec"])
            positions = {sid: int(row[f"id_{sid}"]) for sid in SERVO_IDS}
            frames.append((t, positions))

    print(f"📖 ロード完了: {CSV_PATH} ({len(frames)} フレーム)")

    xml_path = os.path.join(BASE_DIR, "assets", "so100_scene.xml")
    if not os.path.exists(xml_path):
        print(f"❌ シーンモデルが見つかりません: {xml_path}")
        return

    print(f"🤖 SO-ARM100 シーンモデルを読み込み中: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 実機と同様のキャッシュ初期化
    prev_leader_cache = {}
    follower_current_cache = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

    print("🚀 MuJoCo ビューアを起動します。Spaceキーでポーズ/再開、マウスドラッグで視点変更が可能です。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        playback_start = time.time()
        frame_idx = 0

        while viewer.is_running() and frame_idx < len(frames):
            t_target, raw_positions = frames[frame_idx]

            # 実機と同じ実時間同期
            elapsed = time.time() - playback_start
            if elapsed < t_target:
                time.sleep(0.001)
                continue

            # 各軸の目標値を計算し、MuJoCoの関節へ流し込む
            for i, sid in enumerate(SERVO_IDS):
                raw_val = raw_positions[sid]
                target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                prev_leader_cache[sid] = raw_val
                follower_current_cache[sid] = target_val

                rad = raw_to_radian(sid, target_val)
                if i < model.nq:
                    data.qpos[i] = rad

            mujoco.mj_forward(model, data)
            viewer.sync()

            sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {frames[-1][0]:6.2f}s [Frame {frame_idx + 1}/{len(frames)}]")
            sys.stdout.flush()

            frame_idx += 1

        print("\n✅ シミュレーション再生が完了しました。")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()