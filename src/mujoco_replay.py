import sys
import time
import os
import csv
import math
import mujoco
import mujoco.viewer

# ==========================================
# 1. ファイル設定・マッピング定義
# ==========================================
CSV_FILENAME = "arm_nobasu.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "motions", CSV_FILENAME)

SERVO_IDS = [1, 2, 3, 4, 5, 6]
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# teleop_sync.py と同一の可動範囲プロファイル[cite: 3]
JOINT_CONFIG = {
    1: {"type": "bounded", "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,  "f_min": 850,  "f_max": 3400, "f_cross": False, "init": 2048},
    2: {"type": "bounded", "r_min": 1715, "r_max": 4096 + 100,  "r_cross": True,  "f_min": 942,  "f_max": 3270, "f_cross": False, "init": 973},
    3: {"type": "bounded", "r_min": 900,  "r_max": 3100,        "r_cross": False, "f_min": 834,  "f_max": 3061, "f_cross": False, "init": 3061},
    4: {"type": "bounded", "r_min": 1650, "r_max": 4015,        "r_cross": False, "f_min": 735,  "f_max": 3214, "f_cross": False, "init": 735},
    5: {"type": "infinite", "init": 1023},
    6: {"type": "bounded", "r_min": 1990, "r_max": 3000,        "r_cross": False, "f_min": 1890, "f_max": 3152, "f_cross": False, "init": 1837},
}

# ==========================================
# 2. 角度計算ヘルパー
# ==========================================
def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    """リーダーRaw値からフォロワーのTarget値（0〜4095）を計算[cite: 3]"""
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
            
        return int(max(200, min(3900, target_linear)))

    elif config["type"] == "infinite":
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        diff = raw_leader - prev_raw
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096
            
        current_target = follower_current_cache.get(sid, config["init"])
        return (current_target + (diff * direction)) % 4096


def raw_to_radian(sid, target_val):
    """フォロワーTarget値（0〜4095）を中心基準のラジアン角（-π〜+π）に変換"""
    init_val = JOINT_CONFIG[sid].get("init", 2048)
    return (target_val - init_val) * (2.0 * math.pi / 4096.0)


# ==========================================
# 3. 簡易MJCFモデル定義（モデルファイル未指定時用）
# ==========================================
SIMPLE_ARM_XML = """
<mujoco model="so_arm100">
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    
    <!-- Base (Joint 1) -->
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.04 0.05" rgba="0.2 0.2 0.2 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.03 0.03 0.03" rgba="0.3 0.5 0.9 1"/>
        
        <!-- Shoulder (Joint 2) -->
        <body name="link2" pos="0 0 0.04">
          <joint name="joint2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02" rgba="0.9 0.4 0.3 1"/>
          
          <!-- Elbow (Joint 3) -->
          <body name="link3" pos="0 0 0.12">
            <joint name="joint3" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.018" rgba="0.3 0.8 0.4 1"/>
            
            <!-- Wrist Pitch (Joint 4) -->
            <body name="link4" pos="0 0 0.12">
              <joint name="joint4" type="hinge" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.015" rgba="0.8 0.7 0.2 1"/>
              
              <!-- Wrist Roll (Joint 5) -->
              <body name="link5" pos="0 0 0.05">
                <joint name="joint5" type="hinge" axis="0 0 1"/>
                <geom type="cylinder" size="0.015 0.015" rgba="0.6 0.3 0.8 1"/>
                
                <!-- Gripper (Joint 6) -->
                <body name="link6" pos="0 0 0.02">
                  <joint name="joint6" type="hinge" axis="1 0 0"/>
                  <geom type="box" size="0.01 0.03 0.01" rgba="0.9 0.2 0.2 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# ==========================================
# 4. 再生メイン処理（モデルロード部分の修正）
# ==========================================
# ==========================================
# 4. 再生メイン処理
# ==========================================
def main():
    # --- 1. CSVデータの存在確認とロード（※これが抜けていました） ---
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

    # --- 2. シーンモデルの読み込み ---
    xml_path = os.path.join(BASE_DIR, "assets", "so100_scene.xml")

    if os.path.exists(xml_path):
        print(f"🤖 SO-ARM100 シーンモデルを読み込み中: {xml_path}")
        model = mujoco.MjModel.from_xml_path(xml_path)
    else:
        print("⚠️ モデルが見つからないため、簡易モデルで起動します。")
        model = mujoco.MjModel.from_xml_string(SIMPLE_ARM_XML)

    data = mujoco.MjData(model)
    prev_leader_cache = {}
    follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}

    print("🚀 MuJoCo ビューアを起動します。Spaceキーでポーズ/再開、マウス操作で視点回転・ズームが可能です。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- 視認性向上のためのビジュアル設定 ---
        model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
        model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]

        playback_start = time.time()
        frame_idx = 0

        while viewer.is_running() and frame_idx < len(frames):
            t_target, raw_positions = frames[frame_idx]
            
            # CSVのタイムスタンプに合わせて同期
            elapsed = time.time() - playback_start
            if elapsed < t_target:
                time.sleep(0.001)
                continue

            # 各軸の目標値を計算してジョイント位置に設定
            for i, sid in enumerate(SERVO_IDS):
                raw_val = raw_positions[sid]
                target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                prev_leader_cache[sid] = raw_val
                follower_current_cache[sid] = target_val
                
                # ラジアン角に変換してMuJoCoジョイントに代入
                rad = raw_to_radian(sid, target_val)
                if i < model.nq:
                    data.qpos[i] = rad

            mujoco.mj_step(model, data)
            viewer.sync()

            sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {frames[-1][0]:6.2f}s [Frame {frame_idx + 1}/{len(frames)}]")
            sys.stdout.flush()

            frame_idx += 1

        print("\n✅ シミュレーション再生が完了しました。")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()