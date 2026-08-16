import sys
import time
import serial
import csv
import os

# ==========================================
# 1. 再生シーケンス・通信設定
# ==========================================
# 実行したいモーションファイルと再生回数をリストで指定
# [ ["ファイル名.csv", 再生回数], ... ]
MOTION_SEQUENCE = [
    ["motion_A.csv", 2],
    ["motion_B.csv", 1],
    ["motion_A.csv", 1]
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIONS_DIR = os.path.join(BASE_DIR, "motions")

# 安全動作の時間設定
HOME_RETURN_DURATION = 2.5         # 規定初期位置への復帰にかける秒数
MOTION_START_TRANSITION = 2.0      # モーション開始点への位置合わせにかける秒数

FOLLOWER_PORT = 'COM4'             # フォロワーアームのポート
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# ==========================================
# 2. リーダー・フォロワー可動範囲＆初期位置プロファイル
# ==========================================
JOINT_CONFIG = {
    1: {"type": "bounded", "r_min": 2850, "r_max": 4096 + 1400, "r_cross": True,  "f_min": 850,  "f_max": 3400, "f_cross": False, "init": 2130},
    2: {"type": "bounded", "r_min": 1715, "r_max": 4096 + 100,  "r_cross": True,  "f_min": 942,  "f_max": 3270, "f_cross": False, "init": 973},
    3: {"type": "bounded", "r_min": 900,  "r_max": 3100,        "r_cross": False, "f_min": 834,  "f_max": 3061, "f_cross": False, "init": 3061},
    4: {"type": "bounded", "r_min": 1650, "r_max": 4015,        "r_cross": False, "f_min": 735,  "f_max": 3214, "f_cross": False, "init": 735},
    5: {"type": "infinite", "init": 1023},
    6: {"type": "bounded", "r_min": 1990, "r_max": 3000,        "r_cross": False, "f_min": 1890, "f_max": 3152, "f_cross": False, "init": 1837},
}

# ==========================================
# 3. 通信・制御ヘルパー関数
# ==========================================
def read_servo_position(ser, servo_id):
    """フォロワーのサーボから現在位置の生データ（0〜4095）を取得（リトライ付き）"""
    addr = 0x38
    read_len = 2
    length = 4
    checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])

    for _ in range(3):
        ser.reset_input_buffer()
        ser.write(packet)
        time.sleep(0.002)
        response = ser.read(8)

        if len(response) >= 7 and response[2] == servo_id:
            pos = response[5] | (response[6] << 8)
            if 0 <= pos <= 4095:
                return pos
    return None


def write_follower_position(ser, servo_id, pos):
    """フォロワー側へ目標位置を書き込む"""
    pos = int(max(0, min(4095, pos)))
    addr = 0x2A
    length = 5
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    checksum = ~(servo_id + length + 0x03 + addr + pos_l + pos_h) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, pos_l, pos_h, checksum])
    ser.write(packet)
    time.sleep(0.001)


def set_torque(ser, servo_id, enable):
    """サーボのトルクON/OFF"""
    addr = 0x28
    length = 4
    en_val = 1 if enable else 0
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
    ser.write(packet)
    time.sleep(0.002)


def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    """リーダー生値からフォロワーのTarget位置を計算"""
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


def smooth_move_follower(ser, start_positions, target_positions, duration=2.0, steps=60):
    """
    start_positions から target_positions へ指定秒数（duration）かけて
    細かくステップ補間を行い、フォロワーを滑らかに移動させる
    """
    interval = duration / steps
    for step in range(1, steps + 1):
        ratio = step / steps
        for sid in SERVO_IDS:
            start_p = start_positions.get(sid, JOINT_CONFIG[sid]["init"])
            target_p = target_positions.get(sid, JOINT_CONFIG[sid]["init"])

            current_p = int(start_p + ratio * (target_p - start_p))
            write_follower_position(ser, sid, current_p)
        time.sleep(interval)


def load_motion_data(filepath):
    """CSVからモーションデータを読み込む"""
    if not os.path.exists(filepath):
        print(f"❌ ファイルが存在しません: {filepath}")
        return None

    frames = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_sec"])
            positions = {sid: int(row[f"id_{sid}"]) for sid in SERVO_IDS}
            frames.append((t, positions))
    return frames


# ==========================================
# 4. 再生メイン処理
# ==========================================
def main():
    if not MOTION_SEQUENCE:
        print("⚠️ 再生シーケンス（MOTION_SEQUENCE）が空です。")
        return

    # 事前に全ファイルの存在確認
    loaded_motions = []
    for filename, count in MOTION_SEQUENCE:
        filepath = os.path.join(MOTIONS_DIR, filename)
        frames = load_motion_data(filepath)
        if frames is None:
            print("🚨 読み込めないモーションファイルが存在するため、実行を中止します。")
            return
        loaded_motions.append({
            "filename": filename,
            "filepath": filepath,
            "repeat_count": count,
            "frames": frames
        })

    print(f"📋 登録されたシーケンス数: {len(loaded_motions)} 種類")
    for idx, item in enumerate(loaded_motions, start=1):
        print(f"  {idx}. {item['filename']} (再生回数: {item['repeat_count']}回 / {len(item['frames'])}フレーム)")

    try:
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)
        ser_follower.reset_input_buffer()

        # 規定の初期位置（Home）
        home_positions = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}

        # トルクON
        for sid in SERVO_IDS:
            set_torque(ser_follower, sid, True)

        # 起動直後の物理姿勢を確実に取得
        print("\n🔍 フォロワーの現在姿勢を確認中...")
        current_state = {}
        for sid in SERVO_IDS:
            pos = read_servo_position(ser_follower, sid)
            current_state[sid] = pos if pos is not None else home_positions[sid]

        # 最初の周回前に、現在位置 ➔ Home への復帰
        print(f"🏠 規定の初期位置（Home）へ復帰中 ({HOME_RETURN_DURATION}秒)...")
        smooth_move_follower(ser_follower, current_state, home_positions, duration=HOME_RETURN_DURATION)
        current_state = dict(home_positions)
        time.sleep(0.3)

        # 複数モーションのシーケンス実行
        total_types = len(loaded_motions)
        for seq_idx, motion_item in enumerate(loaded_motions, start=1):
            filename = motion_item["filename"]
            repeat_count = motion_item["repeat_count"]
            frames = motion_item["frames"]

            # このモーションの第1フレーム（開始姿勢）を算出
            first_raw_positions = frames[0][1]
            first_targets = {}
            temp_prev = {}
            temp_curr = dict(home_positions)
            for sid in SERVO_IDS:
                raw_val = first_raw_positions[sid]
                first_targets[sid] = calculate_target(sid, raw_val, temp_prev, temp_curr)

            print(f"\n##################################################")
            print(f" 🎬 [モーション {seq_idx}/{total_types}] : {filename}")
            print(f"##################################################")

            for loop_idx in range(repeat_count):
                print(f"\n--- 🔄 {filename} [{loop_idx + 1} / {repeat_count} 周目] ---")

                # 直前の動作（前回の周回または直前のモーション）からHomeへ復帰（現在位置がHomeでない場合）
                if current_state != home_positions:
                    print(f"🏠 規定の初期位置へ復帰中 ({HOME_RETURN_DURATION}秒)...")
                    smooth_move_follower(ser_follower, current_state, home_positions, duration=HOME_RETURN_DURATION)
                    current_state = dict(home_positions)
                    time.sleep(0.3)

                # Home ➔ モーション開始点へゆっくりアプローチ
                print(f"🎯 開始姿勢へアプローチ中 ({MOTION_START_TRANSITION}秒)...")
                smooth_move_follower(ser_follower, current_state, first_targets, duration=MOTION_START_TRANSITION)
                current_state = dict(first_targets)
                time.sleep(0.4)

                # モーション再生実行
                print(f"▶️ 再生実行中...")
                prev_leader_cache = {}
                follower_current_cache = dict(home_positions)
                playback_start_time = time.time()

                for t_target, raw_positions in frames:
                    while (time.time() - playback_start_time) < t_target:
                        time.sleep(0.001)

                    for sid in SERVO_IDS:
                        raw_val = raw_positions[sid]
                        target_val = calculate_target(sid, raw_val, prev_leader_cache, follower_current_cache)
                        write_follower_position(ser_follower, sid, target_val)

                        prev_leader_cache[sid] = raw_val
                        follower_current_cache[sid] = target_val
                        current_state[sid] = target_val

                    sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {frames[-1][0]:6.2f}s")
                    sys.stdout.flush()
                print("")

                # 各周回の再生完了後、一度Homeへ戻す
                print(f"🏠 モーション終了 ➔ Homeへ復帰中 ({HOME_RETURN_DURATION}秒)...")
                smooth_move_follower(ser_follower, current_state, home_positions, duration=HOME_RETURN_DURATION)
                current_state = dict(home_positions)
                time.sleep(0.3)

        print("\n✅ 全てのモーションシーケンスが安全に完了しました。")

    except KeyboardInterrupt:
        print("\n\n🛑 再生を中断しました。")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
    finally:
        if 'ser_follower' in locals() and ser_follower.is_open:
            for sid in SERVO_IDS:
                set_torque(ser_follower, sid, False)
            ser_follower.close()
            print("✅ フォロワーのトルクをOFFにし、ポートをクローズしました。")


if __name__ == "__main__":
    main()