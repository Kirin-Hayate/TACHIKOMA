"""
==============================================================================
TACHIKOMA 自然言語対話エージェント (src/tachikoma_agent_1.py)
==============================================================================
【役割】
1. ターミナルから自然言語で指示を入力（例: 「AからBに運んで、そのあとCに持っていって！」）。
2. LLM (Gemini) が motions.json を参照して実行計画 (JSON) を自動生成。
3. まず 3Dシミュレータ (MuJoCo) で動作セットをプレビュー再生。
4. ユーザーが画面・計画を確認して 'y' を入力すると、実機フォロワーで安全に動作を実行。
==============================================================================
"""

import sys
import time
import csv
import os
import math
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import (
    FOLLOWER_PORT,
    BAUDRATE,
    SERVO_IDS,
    JOINT_CONFIG
)
from core.sts3215 import STS3215Driver
from core.kinematics import calculate_target
from core.sim_viewer import MujocoSimViewer
from core.llm_planner import LLMTaskPlanner

# 安全動作パラメータ
HOME_RETURN_DURATION = 2.5
MOTION_START_TRANSITION = 2.0


def smooth_move(follower_driver, sim_viewer, target_positions, fallback_state=None, duration=2.5, steps=75):
    """コサインS字加減速による安全補間移動"""
    start_positions = {}
    for sid in SERVO_IDS:
        if follower_driver is not None:
            p = follower_driver.read_position(sid)
            start_positions[sid] = p if p is not None else (fallback_state.get(sid, JOINT_CONFIG[sid]["init"]) if fallback_state else JOINT_CONFIG[sid]["init"])
        else:
            start_positions[sid] = fallback_state.get(sid, JOINT_CONFIG[sid]["init"]) if fallback_state else JOINT_CONFIG[sid]["init"]

    interval = duration / steps
    for step in range(1, steps + 1):
        t = step / steps
        ratio = (1.0 - math.cos(t * math.pi)) / 2.0

        current_step_positions = {}
        for sid in SERVO_IDS:
            start_p = start_positions[sid]
            target_p = target_positions.get(sid, JOINT_CONFIG[sid]["init"])
            current_p = int(start_p + ratio * (target_p - start_p))
            current_step_positions[sid] = current_p

            if follower_driver is not None:
                follower_driver.write_position(sid, current_p)

        if sim_viewer is not None:
            sim_viewer.update_joints(current_step_positions)

        time.sleep(interval)


def load_motion_frames(filepath):
    """CSVからフレームデータを読み込む"""
    if not os.path.exists(filepath):
        return None
    frames = []
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp_sec"])
            positions = {sid: int(row[f"id_{sid}"]) for sid in SERVO_IDS}
            frames.append((t, positions))
    return frames


def execute_sequence(sequence_plan, follower_driver=None, sim_viewer=None):
    """
    プラン（ファイルと回数のリスト）を順に実行する
    """
    home_positions = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}
    current_state = dict(home_positions)

    # 1. 規定のHomeへ復帰
    smooth_move(follower_driver, sim_viewer, home_positions, fallback_state=current_state, duration=HOME_RETURN_DURATION)
    current_state = dict(home_positions)
    time.sleep(0.2)

    for item in sequence_plan:
        filename = item["file"]
        repeat_count = item["repeat"]
        filepath = os.path.join(BASE_DIR, "motions", filename)

        frames = load_motion_frames(filepath)
        if not frames:
            print(f"⚠️ ファイルの読み込みに失敗しました: {filename}")
            continue

        first_raw = frames[0][1]
        first_targets = {}
        temp_prev = {}
        temp_curr = dict(home_positions)
        for sid in SERVO_IDS:
            first_targets[sid] = calculate_target(sid, first_raw[sid], temp_prev, temp_curr)

        for r in range(repeat_count):
            # Home ➔ 開始姿勢
            smooth_move(follower_driver, sim_viewer, first_targets, fallback_state=current_state, duration=MOTION_START_TRANSITION)
            current_state = dict(first_targets)
            time.sleep(0.2)

            # 再生
            prev_leader = {}
            f_current = dict(home_positions)
            start_time = time.time()

            for t_target, raw_pos in frames:
                if sim_viewer is not None and not sim_viewer.is_running():
                    return

                while (time.time() - start_time) < t_target:
                    time.sleep(0.001)

                target_pos = {}
                for sid in SERVO_IDS:
                    val = calculate_target(sid, raw_pos[sid], prev_leader, f_current)
                    prev_leader[sid] = raw_pos[sid]
                    f_current[sid] = val
                    target_pos[sid] = val
                    current_state[sid] = val

                    if follower_driver is not None:
                        follower_driver.write_position(sid, val)

                if sim_viewer is not None:
                    sim_viewer.update_joints(target_pos)

            # 終了後 Home へ戻す
            smooth_move(follower_driver, sim_viewer, home_positions, fallback_state=current_state, duration=HOME_RETURN_DURATION)
            current_state = dict(home_positions)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="TACHIKOMA エージェントCLI")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    args = parser.parse_args()

    motions_json_path = os.path.join(BASE_DIR, "motions", "motions.json")
    planner = LLMTaskPlanner(motions_json_path)

    follower = None
    if args.arm:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"⚠️ 実機接続失敗: {e}")
            follower = None

    print("==================================================")
    print(" 🤖 TACHIKOMA 自然言語対話エージェント")
    print(f" ⚙️ 実機実行モード: {'有効 (--arm)' if follower else 'シミュレーションプレビューのみ'}")
    print("==================================================")

    try:
        while True:
            user_msg = input("\n🗣️ あなたの指示 (終了: q) > ").strip()
            if user_msg.lower() in ['q', 'quit', 'exit']:
                break
            if not user_msg:
                continue

            print("🧠 プラン生成中...")
            plan = planner.plan(user_msg)

            print(f"\n💬 タチコマ: {plan.get('reply_text')}")
            print(f"💡 思考理由: {plan.get('thought')}")
            
            sequence = plan.get("sequence", [])
            if not sequence:
                print("⚠️ 有効なシーケンスが生成されませんでした。")
                continue

            print("\n📋 【実行予定シーケンス】")
            for idx, item in enumerate(sequence, start=1):
                print(f"   {idx}. {item['file']} (x{item['repeat']}回)")

            # --- 1. 3D画面でのプレビュー再生 ---
            print("\n🖥️ 3Dシミュレータでプレビューを再生します...")
            try:
                sim = MujocoSimViewer()
                with sim.launch():
                    execute_sequence(sequence, follower_driver=None, sim_viewer=sim)
            except Exception as e:
                print(f"⚠️ シミュレータプレビューエラー: {e}")

            # --- 2. 実機実行の確認 ---
            if follower is not None:
                confirm = input("\n❓ この動作を実機で実行しますか？ [y/N] > ").strip().lower()
                if confirm == 'y':
                    print("🤖 実機でシーケンスを実行します...")
                    # トルクON
                    for sid in SERVO_IDS:
                        follower.set_torque(sid, True)
                    
                    execute_sequence(sequence, follower_driver=follower, sim_viewer=None)
                    
                    # 終了後は安全のため脱力
                    for sid in SERVO_IDS:
                        follower.set_torque(sid, False)
                    print("✅ 実機動作が完了しました。")
                else:
                    print("🛑 実機実行をキャンセルしました。")

    except KeyboardInterrupt:
        print("\n\n終了します。")
    finally:
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
            print("✅ フォロワーポートをクローズしました。")


if __name__ == "__main__":
    main()