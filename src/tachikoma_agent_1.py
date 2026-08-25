"""
==============================================================================
TACHIKOMA パラメトリック対話エージェント (src/tachikoma_agent.py)
==============================================================================
【役割】
1. ターミナルから自然言語指示を入力（例: 「右のものを左へ」「いま何時？」）。
2. LLM (Gemini) が搬送タスクの要否と掴み・配置角度を判定。
3. タスクなし（対話・情報照会）の場合は応答のみ表示して終了。
4. タスクありの場合は ParametricMotionGenerator で軌道を生成し、
   3Dシミュレータ (MuJoCo) でプレビュー再生 ➔ ユーザー承認 ('y') 後に実機実行。
==============================================================================
"""
import sys
import time
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
from core.sim_viewer import MujocoSimViewer
from core.llm_planner import LLMTaskPlanner
from core.motion_generator import ParametricMotionGenerator

HOME_RETURN_DURATION = 2.5
START_APPROACH_DURATION = 2.0


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


def play_sequence(task_list, generator, follower_driver=None, sim_viewer=None):
    """複数の搬送タスクを順番に実行"""
    home_positions = {sid: JOINT_CONFIG[sid]["init"] for sid in SERVO_IDS}
    current_state = dict(home_positions)

    # 1. 規定のHomeへ復帰
    smooth_move(follower_driver, sim_viewer, home_positions, fallback_state=current_state, duration=HOME_RETURN_DURATION)
    current_state = dict(home_positions)
    time.sleep(0.2)

    total_tasks = len(task_list)
    for t_idx, task in enumerate(task_list, start=1):
        pick_raw = task["theta_pick_raw"]
        place_raw = task["theta_place_raw"]

        print(f"\n--- 🎬 [ステップ {t_idx}/{total_tasks}] : [{task.get('pick_location')}] ➔ [{task.get('place_location')}] ---")
        frames = generator.generate(theta_pick_raw=pick_raw, theta_place_raw=place_raw)

        # 開始姿勢へのS字アプローチ
        first_target = frames[0][1]
        smooth_move(follower_driver, sim_viewer, first_target, fallback_state=current_state, duration=START_APPROACH_DURATION)
        current_state = dict(first_target)
        time.sleep(0.2)

        # 軌道再生
        start_time = time.time()
        total_duration = frames[-1][0]

        for frame_idx, (t_target, target_positions) in enumerate(frames):
            if sim_viewer is not None and not sim_viewer.is_running():
                return

            while (time.time() - start_time) < t_target:
                time.sleep(0.001)

            current_state = dict(target_positions)
            if follower_driver is not None:
                for sid in SERVO_IDS:
                    follower_driver.write_position(sid, target_positions[sid])

            if sim_viewer is not None:
                sim_viewer.update_joints(target_positions)

            sys.stdout.write(f"\r⏱️ 再生中: {t_target:6.2f}s / {total_duration:6.2f}s [Frame {frame_idx + 1}/{len(frames)}]  ")
            sys.stdout.flush()

        print("")
        # 1ステップ終了後 Home 復帰
        smooth_move(follower_driver, sim_viewer, home_positions, fallback_state=current_state, duration=HOME_RETURN_DURATION)
        current_state = dict(home_positions)
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="TACHIKOMA パラメトリック対話エージェント")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    parser.add_argument("--template", type=str, default="motion_20260825_222909.csv", help="基準テンプレートCSV")
    args = parser.parse_args()

    print("==================================================")
    print(" 🤖 TACHIKOMA パラメトリック対話エージェント")
    print(f" ⚙️ 実機実行モード: {'有効 (--arm)' if args.arm else 'シミュレーションプレビューのみ'}")
    print(f" 📁 基準テンプレート: {args.template}")
    print("==================================================")

    planner = LLMTaskPlanner()
    generator = ParametricMotionGenerator(args.template)

    follower = None
    if args.arm:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"⚠️ 実機接続失敗: {e}")
            follower = None

    try:
        while True:
            user_msg = input("\n🗣️ 指示を入力 (終了: q) > ").strip()
            if user_msg.lower() in ['q', 'quit', 'exit']:
                break
            if not user_msg:
                continue

            print("🧠 指示解析中...")
            plan = planner.plan(user_msg)

            print(f"\n💬 応答: {plan.get('reply_text')}")
            print(f"💡 ログ: {plan.get('thought')}")

            tasks = plan.get("tasks") or []

            # 物理タスクがない場合は対話のみで待機
            if not tasks:
                print("ℹ️ 物理マニピュレーション不要と判断しました。待機状態を維持します。")
                continue

            print(f"\n📋 【生成された搬送シーケンス: 全 {len(tasks)} 件】")
            for idx, t in enumerate(tasks, start=1):
                print(f"   {idx}. [{t.get('pick_location')}] ➔ [{t.get('place_location')}] (ID1: {t.get('theta_pick_raw')} ➔ {t.get('theta_place_raw')})")

            # --- 1. 3Dシミュレータプレビュー ---
            print("\n🖥️ 3Dシミュレータでプレビューを再生します...")
            try:
                sim = MujocoSimViewer()
                with sim.launch():
                    play_sequence(tasks, generator, follower_driver=None, sim_viewer=sim)
            except Exception as e:
                print(f"⚠️ シミュレータプレビューエラー: {e}")

            # --- 2. 実機承認実行 ---
            if follower is not None:
                confirm = input("\n❓ このシーケンスを実機フォロワーで実行しますか？ [y/N] > ").strip().lower()
                if confirm == 'y':
                    print("🤖 実機でシーケンスを実行中...")
                    for sid in SERVO_IDS:
                        follower.set_torque(sid, True)

                    play_sequence(tasks, generator, follower_driver=follower, sim_viewer=None)

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