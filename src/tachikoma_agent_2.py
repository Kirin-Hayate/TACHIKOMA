"""
==============================================================================
TACHIKOMA パラメトリック対話エージェント (動的 IK・ラジアン完全版)
(src/tachikoma_agent_2.py)
==============================================================================
【役割】
1. ターミナルから自然言語指示を入力（例: 「正面の手前にあるものを右奥へ運んで」「いま何時？」）。
2. LLM (Gemini) が搬送タスクの要否と、Pick / Place の極座標 [r, theta, z] を特定。
3. タスクなし（対話・情報照会）の場合は応答のみを表示して待機。
4. タスクありの場合はその場で動的に IK 軌道を生成し、
   3Dシミュレータ (MuJoCo) でプレビュー再生 ➔ ユーザー承認 ('y') 後に実機実行。

【実行コマンド例】
  - 画面シミュレーションプレビューのみ（実機なしテスト）:
      python src/tachikoma_agent_1.py

  - 実機フォロワー接続モード（承認後に実機が動作）:
      python src/tachikoma_agent_1.py --arm
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
from core.kinematics import (
    raw_to_radian,
    radian_to_raw,
    get_home_radians
)

# 安全動作パラメータ (秒単位)
HOME_RETURN_DURATION = 2.5
START_APPROACH_DURATION = 2.0


def smooth_move_rad(follower_driver, sim_viewer, target_rad, fallback_state_rad=None, duration=2.5, steps=75):
    """
    物理ラジアン空間でのコサイン S 字加減速による安全補間移動。
    初速・終速がゼロになるため、アームの急発進・急停止を防止します。
    """
    start_positions = {}
    for sid in SERVO_IDS:
        if follower_driver is not None:
            p_raw = follower_driver.read_position(sid)
            if p_raw is not None:
                start_positions[sid] = raw_to_radian(sid, p_raw)
            elif fallback_state_rad and sid in fallback_state_rad:
                start_positions[sid] = fallback_state_rad[sid]
            else:
                start_positions[sid] = raw_to_radian(sid, JOINT_CONFIG[sid]["init"])
        else:
            if fallback_state_rad and sid in fallback_state_rad:
                start_positions[sid] = fallback_state_rad[sid]
            else:
                start_positions[sid] = raw_to_radian(sid, JOINT_CONFIG[sid]["init"])

    interval = duration / steps
    for step in range(1, steps + 1):
        t = step / steps
        ratio = (1.0 - math.cos(t * math.pi)) / 2.0

        current_step_rad = {}
        for sid in SERVO_IDS:
            start_val = start_positions[sid]
            target_val = target_rad.get(sid, raw_to_radian(sid, JOINT_CONFIG[sid]["init"]))
            current_val = start_val + ratio * (target_val - start_val)
            current_step_rad[sid] = current_val

            # 実機フォロワー送信時のみ Raw 値に変換
            if follower_driver is not None:
                raw_val = radian_to_raw(sid, current_val)
                follower_driver.write_position(sid, raw_val)

        # 3D シミュレータ側はラジアン直接反映
        if sim_viewer is not None:
            sim_viewer.update_joints_rad(current_step_rad)

        time.sleep(interval)


def play_sequence(task_list, generator, follower_driver=None, sim_viewer=None):
    """
    指示された搬送タスクリストを順番に実行。
    各タスクごとに動的に IK を解き、軌道を生成・再生します。
    """
    home_rad = get_home_radians()
    current_state_rad = dict(home_rad)

    # 1. まず規定の Home 位置へ安全移動
    smooth_move_rad(follower_driver, sim_viewer, home_rad, fallback_state_rad=current_state_rad, duration=HOME_RETURN_DURATION)
    current_state_rad = dict(home_rad)
    time.sleep(0.2)

    total_tasks = len(task_list)
    for t_idx, task in enumerate(task_list, start=1):
        pick_coord = task["pick"]
        place_coord = task["place"]

        print(f"\n--- 🎬 [ステップ {t_idx}/{total_tasks}] : {task.get('description', '搬送タスク')} ---")
        print(f"    Pick 座標 : 距離={pick_coord['r']*100:.1f}cm, 角度={pick_coord['theta_deg']:+.1f}°, 高さ={pick_coord['z']*1000:.1f}mm")
        print(f"    Place座標 : 距離={place_coord['r']*100:.1f}cm, 角度={place_coord['theta_deg']:+.1f}°, 高さ={place_coord['z']*1000:.1f}mm")

        # その場で動的に逆運動学 (IK) 軌道を生成
        success, frames, ik_log = generator.generate_from_coords(pick_coord, place_coord)
        if not success:
            print(f"🚨 [IK 算出不能] {ik_log}")
            print("👉 このタスクはスキップし、安全のためその場で待機します。")
            continue

        print(f"    ⚙️ {ik_log} / 総フレーム数: {len(frames)} (約 {frames[-1][0]:.1f} 秒)")

        # モーション開始姿勢へ S 字アプローチ
        first_target = frames[0][1]
        smooth_move_rad(follower_driver, sim_viewer, first_target, fallback_state_rad=current_state_rad, duration=START_APPROACH_DURATION)
        current_state_rad = dict(first_target)
        time.sleep(0.2)

        # 軌道再生ループ (速度倍率対応)
        total_duration = frames[-1][0]
        sim_time = 0.0
        last_wall_time = time.time()
        frame_idx = 0

        while frame_idx < len(frames):
            if sim_viewer is not None and not sim_viewer.is_running():
                return

            now = time.time()
            dt = now - last_wall_time
            last_wall_time = now

            # 一時停止中の処理
            if sim_viewer is not None and sim_viewer.paused:
                time.sleep(0.02)
                continue

            speed = sim_viewer.playback_speed if sim_viewer is not None else 1.0
            sim_time += dt * speed

            t_target, rad_positions = frames[frame_idx]
            if sim_time < t_target:
                time.sleep(0.001)
                continue

            current_state_rad = dict(rad_positions)

            # 実機フォロワー送信時のみ Raw 値に変換
            if follower_driver is not None:
                for sid in SERVO_IDS:
                    raw_val = radian_to_raw(sid, rad_positions[sid])
                    follower_driver.write_position(sid, raw_val)

            # 3D シミュレータ側はラジアン直接反映
            if sim_viewer is not None:
                sim_viewer.update_joints_rad(rad_positions)

            speed_str = f"({speed:.0f}x)" if speed > 1.0 else ""
            sys.stdout.write(f"\r⏱️ 再生中{speed_str}: {t_target:6.2f}s / {total_duration:6.2f}s [Frame {frame_idx + 1}/{len(frames)}]  ")
            sys.stdout.flush()

            frame_idx += 1

        print("")
        # 1ステップ終了後 Home 復帰
        smooth_move_rad(follower_driver, sim_viewer, home_rad, fallback_state_rad=current_state_rad, duration=HOME_RETURN_DURATION)
        current_state_rad = dict(home_rad)
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="TACHIKOMA パラメトリック対話エージェント (動的 IK 版)")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")
    args = parser.parse_args()

    print("==================================================")
    print(" 🤖 TACHIKOMA パラメトリック対話エージェント (動的 IK 版)")
    print(f" ⚙️ 実機実行モード: {'有効 (--arm)' if args.arm else 'シミュレーションプレビューのみ'}")
    print(" 📐 軌道生成方式: オンデマンド数理 IK (テンプレートフリー)")
    print("==================================================")

    planner = LLMTaskPlanner()
    generator = ParametricMotionGenerator()

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
            for idx, task in enumerate(tasks, start=1):
                p = task['pick']
                d = task['place']
                print(f"   {idx}. {task.get('description', '')} (Pick: r={p['r']*100:.0f}cm, θ={p['theta_deg']:+.0f}° ➔ Place: r={d['r']*100:.0f}cm, θ={d['theta_deg']:+.0f}°)")

            # --- 1. 3D シミュレータプレビュー ---
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