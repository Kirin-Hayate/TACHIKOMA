"""
==============================================================================
リアルタイム遠隔操作 ＆ 可視化制御スクリプト (src/teleop_realtime_sim.py)
==============================================================================
【役割】
リーダーアームの操作を読み取り、以下の出力モードをフラグで自由に切り替えて実行します。
  1. ENABLE_FOLLOWER=True,  ENABLE_SIMULATOR=True  👉 実機追従 ＋ 3D画面表示（両方）
  2. ENABLE_FOLLOWER=False, ENABLE_SIMULATOR=True  👉 PC画面上の 3D シミュレータのみ
  3. ENABLE_FOLLOWER=True,  ENABLE_SIMULATOR=False 👉 実機フォロワー追従のみ（画面なし・軽量）
  4. ENABLE_FOLLOWER=False, ENABLE_SIMULATOR=False 👉 ターミナルでの数値モニタのみ
==============================================================================
"""

import sys
import time
import os

# ==============================================================================
# 1. 共通モジュール・設定のインポート
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import (
    LEADER_PORT,
    FOLLOWER_PORT,
    BAUDRATE,
    SAMPLING_RATE_HZ,
    SERVO_IDS,
    JOINT_CONFIG
)
from core.sts3215 import STS3215Driver
from core.kinematics import calculate_target
from core.sim_viewer import MujocoSimViewer

# ==============================================================================
# 2. 出力モード切り替えフラグ
# ==============================================================================
ENABLE_FOLLOWER  = False   # 👈 True: 実機フォロワーへ送信 / False: 実機送信OFF
ENABLE_SIMULATOR = True   # 👈 True: MuJoCo 3D画面を描画 / False: 3D画面OFF (CUIのみ)


def main():
    # 動作モードの表示文字列を作成
    mode_desc = []
    if ENABLE_FOLLOWER:
        mode_desc.append("実機フォロワー同期")
    if ENABLE_SIMULATOR:
        mode_desc.append("MuJoCo 3D表示")
    if not mode_desc:
        mode_desc.append("ターミナル数値モニタのみ")
    mode_str = " + ".join(mode_desc)

    print("==================================================")
    print(f" 🎬 TACHIKOMA リアルタイム遠隔操作 [{mode_str}]")
    print("==================================================")
    print("👉 リーダーアームを操作してください。")
    print("👉 終了時は [Ctrl+C] または ウィンドウを閉じてください。\n")

    leader = None
    follower = None
    sim = None

    # 1. リーダーポート（操作側）のオープン
    try:
        leader = STS3215Driver(LEADER_PORT, baudrate=BAUDRATE, timeout=0.01)
        print(f"✅ リーダー接続完了 ({LEADER_PORT})")
    except Exception as e:
        print(f"❌ リーダーの接続に失敗しました ({LEADER_PORT}): {e}")
        return

    # 2. フォロワーポート（実機追従側）のオープン
    is_follower_active = False
    if ENABLE_FOLLOWER:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            for sid in SERVO_IDS:
                follower.set_torque(sid, True)
            is_follower_active = True
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"⚠️ 実機フォロワー接続失敗: {e}")
            print("👉 実機送信はスキップします。")
    else:
        print("💡 ENABLE_FOLLOWER = False のため、実機フォロワーへの接続はスキップしました。")

    # 3. MuJoCo 3Dシミュレータビューアの初期化
    if ENABLE_SIMULATOR:
        try:
            sim = MujocoSimViewer()
        except Exception as e:
            print(f"⚠️ シミュレータの初期化に失敗しました: {e}")
            print("👉 3D描画はスキップして継続します。")
            sim = None
    else:
        print("💡 ENABLE_SIMULATOR = False のため、3Dビューアの起動はスキップしました。")

    # 追従計算用キャッシュの準備
    prev_leader_cache = {}
    follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}
    last_valid_positions = {sid: 2048 for sid in SERVO_IDS}

    interval = 1.0 / SAMPLING_RATE_HZ  # 50Hz = 0.02秒
    frame_count = 0
    first_draw = True

    # ターミナルちらつき防止のためカーソルを非表示
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        # メインループを実行する関数
        def run_loop():
            nonlocal frame_count, first_draw
            while True:
                # ビューア起動中にウィンドウが閉じられたら終了
                if sim is not None and not sim.is_running():
                    break

                loop_start = time.time()

                target_positions = {}
                lines = []
                lines.append("========================================")
                lines.append(f" ⏱️ リアルタイム動作中 [{frame_count:5d} frames]")
                lines.append(f" 🤖 実機フォロワー: {'ON' if is_follower_active else 'OFF'} | 🖥️ 3D画面: {'ON' if sim is not None else 'OFF'}")
                lines.append("========================================")

                # --- 全軸の角度取得 ➔ 目標値計算 ➔ 実機送信 ＆ 3D描画 ---
                for sid in SERVO_IDS:
                    raw_pos = leader.read_position(sid)
                    if raw_pos is not None:
                        last_valid_positions[sid] = raw_pos

                    current_raw = last_valid_positions[sid]

                    # 目標値を計算 (0〜4095)
                    target_pos = calculate_target(sid, current_raw, prev_leader_cache, follower_current_cache)
                    prev_leader_cache[sid] = current_raw
                    follower_current_cache[sid] = target_pos
                    target_positions[sid] = target_pos

                    # 実機フォロワーが有効な場合のみパケット送信
                    if is_follower_active:
                        follower.write_position(sid, target_pos)

                    lines.append(f" [ID {sid}]  Raw: {current_raw:4d}  |  Target: {target_pos:4d}")

                # 3Dシミュレータが有効な場合は関節姿勢を反映
                if sim is not None:
                    sim.update_joints(target_positions)

                lines.append("========================================")
                lines.append(" [Ctrl+C] で終了")

                # コンソール上書き表示
                if not first_draw:
                    sys.stdout.write(f"\033[{len(lines)}A")
                else:
                    first_draw = False

                sys.stdout.write("\n".join(line + "\033[K" for line in lines) + "\n")
                sys.stdout.flush()

                frame_count += 1

                # 50Hz周期を維持
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # 3Dシミュレータが有効なら launch コンテキスト内で実行、無効ならそのままループ実行
        if sim is not None:
            with sim.launch():
                run_loop()
        else:
            run_loop()

    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n\n🛑 停止しました。\n")
    except Exception as e:
        sys.stdout.write("\033[?25h")
        print(f"\n❌ エラー: {e}")
    finally:
        sys.stdout.write("\033[?25h")
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
        if leader is not None:
            leader.close()
        print("✅ 全ポートをクローズし、安全に終了しました。")


if __name__ == "__main__":
    main()