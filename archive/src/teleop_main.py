"""
==============================================================================
TACHIKOMA 統合遠隔操作スクリプト (ラジアン統一版)
(src/teleop_main.py)
==============================================================================
【役割】
リーダーアーム（操作側）の動きをリアルタイムに読み取り、フォロワーアーム（追従側）
および 3D シミュレータへ伝達します。
内部演算および CSV 記録データを「物理関節ラジアン」で統一しており、
記録したファイルはそのまま `replay_main.py` で完全互換再生が可能です。

コマンドライン引数（実行時オプション）によって、以下の機能を自由に組み合わせられます:
  1. 実機フォロワーの追従  （デフォルト: OFF / 有効化: --arm）
  2. 画面上の 3D 描画     （デフォルト: OFF / 有効化: --sim）
  3. 動作データの CSV 記録（デフォルト: OFF / 有効化: --record）

【安全機能】
1. 起動時アプローチ補間（急発進防止）:
   メインループ開始前に、トルク OFF 状態で実機フォロワーの現在姿勢を読み取ります。
   そこからリーダーアームの初期姿勢へ「コサイン S 字補間（初速 0）」で
   ゆっくり移動してから、50Hz のリアルタイム遠隔操作へ移行します。
2. ハードウェア境界での Raw 値変換:
   演算・記録・画面描画はすべて物理ラジアン角で行い、実機サーボへの送信直前にのみ
   `radian_to_raw()` を通して書き込みます。

【実行コマンド例】
  - 画面シミュレーションのみ（安全な動作確認用）
      python src/teleop_main.py --sim

  - 実機同期 ＋ 画面表示（通常の遠隔操作）
      python src/teleop_main.py --arm --sim

  - 実機同期 ＋ 画面表示 ＋ ラジアン CSV 記録（モーションティーチング）
      python src/teleop_main.py --arm --sim --record

  - 実機同期のみ（画面なし・軽量動作）
      python src/teleop_main.py --arm
==============================================================================
"""

import sys
import time
import csv
import os
import math
import argparse
from datetime import datetime

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
from core.kinematics import (
    raw_to_radian,
    radian_to_raw,
    calculate_target_rad,
    get_home_radians
)
from core.sim_viewer import MujocoSimViewer

# 起動時にリーダーの姿勢へフォロワーを合わせる秒数
STARTUP_APPROACH_DURATION = 2.5


# ==============================================================================
# 2. 実行時コマンドライン引数の解析 (argparse)
# ==============================================================================
def parse_arguments():
    """コマンドライン引数を定義・取得する関数"""
    parser = argparse.ArgumentParser(
        description="TACHIKOMA リアルタイム遠隔操作スクリプト (ラジアン統一版)"
    )

    # --arm オプション (実機フォロワーへ送信する)
    parser.add_argument(
        "--arm",
        action="store_true",
        help="実機フォロワーへのコマンド送信・追従動作を有効化します（デフォルト: OFF）"
    )

    # --sim オプション (3D シミュレータを描画する)
    parser.add_argument(
        "--sim",
        action="store_true",
        help="MuJoCo 3Dシミュレーション描画ウィンドウを起動します（デフォルト: OFF）"
    )

    # --record オプション (ラジアン CSV を保存する)
    parser.add_argument(
        "--record",
        action="store_true",
        help="CSVファイルへのラジアン動作データ記録を有効化します（デフォルト: OFF）"
    )

    return parser.parse_args()


# ==============================================================================
# 3. 補助関数 (安全な S 字補間)
# ==============================================================================
def smooth_move_rad(follower_driver, sim_viewer, start_rad, target_rad, duration=2.5, steps=75):
    """
    開始姿勢から目標姿勢へコサイン S 字加減速で滑らかに移動する（ラジアン空間）。
    初速と終速がゼロになるため、モーターへの過負荷や急発進が起きません。
    """
    interval = duration / steps
    for step in range(1, steps + 1):
        t = step / steps
        ratio = (1.0 - math.cos(t * math.pi)) / 2.0

        current_step_rad = {}
        for sid in SERVO_IDS:
            s_val = start_rad.get(sid, 0.0)
            e_val = target_rad.get(sid, 0.0)
            cur_val = s_val + ratio * (e_val - s_val)
            current_step_rad[sid] = cur_val

            # 実機フォロワー送信時のみ Raw 値に変換
            if follower_driver is not None:
                raw_val = radian_to_raw(sid, cur_val)
                follower_driver.write_position(sid, raw_val)

        # 3D シミュレータ側はラジアンのまま直接反映
        if sim_viewer is not None:
            sim_viewer.update_joints_rad(current_step_rad)

        time.sleep(interval)


# ==============================================================================
# 4. メイン処理
# ==============================================================================
def main():
    args = parse_arguments()

    enable_follower = args.arm
    enable_sim = args.sim
    enable_record = args.record

    mode_list = []
    if enable_follower:
        mode_list.append("🤖 実機フォロワー同期")
    if enable_sim:
        mode_list.append("🖥️ 3D画面描画")
    if enable_record:
        mode_list.append("📁 ラジアンCSV記録")
    if not mode_list:
        mode_list.append("ターミナル数値モニタのみ")

    print("==================================================")
    print(" 🎬 TACHIKOMA 統合遠隔操作システム (ラジアン統一版)")
    print(f" ⚙️ 有効機能: {' + '.join(mode_list)}")
    print("==================================================")

    # --- 保存先ファイルの設定（記録モード時のみ） ---
    output_filepath = None
    csv_file = None
    csv_writer = None

    if enable_record:
        motions_dir = os.path.join(BASE_DIR, "motions")
        os.makedirs(motions_dir, exist_ok=True)
        filename = f"teleop_rad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_filepath = os.path.join(motions_dir, filename)
        print(f"📁 記録先: {output_filepath}")

    leader = None
    follower = None
    sim = None

    # --- 1. リーダーポート（操作側）のオープン ---
    try:
        leader = STS3215Driver(LEADER_PORT, baudrate=BAUDRATE, timeout=0.01)
        print(f"✅ リーダー接続完了 ({LEADER_PORT})")
    except Exception as e:
        print(f"❌ リーダーの接続に失敗しました ({LEADER_PORT}): {e}")
        return

    # --- 2. フォロワーポート（実機追従側）のオープン判定 ---
    is_follower_active = False
    if enable_follower:
        try:
            follower = STS3215Driver(FOLLOWER_PORT, baudrate=BAUDRATE, timeout=0.01)
            is_follower_active = True
            print(f"✅ 実機フォロワー接続完了 ({FOLLOWER_PORT})")
        except Exception as e:
            print(f"⚠️ 実機フォロワー接続失敗: {e}")
            print("👉 実機送信はスキップして継続します。")
    else:
        print("💡 --arm が指定されていないため、実機フォロワーは動作しません（安全モード）。")

    # --- 3. MuJoCo 3D シミュレータの初期化判定 ---
    if enable_sim:
        try:
            sim = MujocoSimViewer()
            print("✅ 3Dシミュレータ初期化完了")
        except Exception as e:
            print(f"⚠️ 3Dシミュレータ初期化失敗: {e}")
            sim = None
    else:
        print("💡 --sim が指定されていないため、3D画面描画はスキップします。")

    # --- 4. CSV 記録ファイルのオープン (ラジアンヘッダー: q1〜q6) ---
    if enable_record:
        csv_file = open(output_filepath, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        header = ["timestamp_sec", "q1", "q2", "q3", "q4", "q5", "q6"]
        csv_writer.writerow(header)

    # 追従計算用キャッシュの準備
    prev_leader_cache = {}
    follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}
    last_valid_positions = {sid: 2048 for sid in SERVO_IDS}

    # --------------------------------------------------------------------------
    # 🛡️ 起動時アプローチ補間（現在姿勢 ➔ リーダー初期姿勢）
    # --------------------------------------------------------------------------
    print("\n🔍 リーダーおよびフォロワーの初期姿勢をスキャン中...")
    initial_leader_raw = {}
    initial_targets_rad = {}
    follower_startup_rad = {}

    # リーダーの現在姿勢を全軸読み取り
    for sid in SERVO_IDS:
        pos = leader.read_position(sid)
        initial_leader_raw[sid] = pos if pos is not None else 2048
        last_valid_positions[sid] = initial_leader_raw[sid]

    # リーダー初期姿勢に対応するフォロワー目標ラジアンを計算
    for sid in SERVO_IDS:
        target_rad = calculate_target_rad(sid, initial_leader_raw[sid], prev_leader_cache, follower_current_cache)
        initial_targets_rad[sid] = target_rad
        prev_leader_cache[sid] = initial_leader_raw[sid]
        follower_current_cache[sid] = radian_to_raw(sid, target_rad)

    # 実機フォロワーの静止姿勢をスキャン（トルク OFF 安全状態）
    if is_follower_active:
        for sid in SERVO_IDS:
            pos = follower.read_position(sid)
            follower_startup_rad[sid] = raw_to_radian(sid, pos) if pos is not None else initial_targets_rad[sid]

        # 読み取った角度を保持した状態でトルク ON
        for sid in SERVO_IDS:
            follower.write_position(sid, radian_to_raw(sid, follower_startup_rad[sid]))
            follower.set_torque(sid, True)
    else:
        follower_startup_rad = dict(initial_targets_rad)

    # 3D モデル画面を初期姿勢に同期
    if sim is not None:
        sim.update_joints_rad(follower_startup_rad)

    # S 字補間でリーダーの現在姿勢へゆっくりアプローチ
    if is_follower_active or sim is not None:
        print(f"🎯 フォロワーをリーダーの現在姿勢へ同期中 ({STARTUP_APPROACH_DURATION}秒)...")
        smooth_move_rad(follower, sim, follower_startup_rad, initial_targets_rad, duration=STARTUP_APPROACH_DURATION)
        print("✅ 同期完了！リアルタイム遠隔操作を開始します。\n")

    print("👉 リーダーアームを操作してください。")
    print("👉 終了時は [Ctrl+C] または ウィンドウを閉じてください。\n")

    start_time = time.time()
    interval = 1.0 / SAMPLING_RATE_HZ  # 50Hz = 0.02秒
    frame_count = 0
    first_draw = True

    # ターミナルのカーソルを非表示
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        def run_loop():
            nonlocal frame_count, first_draw
            while True:
                if sim is not None and not sim.is_running():
                    break

                loop_start = time.time()
                current_timestamp = loop_start - start_time

                current_csv_row = [f"{current_timestamp:.4f}"] if enable_record else []
                target_positions_rad = {}

                # ターミナル表示文字列の作成
                lines = []
                lines.append("=========================================================")
                lines.append(f" ⏱️ 動作中  {current_timestamp:5.2f}s [{frame_count:5d} frames]")
                lines.append(f" 🤖 実機: {'ON' if is_follower_active else 'OFF'} | 🖥️ 3D: {'ON' if sim is not None else 'OFF'} | 📁 記録: {'ON' if enable_record else 'OFF'}")
                lines.append("---------------------------------------------------------")

                # --- 各軸の読み取り・計算・送信 ---
                for sid in SERVO_IDS:
                    raw_pos = leader.read_position(sid)
                    if raw_pos is not None:
                        last_valid_positions[sid] = raw_pos

                    current_raw = last_valid_positions[sid]

                    # 目標ラジアン計算 (infinite 軸は内部で follower_current_cache をアンバウンド更新)
                    target_rad = calculate_target_rad(sid, current_raw, prev_leader_cache, follower_current_cache)
                    target_positions_rad[sid] = target_rad

                    # 実機サーボ用の物理クランプ値 (0〜4095)
                    target_raw = radian_to_raw(sid, target_rad)
                    prev_leader_cache[sid] = current_raw

                    # ★ 修正ポイント:
                    # bounded 軸のみクランプ値を反映し、infinite 軸は calculate_target_rad 内で
                    # 保持した仮想累積値を維持する
                    if JOINT_CONFIG[sid]["type"] == "bounded":
                        follower_current_cache[sid] = target_raw

                    # CSV 記録用データ（ラジアン）へ追加
                    if enable_record:
                        current_csv_row.append(f"{target_rad:.5f}")

                    # 実機フォロワーへ送信（--arm 指定時のみ）
                    if is_follower_active:
                        follower.write_position(sid, target_raw)

                    deg_val = math.degrees(target_rad)
                    lines.append(f" [ID {sid}] Raw: {current_raw:4d} ➔ Target: {target_rad:+6.3f} rad ({deg_val:+6.1f}°)")

                # 3D モデルの描画更新（ラジアン直接反映）
                if sim is not None:
                    sim.update_joints_rad(target_positions_rad)

                lines.append("=========================================================")
                lines.append(" [Ctrl+C] で終了")

                # コンソール上書き表示
                if not first_draw:
                    sys.stdout.write(f"\033[{len(lines)}A")
                else:
                    first_draw = False

                sys.stdout.write("\n".join(line + "\033[K" for line in lines) + "\n")
                sys.stdout.flush()

                # CSV 書き込み
                if enable_record and csv_writer:
                    csv_writer.writerow(current_csv_row)

                frame_count += 1

                # 50Hz 周期の維持
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if sim is not None:
            with sim.launch():
                run_loop()
        else:
            run_loop()

    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n\n🛑 停止しました。\n")
        if enable_record:
            print(f"📁 計 {frame_count} フレームをラジアン形式で '{output_filepath}' に保存しました。")
    except Exception as e:
        sys.stdout.write("\033[?25h")
        print(f"\n❌ エラー: {e}")
    finally:
        sys.stdout.write("\033[?25h")  # カーソル再表示
        if csv_file is not None and not csv_file.closed:
            csv_file.close()
        if follower is not None:
            for sid in SERVO_IDS:
                follower.set_torque(sid, False)
            follower.close()
        if leader is not None:
            leader.close()
        print("✅ 全リソースを安全にクローズしました。")


if __name__ == "__main__":
    main()