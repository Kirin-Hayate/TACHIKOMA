import sys
import time
from scservo_sdk import* # STS3215 用 SDK

# ==========================================
# 1. 通信設定・ポート設定
# ==========================================
LEADER_PORT = "COM3"
FOLLOWER_PORT = "COM4"
BAUDRATE = 1000000  # デフォルト通信速度（必要に応じて変更）

# モーターID一覧（ID 1 〜 ID 6）
SERVO_IDS = [1, 2, 3, 4, 5, 6]

# STS3215 レジスタアドレス定義
ADDR_STS_TORQUE_ENABLE = 40
ADDR_STS_GOAL_POSITION = 42
ADDR_STS_PRESENT_POSITION = 56
ADDR_STS_GOAL_SPEED = 46
ADDR_STS_GOAL_ACC = 47

# ==========================================
# 2. 基準オフセット値（生データ）
# ==========================================
# ゼロ点姿勢で取得した値
OFFSET_LEADER = {1: 0, 2: 1750, 3: 3104, 4: 1643, 5: 1952, 6: 1980}

OFFSET_FOLLOWER = {1: 2166, 2: 1967, 3: 2022, 4: 3780, 5: 1021, 6: 1843}

# サーボの回転方向反転（1: そのまま, -1: 逆回転）※動作方向が逆の場合は -1 に設定
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# 安全リミット（Follower への指令値制限）
LIMIT_MIN = 50
LIMIT_MAX = 4045


# ==========================================
# 3. 差分計算・ラップアラウンド対策関数
# ==========================================
def calculate_follower_target(servo_id, raw_leader):
    offset_l = OFFSET_LEADER[servo_id]
    offset_f = OFFSET_FOLLOWER[servo_id]
    direction = DIRECTION[servo_id]

    # Leader の相対変位
    diff = raw_leader - offset_l

    # 4096境界のラップアラウンド（急激なジャンプ）対策
    if diff > 2048:
        diff -= 4096
    elif diff < -2048:
        diff += 4096

    # 方向を反映して Follower の目標値を算出
    target = offset_f + (diff * direction)

    # 安全リミッター
    if target < LIMIT_MIN:
        target = LIMIT_MIN
    elif target > LIMIT_MAX:
        target = LIMIT_MAX

    return int(target)


# ==========================================
# 4. メイン処理
# ==========================================
def main():
    # ポートハンドラーとパケットハンドラーの初期化
    port_leader = PortHandler(LEADER_PORT)
    port_follower = PortHandler(FOLLOWER_PORT)
    packet_handler = PacketHandler(PROTOCOL_STS)

    # ポートオープン
    if not port_leader.openPort() or not port_leader.setBaudRate(BAUDRATE):
        print(f"❌ Leader ポート ({LEADER_PORT}) のオープンに失敗しました。")
        sys.exit(1)

    if not port_follower.openPort() or not port_follower.setBaudRate(BAUDRATE):
        print(f"❌ Follower ポート ({FOLLOWER_PORT}) のオープンに失敗しました。")
        port_leader.closePort()
        sys.exit(1)

    print(f"✅ Leader ({LEADER_PORT}) & Follower ({FOLLOWER_PORT}) 接続成功")

    try:
        # Leader: トルクOFF（手動で動かせるように脱力）
        for sid in SERVO_IDS:
            packet_handler.write1ByteTxRx(
                port_leader, sid, ADDR_STS_TORQUE_ENABLE, 0
            )

        # Follower: トルクON
        for sid in SERVO_IDS:
            packet_handler.write1ByteTxRx(
                port_follower, sid, ADDR_STS_TORQUE_ENABLE, 1
            )
            # 安全のため初動スピード・加速度を適度に設定
            packet_handler.write2ByteTxRx(
                port_follower, sid, ADDR_STS_GOAL_SPEED, 1500
            )
            packet_handler.write1ByteTxRx(
                port_follower, sid, ADDR_STS_GOAL_ACC, 50
            )

        print("\n🚀 同期制御を開始します。[Ctrl + C] で停止")

        while True:
            targets = {}
            # 1. Leader の角度読み取り & 目標値計算
            for sid in SERVO_IDS:
                raw_pos, result, error = packet_handler.read2ByteTxRx(
                    port_leader, sid, ADDR_STS_PRESENT_POSITION
                )
                if result == COMM_SUCCESS:
                    targets[sid] = calculate_follower_target(sid, raw_pos)
                else:
                    # 読み取りエラー時はスキップ
                    targets[sid] = None

            # 2. Follower へ角度書き込み
            for sid, target_pos in targets.items():
                if target_pos is not None:
                    packet_handler.write2ByteTxRx(
                        port_follower, sid, ADDR_STS_GOAL_POSITION, target_pos
                    )

            # 画面表示（確認用）
            log_str = " | ".join(
                [
                    f"ID{sid}:{targets[sid]}"
                    for sid in SERVO_IDS
                    if targets[sid] is not None
                ]
            )
            print(f"\rTarget -> {log_str}", end="")

            # ループ周期調整（約 50Hz）
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n⏹ 同期を停止しました。安全のため Follower のトルクを解放します。")
        for sid in SERVO_IDS:
            packet_handler.write1ByteTxRx(
                port_follower, sid, ADDR_STS_TORQUE_ENABLE, 0
            )

    finally:
        port_leader.closePort()
        port_follower.closePort()
        print("✅ ポートをクローズしました。")


if __name__ == "__main__":
    main()
    