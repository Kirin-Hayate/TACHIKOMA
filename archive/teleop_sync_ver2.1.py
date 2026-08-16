import sys
import time
import serial

# ==========================================
# 1. 通信設定・ポート設定
# ==========================================
LEADER_PORT = 'COM3'      # 操作側（リーダー）が接続されているシリアルポート
FOLLOWER_PORT = 'COM4'    # 追従側（フォロワー）が接続されているシリアルポート
BAUDRATE = 1000000        # サーボモーターの通信速度（1Mbps）

# 🧪 【試験用フィルタ】テストしたい関節のIDのみをリストに指定します
# 例: [1] のみにするとID1だけが同期し、他のサーボはトルクがOFFになります
TEST_IDS = [1]

SERVO_IDS = [1, 2, 3, 4, 5, 6]
# 各関節の回転方向の向き（必要に応じて 1 または -1 に反転させます）
DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# ==========================================
# 2. 画像に基づく各関節の設定プロファイル
# ==========================================
JOINT_CONFIG = {
    # type: "bounded" = 物理的な最小・最大範囲を持つ有限軸
    # r_min / r_max = 許容する可動範囲の生データ値
    # cross: True = 0 と 4095 の境界をまたぐ仕様（連続空間へ拡張するための処理対象）
    1: {"type": "bounded", "r_min": 2850, "r_max": 4096 + 1250, "cross": True},  # ID1: 2850 ~ 5346 (4095→0跨ぎ)
    2: {"type": "bounded", "r_min": 1715, "r_max": 4096 + 10,   "cross": True},  # ID2: 1715 ~ 4106 (境界跨ぎ有)
    3: {"type": "bounded", "r_min": 900,  "r_max": 3100,       "cross": False}, # ID3: 900 ~ 3100 (跨ぎ無)
    4: {"type": "bounded", "r_min": 1650, "r_max": 4015,       "cross": False}, # ID4: 1650 ~ 4015 (跨ぎ無)
    
    # type: "infinite" = 配線の許す限り無限に回転する軸（差分追従モード）
    5: {"type": "infinite", "init": 1930},                                      # ID5: 無限回転軸
    
    6: {"type": "bounded", "r_min": 1990, "r_max": 3000,       "cross": False}, # ID6: 1990 ~ 3000 (跨ぎ無)
}

# フォロワー側の初期位置（起動時に各サーボへ最初に与える基準値）
OFFSET_FOLLOWER = {
    1: 2048,
    2: 1750,
    3: 3100,
    4: 1645,
    5: 1930,
    6: 1990,
}


# ==========================================
# 3. 通信ヘルパー関数（シリアル通信の低レイヤー処理）
# ==========================================
def read_servo_position(ser, servo_id):
    """
    指定したサーボIDに対して現在位置の読み取りパケットを送信し、
    返ってきたレスポンスから位置の生データ（Raw Value: 0〜4095）を抽出する関数
    """
    addr = 0x38       # 現在位置が格納されているレジスタアドレス
    read_len = 2      # 読み取るバイト数（2バイト = 16ビット）
    length = 4        # パケット長
    # チェックサムの計算（通信エラー検知用）
    checksum = ~(servo_id + length + 0x02 + addr + read_len) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x02, addr, read_len, checksum])
    
    ser.write(packet)
    time.sleep(0.001) # 応答待ちの微小ウェイト
    response = ser.read(8) # サーボからの返信データを受信
    
    # レスポンスのヘッダとIDが一致しているか検証し、下位・上位バイトを結合して数値を復元
    if len(response) >= 7 and response[2] == servo_id:
        return response[5] | (response[6] << 8)
    return None # 読み取り失敗時


def write_follower_position(ser, servo_id, pos):
    """
    フォロワー側のサーボへ目標位置（Target）の書き込みパケットを送信する関数
    安全のため、指令値は強制的に対象範囲（0〜4095）にクランプ（制限）します
    """
    pos = int(max(0, min(4095, pos)))
    
    addr = 0x2A       # 目標位置（Goal Position）を書き込むレジスタアドレス
    length = 5        # パケット長
    pos_l = pos & 0xFF        # 下位8ビット
    pos_h = (pos >> 8) & 0xFF # 上位8ビット
    # チェックサム計算
    checksum = ~(servo_id + length + 0x03 + addr + pos_l + pos_h) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, pos_l, pos_h, checksum])
    
    ser.write(packet)
    time.sleep(0.001)


def set_torque(ser, servo_id, enable):
    """
    サーボモーターのトルク（モーターの保持力）をON/OFFする関数
    enable = True で脱力解除（手動で動かせる/保持する）、Falseで脱力
    """
    addr = 0x28       # トルク制御用レジスタアドレス
    length = 4
    en_val = 1 if enable else 0
    checksum = ~(servo_id + length + 0x03 + addr + en_val) & 0xFF
    packet = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr, en_val, checksum])
    
    ser.write(packet)
    time.sleep(0.002)


# ==========================================
# 4. 座標変換・同期ロジック
# ==========================================
def calculate_target(sid, raw_leader, prev_raw_cache, follower_current_cache):
    """
    リーダーの生値（raw_leader）を受け取り、
    有限軸なら「線形連続マッピング」、無限軸なら「差分相対追従」を行って
    フォロワー向けの安全な目標値を計算して返す関数
    """
    config = JOINT_CONFIG[sid]
    direction = DIRECTION[sid]
    
    # --- パターンA: 有限軸の処理（線形連続マッピング） ---
    if config["type"] == "bounded":
        r_min = config["r_min"]
        r_max = config["r_max"]
        cross = config["cross"]
        
        raw = raw_leader
        # 0/4095の境界を跨ぐ軸の場合、小さな値（0付近）にいるときは4096を足して連続した数値に拡張する
        if cross and raw < (r_max - 4096):
            raw += 4096
            
        # 範囲が同一の場合のゼロ除算防止
        if r_max == r_min:
            ratio = 0.0
        else:
            # 最小〜最大値の範囲における「現在の位置の割合（0.0〜1.0）」を算出
            ratio = (raw - r_min) / (r_max - r_min)
            
        # はみ出しを防ぐため割合を0.0から1.0の間にクランプ
        ratio = max(0.0, min(1.0, ratio))
        
        # フォロワー側のフルレンジ（0〜4095）のスケールに変換
        target = ratio * 4095
        
        # 逆転設定（DIRECTION == -1）の場合は向きを反転させる
        if direction == -1:
            target = 4095 - target
            
        return int(target)

    # --- パターンB: 無限回転軸の処理（差分・相対追従） ---
    elif config["type"] == "infinite":
        # 前回取得したリーダーの値を取得（初回は現在の値をそのまま代入）
        prev_raw = prev_raw_cache.get(sid, raw_leader)
        # 今回と前回の差分（変位量）を計算
        diff = raw_leader - prev_raw
        
        # 0/4095の境界を跨いだ瞬間における差分の数学的破綻を防ぐための補正
        if diff > 2048:
            diff -= 4096
        elif diff < -2048:
            diff += 4096
            
        # フォロワーの前回目標値に、リーダーの移動差分を足し込む
        current_target = follower_current_cache.get(sid, config["init"])
        new_target = current_target + (diff * direction)
        
        return int(new_target)


# ==========================================
# 5. メインループ（全体の制御フロー）
# ==========================================
def main():
    try:
        # シリアルポートのオープン（タイムアウトを0.01秒に設定して高速化）
        ser_leader = serial.Serial(LEADER_PORT, BAUDRATE, timeout=0.01)
        ser_follower = serial.Serial(FOLLOWER_PORT, BAUDRATE, timeout=0.01)
        
        print(f"✅ Leader ({LEADER_PORT}) & Follower ({FOLLOWER_PORT}) 接続成功")
        print(f"🎯 テスト対象ID: {TEST_IDS}\n")
        
        # 通信バッファのゴミデータをクリア
        ser_leader.reset_input_buffer()
        ser_follower.reset_input_buffer()

        # フォロワー側のトルク設定
        # TEST_IDSに含まれているモータはトルクON、それ以外は安全のためトルクOFFにする
        for sid in SERVO_IDS:
            if sid in TEST_IDS:
                set_torque(ser_follower, sid, True)
            else:
                set_torque(ser_follower, sid, False)
            
        print("🚀 詳細モニタリング付き同期制御を開始します。[Ctrl + C] で停止\n")

        # 状態保持用キャッシュの初期化
        prev_leader_cache = {}
        follower_current_cache = {sid: JOINT_CONFIG[sid].get("init", 2048) for sid in SERVO_IDS}

        # メインの無限ループ（約50Hzで循環）
        while True:
            log_msgs = []
            for sid in SERVO_IDS:
                # 1. リーダー（COM3）から各サーボの生値（Raw）を読み取る
                raw_pos = read_servo_position(ser_leader, sid)
                
                if raw_pos is not None:
                    # テスト対象に含まれている軸のみ計算と書き込みを実行
                    if sid in TEST_IDS:
                        # 2. 補正・マッピング後の目標値（Target）を計算
                        target_pos = calculate_target(sid, raw_pos, prev_leader_cache, follower_current_cache)
                        
                        # 3. フォロワー（COM4）へ目標位置を書き込む
                        write_follower_position(ser_follower, sid, target_pos)
                        
                        # 次回ループ用のキャッシュを更新
                        prev_leader_cache[sid] = raw_pos
                        follower_current_cache[sid] = target_pos
                    
                    # 4. フォロワー側の実際の現在位置（Foll）を読み取って検証
                    foll_pos = read_servo_position(ser_follower, sid)
                    foll_str = str(foll_pos) if foll_pos is not None else "N/A"
                    
                    # ターミナル表示用のログ文字列を組み立て
                    if sid in TEST_IDS:
                        target_val = follower_current_cache[sid]
                        log_msgs.append(f"[ID{sid}] Raw:{raw_pos:4d} | Target:{target_val:4d} | Foll:{foll_str}")
                    else:
                        log_msgs.append(f"[ID{sid}] (OFF)")
                else:
                    log_msgs.append(f"[ID{sid}] N/A")

            # ターミナル上に1行で全軸の状況（Raw, Target, Foll）をリアルタイム表示
            print("\r" + "  ".join(log_msgs), end="", flush=True)
            time.sleep(0.02) # 制御周期の調整（約50Hz）

    except KeyboardInterrupt:
        # [Ctrl + C] が押されたときの安全停止処理
        print("\n\n🛑 停止しました。Follower のトルクをオフにします。")
        if 'ser_follower' in locals() and ser_follower.is_open:
            for sid in SERVO_IDS:
                set_torque(ser_follower, sid, False)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
    finally:
        # プログラム終了時のシリアルポート解放
        if 'ser_leader' in locals() and ser_leader.is_open:
            ser_leader.close()
        if 'ser_follower' in locals() and ser_follower.is_open:
            ser_follower.close()
        print("✅ ポートをクローズしました。")

if __name__ == "__main__":
    main()