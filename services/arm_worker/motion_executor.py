"""
==============================================================================
TACHIKOMA 物理搬送実行 ＆ 自律リトライ制御モジュール (services/arm_worker/motion_executor.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムの Layer 3 (専門サービス層: 🦾物理制御Worker) に位置し、
立案された Pick & Place タスクの逆運動学 (IK) 解の検証、軌道補間、実機サーボへの物理指令発行、
把持空振り検知時の 6段階インテリジェント・リトライ、緊急停止キルスイッチ、
および【シミュレータ / teleop_replay.py / replay_main.py 連携用のモーションCSVエクスポート機能】
を一括提供します。

【所属レイヤー】
Layer 3: 専門サービス層 (物理制御 Worker: arm_worker)

【主な機能と責務】
1. 搬送タスクの逆運動学 (IK) 幾何学的到達性検証:
   - 物理動作前に各ステップ (Pick / Place) の IK 解を算出し、幾何学的不可能時の早期フェイル。
2. 軌道補間 ＆ 実機サーボへのスムーズな指令送信:
   - 線形・S字補間フレームによる全軸 (ID 1〜6) サーボ角の同期制御。
3. 把持成否判定 ＆ 6段階インテリジェント・リトライ:
   - グリッパー (ID 6) の現在位置 Raw 値を読み取り、完全閉止 (空振り) との差分を計測。
   - 失敗時は「深掘り」「長手(+/-)シフト」「短手引き込み」「広角ジッター」を自動適用。
4. 緊急停止 (キルスイッチ) 割り込み対応:
   - abort_flag を毎フレーム監視し、停止指令受信時は即座に全サーボを脱力 (トルクOFF)。
5. モーションCSVエクスポート (replay_main.py / teleop_replay.py 互換):
   - 計算された全軌道フレームを `motions/` フォルダ配下にタイムスタンプ付きCSVとして出力。
   - 列フォーマット: `timestamp_sec,id_1,id_2,id_3,id_4,id_5,id_6,q1,q2,q3,q4,q5,q6`
     (Raw値・物理ラジアン両対応ハイブリッド形式)。

【ローカル開発・ノートPC対応（モック・ドライラン機構）】
- 実機アーム未接続時は自動的にドライランモードとなり、ハードウェア通信をスキップしつつ
  CSV 生成・保存と IK 検証を安全に完遂します。
==============================================================================
"""

import os
import sys
import time
import math
import random
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ------------------------------------------------------------------------------
# 内部依存モジュールのインポート
# ------------------------------------------------------------------------------
from config.joint_config import FOLLOWER_PORT, BAUDRATE, SERVO_IDS

# 1. kinematics（本物の逆運動学ソルバー）を最優先でインポート
try:
    from services.arm_worker.kinematics import (
        get_home_radians,
        solve_ik_tabletop_grasp,
        solve_ik_tabletop_place,
        radian_to_raw,
        raw_to_radian,
        GRIPPER_OPEN_RAD,
        GRIPPER_CLOSE_RAD,
    )
    print("✅ [MotionExecutor] kinematics (本物ソルバー) の読み込みに成功しました。")
except Exception as e:
    print(f"⚠️ [MotionExecutor] kinematics 読み込み失敗のためモックを使用します: {e}")
    GRIPPER_OPEN_RAD = 0.0
    GRIPPER_CLOSE_RAD = 1.0
    def get_home_radians(): return {sid: 0.0 for sid in SERVO_IDS}
    def solve_ik_tabletop_grasp(*args, **kwargs): return {sid: 0.0 for sid in SERVO_IDS}, {sid: 0.0 for sid in SERVO_IDS}, None
    def solve_ik_tabletop_place(*args, **kwargs): return {sid: 0.0 for sid in SERVO_IDS}, {sid: 0.0 for sid in SERVO_IDS}, None
    def radian_to_raw(sid, rad): return 2048
    def raw_to_radian(sid, raw): return 0.0

# 2. ハードウェアドライバ・軌道生成器（未接続時は None で安全継続）
try:
    from drivers.sts3215 import STS3215Driver
except Exception:
    STS3215Driver = None

try:
    from services.arm_worker.motion_generator import ParametricMotionGenerator
except Exception:
    ParametricMotionGenerator = None

from services.sentinel.system_actions import emergency_stop_arm

# 物理定数・リトライ閾値 (tachikoma_agent_3 準拠)
MANUAL_OFFSET_MAJOR_MM = -15.0
MANUAL_OFFSET_MINOR_MM = -50.0
DEFAULT_PICK_Z_MM = 2.0
DEFAULT_PLACE_Z_MM = 20.0
MAX_RETRIES_PER_OBJECT = 6
GRIPPER_CLOSED_RAW = 1889
EMPTY_GRASP_TOLERANCE_RAW = 30

# モーション出力先ディレクトリ (C:\study\tachikoma\motions 相当)
MOTIONS_DIR = os.path.join(BASE_DIR, "motions")


class MotionExecutor:
    """
    ロボットアームの物理搬送シーケンスとリトライ制御、
    およびシミュレータ用CSVエクスポートを管理する実行クラス
    """
    def __init__(
        self,
        follower_driver: Optional[Any] = None,
        generator: Optional[Any] = None
    ):
        self.follower = follower_driver
        if generator is not None:
            self.generator = generator
        elif ParametricMotionGenerator is not None:
            self.generator = ParametricMotionGenerator()
        else:
            self.generator = None

        self.abort_flag = False

    @property
    def is_connected(self) -> bool:
        if self.follower is None:
            return False
        if hasattr(self.follower, "ser") and self.follower.ser is not None:
            return getattr(self.follower.ser, "is_open", True)
        if hasattr(self.follower, "serial") and self.follower.serial is not None:
            return getattr(self.follower.serial, "is_open", True)
        return getattr(self.follower, "is_open", True)

    def trigger_abort(self):
        """緊急停止フラグを立て、即座に全サーボを脱力させる"""
        self.abort_flag = True
        if self.is_real_connected:
            emergency_stop_arm(self.follower)

    def _check_grasp_success(self) -> Tuple[bool, int]:
        """
        グリッパー (ID 6) の現在 Raw 値を読み取り、把持成否を判定する。
        """
        if not self.is_real_connected:
            return True, 1750

        grip_raw = None
        for _ in range(4):
            try:
                grip_raw = self.follower.read_position(6)
                if grip_raw is not None:
                    break
            except Exception:
                pass
            time.sleep(0.05)

        if grip_raw is None:
            return False, -1

        raw_diff = abs(grip_raw - GRIPPER_CLOSED_RAW)
        is_grasped = (raw_diff > EMPTY_GRASP_TOLERANCE_RAW)
        return is_grasped, grip_raw

    def _interpolate_waypoints_to_frames(
        self,
        start_angles: Dict[int, float],
        waypoints: List[Tuple[Dict[int, float], float]],
        dt: float = 0.02
    ) -> List[Tuple[float, Dict[int, float], Dict[int, int]]]:
        """
        キー姿勢リスト [(目標ラジアン辞書, 遷移秒数), ...] の間を
        コサインS字補間し、(経過秒数, ラジアン辞書, Raw値辞書) のフレーム列を生成する。
        """
        frames = []
        current_rad = dict(start_angles)
        elapsed_time = 0.0

        for target_rad, duration in waypoints:
            steps = max(1, int(duration / dt))
            init_rad = dict(current_rad)

            for step in range(1, steps + 1):
                elapsed_time += dt
                # コサインS字補間 (0.0 -> 1.0)
                t = step / steps
                ratio = (1.0 - math.cos(t * math.pi)) / 2.0

                step_rad = {}
                step_raw = {}
                for sid in SERVO_IDS:
                    r = init_rad[sid] + ratio * (target_rad.get(sid, init_rad[sid]) - init_rad[sid])
                    step_rad[sid] = r
                    step_raw[sid] = radian_to_raw(sid, r)

                current_rad = step_rad
                frames.append((round(elapsed_time, 4), step_rad, step_raw))

        return frames

    def save_frames_to_csv(
        self,
        frames: List[Tuple[float, Dict[int, float], Dict[int, int]]],
        filepath: Optional[str] = None,
        prefix: str = "motion_plan"
    ) -> str:
        """
        補間済みフレーム列を teleop_replay.py / replay_main.py 互換の CSV として保存する。

        Args:
            frames: (経過秒, ラジアン辞書, Raw値辞書) のリスト
            filepath: 出力先CSVパス (未指定時は motions/<prefix>_YYYYMMDD_HHMMSS.csv)
            prefix: ファイル名接頭辞

        Returns:
            str: 実際に保存された絶対パス
        """
        if not filepath:
            os.makedirs(MOTIONS_DIR, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MOTIONS_DIR, f"{prefix}_{timestamp_str}.csv")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        fieldnames = ["timestamp_sec"] + [f"id_{sid}" for sid in SERVO_IDS] + [f"q{sid}" for sid in SERVO_IDS]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t_sec, rad_dict, raw_dict in frames:
                row = {"timestamp_sec": f"{t_sec:.3f}"}
                for sid in SERVO_IDS:
                    row[f"id_{sid}"] = raw_dict[sid]
                    row[f"q{sid}"] = f"{rad_dict[sid]:.5f}"
                writer.writerow(row)

        print(f"📄 [MotionExecutor] モーションCSVを出力しました: {filepath} ({len(frames)} フレーム)")
        return filepath

    def execute_single_task_with_retry(
        self,
        target_obj: Dict[str, Any],
        place_x_mm: float,
        place_y_mm: float,
        place_z_mm: float = DEFAULT_PLACE_Z_MM,
        place_angle_deg: float = 0.0,
        reset_abort: bool = False,
        export_csv: bool = False,
        csv_path: Optional[str] = None,
        dry_run: bool = False  # 追加
    ) -> Tuple[bool, str, Optional[str]]:
        """
        単一物体の Pick & Place タスクを、把持空振り検知と 6段階リトライを交えて実行する。
        export_csv=True が指定された場合は motions/ 配下に動作CSVを出力する。

        Returns:
            Tuple[bool, str, Optional[str]]: (タスク成否, 報告メッセージ, 保存されたCSVパス)
        """
        if reset_abort:
            self.abort_flag = False

        if self.abort_flag:
            return False, "🛑 緊急停止指令を受信しているためタスクを開始せず中断しました。", None

        display_name = target_obj.get("display_name") or target_obj.get("name") or "対象物体"
        phys = target_obj.get("physical", {})
        pos_xy = phys.get("position_xy_mm", [target_obj.get("x_mm", 200.0), target_obj.get("y_mm", 0.0)])
        size_mm = phys.get("size_mm", [30.0, 30.0])
        angle_deg = phys.get("angle_deg", target_obj.get("angle_deg", 0.0))

        x_mm, y_mm = float(pos_xy[0]), float(pos_xy[1])
        major_mm, minor_mm = float(size_mm[0]), float(size_mm[1])

        print(f"\n🚀 [MotionExecutor] タスク開始: '{display_name}' ➔ 目標 ({place_x_mm:+.1f}, {place_y_mm:+.1f}, Z:{place_z_mm:.1f}mm)")

        if self.is_real_connected:
            for sid in SERVO_IDS:
                try:
                    self.follower.set_torque(sid, True)
                except Exception:
                    pass

        home_rad = get_home_radians()
        saved_csv_path = None

        try:
            for retry_count in range(MAX_RETRIES_PER_OBJECT):
                if self.abort_flag:
                    return False, "🛑 緊急停止指令を受信したためタスクを中断しました。", None

                cur_major = MANUAL_OFFSET_MAJOR_MM
                cur_minor = MANUAL_OFFSET_MINOR_MM
                cur_z = DEFAULT_PICK_Z_MM

                if retry_count == 1:
                    cur_z = max(0.0, cur_z - 2.0)
                    print(f"   🔄 [リトライ 1/5] 深掘りアプローチ (Z: {cur_z:.1f}mm)")
                elif retry_count == 2:
                    cur_major += 15.0 + random.uniform(-4.0, 4.0)
                    cur_z = max(0.0, cur_z - 1.5)
                    print(f"   🔄 [リトライ 2/5] 長手(+)シフト (Major: {cur_major:+.1f}mm)")
                elif retry_count == 3:
                    cur_major -= 15.0 + random.uniform(-4.0, 4.0)
                    cur_z = max(0.0, cur_z - 1.5)
                    print(f"   🔄 [リトライ 3/5] 長手(-)シフト (Major: {cur_major:+.1f}mm)")
                elif retry_count == 4:
                    cur_minor += 8.0
                    cur_z = max(0.0, cur_z - 2.0)
                    print(f"   🔄 [リトライ 4/5] 短手引き込み深掘り (Minor: {cur_minor:+.1f}mm)")
                elif retry_count >= 5:
                    cur_major += random.uniform(-10.0, 10.0)
                    cur_minor += random.uniform(-6.0, 6.0)
                    cur_z = max(0.0, cur_z - 1.5)
                    print(f"   🔄 [リトライ 5/5] 広角ジッター探索 (Major: {cur_major:+.1f}mm, Minor: {cur_minor:+.1f}mm)")

                # 逆運動学 (IK) 計算
                ik_grasp, ik_pick_wp = None, None
                try:
                    ik_grasp, ik_pick_wp, _ = solve_ik_tabletop_grasp(
                        x_phys_mm=x_mm, y_phys_mm=y_mm, z_phys_mm=cur_z,
                        angle_deg=angle_deg, obj_thickness_mm=minor_mm,
                        gripper_open_rad=GRIPPER_OPEN_RAD, enable_sag_compensation=True,
                        offset_major_mm=cur_major, offset_minor_mm=cur_minor,
                        verbose=False
                    )
                except Exception as e:
                    print(f"   ⚠️ Pick IK 算出例外: {e}")

                if ik_grasp is None or ik_pick_wp is None:
                    print("   ⚠️ Pick 位置の幾何学的解が見つかりません。オフセットを変更して再試行...")
                    continue

                # Place IK 計算
                ik_place_tgt, ik_place_wp = None, None
                try:
                    ik_place_tgt, ik_place_wp, _ = solve_ik_tabletop_place(
                        x_phys_mm=place_x_mm, y_phys_mm=place_y_mm, z_phys_mm=place_z_mm,
                        place_angle_deg=place_angle_deg, enable_sag_compensation=True,
                        verbose=False
                    )
                except Exception as e:
                    print(f"   ⚠️ Place IK 算出例外: {e}")

                # ウェイポイント列の構築
                # [Home] -> [Pick直上] -> [Pick把持] -> [爪閉じ] -> [Pick直上] -> [Place直上] -> [Place接地] -> [爪開き] -> [Place直上] -> [Home]
                wp_pick_open = dict(ik_pick_wp); wp_pick_open[6] = GRIPPER_OPEN_RAD
                wp_grasp_open = dict(ik_grasp); wp_grasp_open[6] = GRIPPER_OPEN_RAD
                wp_grasp_close = dict(ik_grasp); wp_grasp_close[6] = GRIPPER_CLOSE_RAD
                wp_lift_close = dict(ik_pick_wp); wp_lift_close[6] = GRIPPER_CLOSE_RAD

                seq_waypoints: List[Tuple[Dict[int, float], float]] = [
                    (wp_pick_open, 1.2),
                    (wp_grasp_open, 0.6),
                    (wp_grasp_close, 0.4),
                    (wp_lift_close, 0.6)
                ]

                if ik_place_tgt and ik_place_wp:
                    wp_place_approach = dict(ik_place_wp); wp_place_approach[6] = GRIPPER_CLOSE_RAD
                    wp_place_land = dict(ik_place_tgt); wp_place_land[6] = GRIPPER_CLOSE_RAD
                    wp_place_release = dict(ik_place_tgt); wp_place_release[6] = GRIPPER_OPEN_RAD
                    wp_place_up = dict(ik_place_wp); wp_place_up[6] = GRIPPER_OPEN_RAD

                    seq_waypoints.extend([
                        (wp_place_approach, 1.2),
                        (wp_place_land, 0.6),
                        (wp_place_release, 0.4),
                        (wp_place_up, 0.6),
                        (home_rad, 1.2)
                    ])

                # CSV 出力が有効な場合はフレーム列を補間・出力
                if export_csv:
                    frames = self._interpolate_waypoints_to_frames(home_rad, seq_waypoints, dt=0.02)
                    safe_name = display_name.replace(" ", "_")
                    saved_csv_path = self.save_frames_to_csv(
                        frames,
                        filepath=csv_path,
                        prefix=f"task_{safe_name}"
                    )

                # 実機サーボ書き込み
                if self.is_real_connected:
                    if not dry_run and self.is_connected():
                        for target_rads, move_sec in seq_waypoints:
                            if self.abort_flag:
                                return False, "🛑 中断", saved_csv_path
                            for sid in SERVO_IDS:
                                self.follower.write_position(sid, radian_to_raw(sid, target_rads[sid]))
                            time.sleep(move_sec)

                # 把持成否判定
                is_grasped, grip_raw = self._check_grasp_success()
                if is_grasped:
                    print(f"   ✨ 把持成功! (爪Raw: {grip_raw})")
                    return True, f"✅ '{display_name}' の搬送タスクが完了しました (試行回数: {retry_count + 1})。", saved_csv_path
                else:
                    print(f"   ⚠️ 把持空振りを検知 (爪Raw: {grip_raw}, 試行 {retry_count + 1}/{MAX_RETRIES_PER_OBJECT})")
                    if retry_count >= MAX_RETRIES_PER_OBJECT - 1:
                        return False, f"⚠️ '{display_name}' の把持が連続失敗上限に達しました。", saved_csv_path

            return False, f"⚠️ '{display_name}' のIK解または把持条件を満たせませんでした。", saved_csv_path

        except Exception as e:
            return False, f"❌ 物理搬送中に例外が発生しました: {e}", saved_csv_path
        finally:
            if self.is_real_connected:
                emergency_stop_arm(self.follower)

    def execute_plan_sequence(
        self,
        tasks_list: List[Dict[str, Any]],
        world_state: Dict[str, Any],
        on_step_progress: Optional[Any] = None,
        export_csv: bool = False
    ) -> Tuple[bool, List[str], List[str]]:
        """
        LLM が立案した一連のサブタスクリストを順番に実行する。

        Returns:
            Tuple[bool, List[str], List[str]]: (全タスク完了フラグ, 各ステップの結果ログ, 生成されたCSVパス一覧)
        """
        self.abort_flag = False
        results: List[str] = []
        exported_csvs: List[str] = []
        objects = world_state.get("objects", [])

        if not tasks_list:
            return True, ["ℹ️ 実行すべきサブタスクがありませんでした。"], []

        for idx, task in enumerate(tasks_list, 1):
            if self.abort_flag:
                results.append(f"🛑 ステップ {idx} 開始前に緊急停止指令を受信したため中断しました。")
                return False, results, exported_csvs

            target_id = task.get("target_id")
            t_obj = next((o for o in objects if o.get("id") == target_id), None)
            if t_obj is None:
                desc = task.get("description", "")
                t_obj = next((o for o in objects if o.get("name") in desc), None)

            if t_obj is None:
                msg = f"⚠️ [ステップ {idx}] 対象物体 (ID: {target_id}) が机上に存在しません。スキップします。"
                print(msg)
                results.append(msg)
                continue

            place_data = task.get("place", {})
            pl_r_cm = float(place_data.get("r", 0.20)) * 100.0
            pl_th_deg = float(place_data.get("theta_deg", 0.0))
            pl_z_mm = float(place_data.get("z", 0.020)) * 1000.0
            pl_roll_deg = float(place_data.get("wrist_roll_deg", place_data.get("roll_deg", 0.0)))

            pl_r_mm = pl_r_cm * 10.0
            pl_th_rad = math.radians(pl_th_deg)
            pl_x_mm = pl_r_mm * math.cos(pl_th_rad)
            pl_y_mm = pl_r_mm * math.sin(pl_th_rad)

            if on_step_progress:
                on_step_progress(idx, len(tasks_list), t_obj.get("name", "物体"))

            success, log_msg, csv_p = self.execute_single_task_with_retry(
                target_obj=t_obj,
                place_x_mm=pl_x_mm,
                place_y_mm=pl_y_mm,
                place_z_mm=pl_z_mm,
                place_angle_deg=pl_roll_deg,
                export_csv=export_csv
            )
            results.append(log_msg)
            if csv_p:
                exported_csvs.append(csv_p)

            if not success and self.abort_flag:
                break

            time.sleep(0.5)

        all_ok = all("✅" in r for r in results)
        return all_ok, results, exported_csvs


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [TEST] services/arm_worker/motion_executor.py 単体動作 ＆ CSVエクスポート検証")
    print("=" * 70)

    mock_executor = MotionExecutor(follower_driver=None)

    mock_object = {
        "id": 0,
        "name": "red wooden block",
        "display_name": "red wooden block",
        "x_mm": 220.0,
        "y_mm": 30.0,
        "z_mm": 5.0,
        "angle_deg": 15.0,
        "physical": {
            "position_xy_mm": [220.0, 30.0],
            "size_mm": [40.0, 25.0],
            "angle_deg": 15.0
        }
    }

    # 1. 単一タスク実行 ＆ motions/ へのCSV出力テスト
    print("\n1. 単一タスク実行 ＆ モーションCSVエクスポートテスト:")
    success, msg, out_csv = mock_executor.execute_single_task_with_retry(
        target_obj=mock_object,
        place_x_mm=180.0,
        place_y_mm=-120.0,
        place_z_mm=20.0,
        place_angle_deg=0.0,
        export_csv=True
    )
    print(f"   実行結果: {msg}")
    print(f"   生成CSVパス: {out_csv}")
    assert success, "単一タスクの実行に失敗しました"
    assert out_csv is not None and os.path.exists(out_csv), "CSVファイルが出力されていません"

    # 2. 生成されたCSVファイルのフォーマット確認
    print("\n2. 生成された CSV フォーマット検証:")
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    print(f"   ヘッダー列: {headers}")
    print(f"   総フレーム数: {len(rows)} フレーム")
    assert "timestamp_sec" in headers, "timestamp_sec 列が存在しません"
    assert "id_1" in headers and "q1" in headers, "id_1 または q1 列が存在しません"
    assert len(rows) > 0, "CSV内のフレームデータが空です"
    print("   ✅ CSVフォーマット検証 (replay_main.py互換): PASS")

    # 3. 緊急停止割り込みテスト
    print("\n3. 緊急停止 (abort_flag) 割り込みテスト:")
    mock_executor.trigger_abort()
    aborted_success, abort_msg, _ = mock_executor.execute_single_task_with_retry(
        target_obj=mock_object,
        place_x_mm=180.0,
        place_y_mm=0.0
    )
    print(f"   緊急停止時の応答: {abort_msg}")
    assert not aborted_success, "緊急停止割り込みの検知に失敗しました"
    print("   ✅ キルスイッチ割り込みテスト: PASS")

    print("\n" + "=" * 70)
    print("🎉 services/arm_worker/motion_executor.py の全単体テストが正常に完了しました！")
    print(f"📁 出力確認先: {MOTIONS_DIR}")
    print("=" * 70)