"""
==============================================================================
TACHIKOMA ビジョン・LLM統合対話エージェント (自律リトライ＆机上再認識ループ版)
(src/tachikoma_agent_3.py)
==============================================================================
【概要】
1. カメラ画像から机上の全物体を OpenCV で計測し、VLM (Gemini/Qwen) で属性を一括同定。
2. 統合ワールドステート (current_world_state.json) を動的構築。
3. ユーザーの自然言語指示を受け取り、LLM (Gemini) が搬送タスク (Pick/Place) を立案。
4. 3Dシミュレータ (MuJoCo) でプレビュー再生 ➔ ユーザー承認 ('y') 後に実機実行。
5. 実機実行時は爪サーボ (ID 6) による空振り検知と 6段階インテリジェント・リトライを適用。
6. タスク完了（または失敗スキップ）ごとに、机上の物体配置を自動で再認識・再同定し、
   デジタルツインとワールドステートを最新状態にリフレッシュ 。

【実行コマンド例】
  - 画面シミュレーションプレビューのみ（実機なしテスト）:
      python src/tachikoma_agent_3.py

  - 実機フォロワー接続モード（承認後に実機が動作）:
      python src/tachikoma_agent_3.py --arm  
==============================================================================
"""

import sys
import os
import time
import math
import json
import random
import argparse
import threading
from contextlib import contextmanager
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, List, Any

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
from core.vision_projector import VisionProjector
from core.tabletop_detector import TabletopDetector
from core.multimodal_tagger import MultimodalTagger
from core.llm_planner import LLMTaskPlanner
from core.trajectory_executor import TrajectoryExecutor
from core.kinematics import (
    get_home_radians,
    solve_ik_tabletop_grasp,
    solve_ik_tabletop_place,
    calculate_sag_compensation,
    radian_to_raw,
    raw_to_radian,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)

try:
    from core.bus_servo_controller import BusServoController
    HAS_HARDWARE_MODULE = True
except ImportError:
    HAS_HARDWARE_MODULE = False

# パラメータ設定
WORLD_STATE_PATH = os.path.join(BASE_DIR, "config", "current_world_state.json")
MANUAL_OFFSET_MAJOR_MM = -15.0
MANUAL_OFFSET_MINOR_MM = -50.0
PLACE_X_MM = 180.0
PLACE_Y_MM = 160.0
PLACE_Z_MM = 20.0
PLACE_ANGLE_DEG = 0.0
PICK_Z_MM = 2.0

MAX_RETRIES_PER_OBJECT = 6
GRIPPER_CLOSED_RAW = 1889
EMPTY_GRASP_TOLERANCE_RAW = 30
MAX_SLOTS = 16
HALF_Z = 0.015 / 2.0


# ==============================================================================
# ユーティリティ: 経過時間リアルタイム表示コンテキスト
# ==============================================================================
@contextmanager
def task_progress_timer(label: str = "処理中"):
    """コンソール上で経過秒数をリアルタイムに上書き更新表示するユーティリティ"""
    stop_event = threading.Event()
    start_time = time.time()

    def _worker():
        while not stop_event.wait(0.2):
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r   ⏳ [{label}] 経過時間: {elapsed:4.1f} 秒...")
            sys.stdout.flush()

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop_event.set()
        th.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


# ==============================================================================
# 机上画像スキャン ＆ ワールドステート動的構築ルーチン
# ==============================================================================
def scan_and_rebuild_world_state(
    cap: cv2.VideoCapture,
    projector: VisionProjector,
    detector: TabletopDetector,
    tagger: MultimodalTagger,
    use_gemini: bool = True,
    use_qwen: bool = True
) -> Dict[str, Any]:
    """
    カメラからワークスペースを撮影し、OpenCV計測とVLM同定を実行して最新のWorld Stateを生成・保存する 。
    """
    print("\n📸 机上の最新状態をスキャン中...")
    
    # フレーム安定化のための読み飛ばし
    for _ in range(5):
        cap.read()
        time.sleep(0.03)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ カメラフレームの取得に失敗しました。")
        return {"total_objects": 0, "objects": [], "workspace": {"place_area_xy_mm": [PLACE_X_MM, PLACE_Y_MM]}}

    if projector.homography_mat is None:
        projector.update_homography(frame) 

    warped = projector.warp_to_topdown(frame, out_w=500, out_h=500) 
    if warped is None:
        print("⚠️ ArUco マーカー検出による正射影変換に失敗しました。")
        return {"total_objects": 0, "objects": [], "workspace": {"place_area_xy_mm": [PLACE_X_MM, PLACE_Y_MM]}}

    # 幾何輪郭検出
    detected_objs, _ = detector.detect_objects(warped) 
    if not detected_objs:
        print("ℹ️ 机上に物体は検出されませんでした。")
        world_state = {
            "timestamp": time.time(),
            "total_objects": 0,
            "objects": [],
            "workspace": {"place_area_xy_mm": [PLACE_X_MM, PLACE_Y_MM]}
        }
        with open(WORLD_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(world_state, f, ensure_ascii=False, indent=2)
        return world_state

    # VLM による一括同定 (プログレス表示付き) 
    crops = [obj["crop"] for obj in detected_objs if obj["crop"].size > 0] 
    print(f"🔍 検出された {len(crops)} 個の物体を VLM で同定します...")
    with task_progress_timer("VLM 物体同定リクエスト中"):
        object_profiles = tagger.identify_items(crops, use_gemini=use_gemini, use_qwen=use_qwen) 

    # ワールドステート JSON 構築
    world_objects = []
    print("\n📦 === 現在の机上物体同定結果 ===")
    for idx, obj in enumerate(detected_objs):
        prof = object_profiles.get(idx, {}) 
        x_mm, y_mm = obj["phys_xy"]
        major_mm, minor_mm = obj["size_mm"]
        dist_to_base = math.hypot(x_mm, y_mm)
        display_name = prof.get("display_name", f"object_{idx}") 

        print(f"  [#{idx}] {display_name:<22} | 座標: ({x_mm:+5.1f}, {y_mm:+5.1f})mm | 寸法: {major_mm:.0f}x{minor_mm:.0f}mm")

        world_objects.append({
            "id": idx,
            "display_name": display_name,
            "category": prof.get("category", "object"), 
            "color": prof.get("color", ""), 
            "description": prof.get("description", ""), 
            "physical": {
                "position_xy_mm": [round(x_mm, 1), round(y_mm, 1)],
                "size_mm": [round(major_mm, 1), round(minor_mm, 1)],
                "angle_deg": round(obj["angle_deg"], 1),
                "aspect_ratio": round(major_mm / max(1.0, minor_mm), 2)
            },
            "spatial": {
                "relative_position": ("手前" if x_mm < 250 else "奥") + ("左" if y_mm < -50 else "右" if y_mm > 50 else "中央"),
                "distance_to_base_mm": round(dist_to_base, 1)
            }
        })

    world_state = {
        "timestamp": time.time(),
        "total_objects": len(world_objects),
        "objects": world_objects,
        "workspace": {
            "place_area_xy_mm": [PLACE_X_MM, PLACE_Y_MM]
        }
    }

    with open(WORLD_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(world_state, f, ensure_ascii=False, indent=2)
    print(f"📄 机上状態を保存しました: {WORLD_STATE_PATH}")
    return world_state


# ==============================================================================
# シミュレータプレビュー同期ルーチン
# ==============================================================================
def preview_task_in_sim(sim: MujocoSimViewer, executor: TrajectoryExecutor, target_obj: dict):
    """MuJoCo シミュレータ上で把持〜配置の軌道を先行プレビュー再生する"""
    x_mm, y_mm = target_obj["physical"]["position_xy_mm"]
    major_mm, minor_mm = target_obj["physical"]["size_mm"]
    angle_deg = target_obj["physical"]["angle_deg"]

    ik_grasp, ik_pick_wp, _ = solve_ik_tabletop_grasp(
        x_phys_mm=x_mm, y_phys_mm=y_mm, z_phys_mm=PICK_Z_MM,
        angle_deg=angle_deg, obj_thickness_mm=minor_mm,
        gripper_open_rad=GRIPPER_OPEN_RAD, enable_sag_compensation=True,
        offset_major_mm=MANUAL_OFFSET_MAJOR_MM, offset_minor_mm=MANUAL_OFFSET_MINOR_MM,
        verbose=False
    )
    ik_place_target, ik_place_wp, _ = solve_ik_tabletop_place(
        x_phys_mm=PLACE_X_MM, y_phys_mm=PLACE_Y_MM, z_phys_mm=PLACE_Z_MM,
        place_angle_deg=PLACE_ANGLE_DEG, enable_sag_compensation=True,
        verbose=False
    )

    if None in (ik_grasp, ik_pick_wp, ik_place_target, ik_place_wp):
        print("⚠️ [IK 算出不能] プレビュー軌道を生成できませんでした。")
        return False

    home_rad = get_home_radians()  
    pick_wp_open = dict(ik_pick_wp); pick_wp_open[6] = GRIPPER_OPEN_RAD
    pick_grasp_open = dict(ik_grasp); pick_grasp_open[6] = GRIPPER_OPEN_RAD
    pick_grasp_close = dict(ik_grasp); pick_grasp_close[6] = GRIPPER_CLOSE_RAD
    pick_wp_close = dict(ik_pick_wp); pick_wp_close[6] = GRIPPER_CLOSE_RAD
    place_wp_close = dict(ik_place_wp); place_wp_close[6] = GRIPPER_CLOSE_RAD
    place_land_close = dict(ik_place_target); place_land_close[6] = GRIPPER_CLOSE_RAD
    place_land_open = dict(ik_place_target); place_land_open[6] = GRIPPER_OPEN_RAD
    place_wp_open = dict(ik_place_wp); place_wp_open[6] = GRIPPER_OPEN_RAD

    # シミュレータ上のみでアニメーション再生
    executor.move_to_rad(pick_wp_open, duration_sec=1.0, send_to_real=False)
    executor.move_to_rad(pick_grasp_open, duration_sec=0.7, send_to_real=False)
    executor.move_to_rad(pick_grasp_close, duration_sec=0.4, send_to_real=False)
    executor.move_to_rad(pick_wp_close, duration_sec=0.7, send_to_real=False)
    executor.move_to_rad(place_wp_close, duration_sec=1.2, send_to_real=False)
    executor.move_to_rad(place_land_close, duration_sec=0.7, send_to_real=False)
    executor.move_to_rad(place_land_open, duration_sec=0.4, send_to_real=False)
    executor.move_to_rad(place_wp_open, duration_sec=0.7, send_to_real=False)
    executor.move_to_rad(home_rad, duration_sec=1.2, send_to_real=False)
    return True


# ==============================================================================
# 実機実行 ＆ 6段階インテリジェント・リトライルーチン
# ==============================================================================
def execute_real_task_with_retry(
    executor: TrajectoryExecutor,
    controller: Optional[Any],
    target_obj: dict
) -> bool:
    """実機サーボによる Pick & Place と空振り検知時の 6段階戦略的リトライ"""
    x_mm, y_mm = target_obj["physical"]["position_xy_mm"]
    major_mm, minor_mm = target_obj["physical"]["size_mm"]
    angle_deg = target_obj["physical"]["angle_deg"]
    home_rad = get_home_radians()  

    print(f"\n🚀 実機タスク実行開始: [{target_obj['display_name']}]")

    for retry_count in range(MAX_RETRIES_PER_OBJECT):
        cur_major = MANUAL_OFFSET_MAJOR_MM
        cur_minor = MANUAL_OFFSET_MINOR_MM
        cur_z = PICK_Z_MM

        # リトライ段階に応じた摂動の適用
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

        ik_grasp, ik_pick_wp, _ = solve_ik_tabletop_grasp(
            x_phys_mm=x_mm, y_phys_mm=y_mm, z_phys_mm=cur_z,
            angle_deg=angle_deg, obj_thickness_mm=minor_mm,
            gripper_open_rad=GRIPPER_OPEN_RAD, enable_sag_compensation=True,
            offset_major_mm=cur_major, offset_minor_mm=cur_minor,
            verbose=False
        )
        ik_place_target, ik_place_wp, _ = solve_ik_tabletop_place(
            x_phys_mm=PLACE_X_MM, y_phys_mm=PLACE_Y_MM, z_phys_mm=PLACE_Z_MM,
            place_angle_deg=PLACE_ANGLE_DEG, enable_sag_compensation=True,
            verbose=False
        )

        if ik_grasp is None or ik_pick_wp is None or ik_place_target is None or ik_place_wp is None:
            print("⚠️ IK 解が見つかりません。")
            continue

        pick_wp_open = dict(ik_pick_wp); pick_wp_open[6] = GRIPPER_OPEN_RAD
        pick_grasp_open = dict(ik_grasp); pick_grasp_open[6] = GRIPPER_OPEN_RAD
        pick_grasp_close = dict(ik_grasp); pick_grasp_close[6] = GRIPPER_CLOSE_RAD
        pick_wp_close = dict(ik_pick_wp); pick_wp_close[6] = GRIPPER_CLOSE_RAD
        place_wp_close = dict(ik_place_wp); place_wp_close[6] = GRIPPER_CLOSE_RAD
        place_land_close = dict(ik_place_target); place_land_close[6] = GRIPPER_CLOSE_RAD
        place_land_open = dict(ik_place_target); place_land_open[6] = GRIPPER_OPEN_RAD
        place_wp_open = dict(ik_place_wp); place_wp_open[6] = GRIPPER_OPEN_RAD

        # Pick 実行
        executor.move_to_rad(pick_wp_open, duration_sec=1.2, send_to_real=True)
        executor.move_to_rad(pick_grasp_open, duration_sec=0.8, send_to_real=True)
        executor.move_to_rad(pick_grasp_close, duration_sec=0.5, send_to_real=True)
        executor.move_to_rad(pick_wp_close, duration_sec=0.8, send_to_real=True)

        # 把持成否判定 (ID 6 の物理現在値読み取り)
        time.sleep(0.2)
        is_grasped = True
        if executor.is_real_connected and controller and hasattr(controller, 'driver') and controller.driver:
            grip_raw = None
            for _ in range(4):
                grip_raw = controller.driver.read_position(6)
                if grip_raw is not None:
                    break
                time.sleep(0.05)

            if grip_raw is not None:
                raw_diff = abs(grip_raw - GRIPPER_CLOSED_RAW)
                print(f"   🔍 [把持判定] 爪現在 Raw: {grip_raw} (完全閉止 1889 との差: {raw_diff} count)")
                if raw_diff <= EMPTY_GRASP_TOLERANCE_RAW:
                    is_grasped = False
            else:
                is_grasped = False

        if is_grasped:
            print("✨ 物体の把持に成功しました！トレイへ搬送します。")
            executor.move_to_rad(place_wp_close, duration_sec=1.5, send_to_real=True)
            executor.move_to_rad(place_land_close, duration_sec=0.8, send_to_real=True)
            executor.move_to_rad(place_land_open, duration_sec=0.5, send_to_real=True)
            executor.move_to_rad(place_wp_open, duration_sec=0.8, send_to_real=True)
            executor.move_to_home_and_wait(home_rad)
            print("✅ 搬送シーケンスが完了しました。")
            return True
        else:
            print(f"⚠️ 把持空振りを検知 (試行 {retry_count + 1}/{MAX_RETRIES_PER_OBJECT})")
            executor.move_to_rad(pick_wp_open, duration_sec=0.4, send_to_real=True)
            if retry_count < MAX_RETRIES_PER_OBJECT - 1:
                print("⚡ Home を経由せず、上空から即座にオフセット摂動をかけて再試行します...")
                time.sleep(0.2)
            else:
                print("🛑 連続失敗上限に達しました。Home へ戻り、このタスクを終了します。")
                executor.move_to_home_and_wait(home_rad)
                return False

    return False


# ==============================================================================
# メイン対話ループ
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="TACHIKOMA ビジョン・LLM統合対話エージェント")
    parser.add_argument("--arm", action="store_true", help="実機フォロワー接続を有効化")  
    args = parser.parse_args()

    print("==================================================")
    print(" 🤖 TACHIKOMA ビジョン・LLM統合対話エージェント (Agent 3)")
    print(f" ⚙️ 実機実行モード: {'有効 (--arm)' if args.arm else 'シミュレーションプレビューのみ'}")  
    print(" 👁️ 視覚システム  : OpenCV 幾何計測 ＋ VLM セマンティック同定")
    print(" 🧠 プランナー    : Gemini API ワールドステート推論")
    print("==================================================")

    # 1. ハードウェア & シミュレータの初期化
    controller = None
    if args.arm and HAS_HARDWARE_MODULE:
        try:
            controller = BusServoController()
            if controller.connect():
                print("✅ 実機サーボコントローラに接続成功しました。")
            else:
                controller = None
        except Exception as e:
            print(f"⚠️ 実機接続エラー: {e}")
            controller = None

    sim = MujocoSimViewer()
    home_rad = get_home_radians()  
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass
    sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    executor = TrajectoryExecutor(sim=sim, servo_controller=controller)
    if executor.is_real_connected:
        print("🤖 実機を初期 Home 姿勢へ移動中...")
        executor.move_to_home_and_wait(home_rad)

    # 2. ビジョンモジュール初期化
    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = TabletopDetector(projector)
    tagger = MultimodalTagger()
    planner = LLMTaskPlanner()

    # 起動時の初期スキャン
    world_state = scan_and_rebuild_world_state(cap, projector, detector, tagger)

    try:
        while True:
            user_msg = input("\n🗣️ 指示を入力 (再スキャン: r / 終了: q) > ").strip()  
            if user_msg.lower() in ['q', 'quit', 'exit']:  
                break
            if user_msg.lower() == 'r':
                world_state = scan_and_rebuild_world_state(cap, projector, detector, tagger)
                continue
            if not user_msg:  
                continue

            # LLM プランニング (プログレスタイマー付き)
            with task_progress_timer("Gemini 指示解析＆タスク立案中"):
                plan = planner.plan(user_msg, world_state=world_state)

            print(f"\n💬 応答: {plan.get('reply_text')}")  
            print(f"💡 ログ: {plan.get('thought')}")  

            tasks = plan.get("tasks") or []  
            if not tasks:  
                print("ℹ️ 物理マニピュレーション不要と判断しました。待機状態を維持します。")  
                continue

            print(f"\n📋 【生成された搬送タスク: 全 {len(tasks)} 件】")  
            for idx, task in enumerate(tasks, start=1):  
                p = task.get('pick', {})
                d = task.get('place', {})
                print(
                    f"   {idx}. {task.get('description', '')}\n"  
                    f"      Pick : r={p.get('r', 0)*100:4.1f}cm, θ={p.get('theta_deg', 0):+5.1f}°, z={p.get('z', 0)*1000:4.1f}mm\n"  
                    f"      Place: r={d.get('r', 0)*100:4.1f}cm, θ={d.get('theta_deg', 0):+5.1f}°, z={d.get('z', 0)*1000:4.1f}mm"  
                )

            # 対象物体の特定 (target_id を優先、なければ最初の物体)
            target_id = tasks[0].get("target_id")
            target_obj = None
            if target_id is not None:
                for obj in world_state.get("objects", []):
                    if obj["id"] == target_id:
                        target_obj = obj
                        break
            if target_obj is None and world_state.get("objects"):
                target_obj = world_state["objects"][0]

            if target_obj is None:
                print("⚠️ タスク対象の物体データが机上に見当たりません。")
                continue

            # 3. 3D シミュレータプレビュー  
            print("\n🖥️ 3Dシミュレータでプレビューを再生します...")  
            preview_success = preview_task_in_sim(sim, executor, target_obj)

            # 4. 実機承認実行  
            if executor.is_real_connected and preview_success:
                confirm = input("\n❓ このシーケンスを実機フォロワーで実行しますか？ [y/N] > ").strip().lower()  
                if confirm == 'y':  
                    execute_real_task_with_retry(executor, controller, target_obj)
                    # 👉 【タスク完了後の机上自動再スキャン】
                    print("\n🔄 物体配置の変化を検出するため、机上を自動再スキャンします...")
                    world_state = scan_and_rebuild_world_state(cap, projector, detector, tagger)
                else:
                    print("🛑 実機実行をキャンセルしました。")  
            else:
                if not executor.is_real_connected:
                    print("ℹ️ 実機未接続のためプレビューのみで完了しました。")

    except KeyboardInterrupt:
        print("\n\n終了します。")  
    finally:
        if controller is not None:
            controller.close()
            print("✅ フォロワーポートをクローズしました。")  
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()