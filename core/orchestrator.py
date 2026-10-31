"""
==============================================================================
TACHIKOMA 統括オーケストレーター (core/orchestrator.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムの Layer 2 (司令塔・オーケストレーション層) に位置し、
自然言語指示の解析、タスクプランニング、各ステップの幾何学的検証 (IK Pre-check)、
モーションCSV生成、実機/シミュレータ動作の調停、およびタスク完了後の
机上再スキャン (Closed-loop Vision) を一貫して統括します。

【所属レイヤー】
Layer 2: 司令塔・オーケストレーション層 (Core Orchestrator)

【アーキテクチャ連携】
- 上流: interfaces.discord_bot (または CLI) から指示テキストを受信
- コア: core.llm_planner.LLMTaskPlanner によるマルチステップ分解と発話生成
- 視覚: services.arm_worker.vision_pipeline.VisionPipeline による机上スキャン
- 物理: services.arm_worker.motion_executor.MotionExecutor による搬送制御・リトライ・CSV出力
==============================================================================
"""

import os
import sys
import time
import math
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 依存モジュール (元の実動インポート関係を完全維持)
from config.model_config import get_candidate_models
from core.llm_planner import LLMTaskPlanner
from services.arm_worker.vision_pipeline import VisionPipeline
from services.arm_worker.motion_executor import MotionExecutor
from services.arm_worker.kinematics import (
    solve_ik_tabletop_grasp,
    solve_ik_tabletop_place,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD
)

# 定数
WORLD_STATE_PATH = os.path.join(BASE_DIR, "config", "current_world_state.json")
MOTIONS_DIR = os.path.join(BASE_DIR, "motions")


def _log(tag: str, msg: str, level: str = "INFO"):
    """ターミナル向け統一フォーマットログ出力"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = {
        "INFO": "ℹ️",
        "OK": "✅",
        "WARN": "⚠️",
        "ERR": "❌",
        "DEBUG": "🔍",
        "STEP": "🚀",
    }
    sym = symbols.get(level, "•")
    print(f"[{ts}] {sym} [{tag}] {msg}", flush=True)


class TachikomaOrchestrator:
    """
    TACHIKOMA システム全体の対話・タスク計画・動作実行を統括する司令塔クラス
    """
    def __init__(
        self,
        vision_pipeline: Optional[VisionPipeline] = None,
        motion_executor: Optional[MotionExecutor] = None,
        llm_planner: Optional[LLMTaskPlanner] = None,
        world_state_path: str = WORLD_STATE_PATH
    ):
        """
        Args:
            vision_pipeline: 視覚認識パイプライン
            motion_executor: 物理搬送実行クラス
            llm_planner: 自然言語タスクプランナー
            world_state_path: ワールドステートJSON保存先
        """
        _log("INIT", "==================================================", "INFO")
        _log("INIT", "TACHIKOMA Orchestrator 初期化開始", "INFO")

        self.world_state_path = world_state_path
        self.vision = vision_pipeline or VisionPipeline(world_state_path=world_state_path, enable_vlm=True)
        _log("INIT", "VisionPipeline ロード完了", "OK")

        self.executor = motion_executor or MotionExecutor(follower_driver=None)
        _log("INIT", "MotionExecutor ロード完了", "OK")

        candidate_models = get_candidate_models()
        self.planner = llm_planner or LLMTaskPlanner(candidate_models=candidate_models)
        current_model = getattr(self.planner, "model_name", "Gemini")
        _log("INIT", f"LLMTaskPlanner ロード完了 [モデル候補: {candidate_models}] (現行: {current_model})", "OK")

        # 承認待ちプランのキャッシュ
        self.pending_plan: Optional[Dict[str, Any]] = None
        _log("INIT", "Orchestrator 初期化完了", "OK")
        _log("INIT", "==================================================", "INFO")

    def refresh_vision(self, image_path: Optional[str] = None, use_gemini: bool = True) -> Tuple[bool, Dict[str, Any], str]:
        """机上カメラ (または指定画像) をスキャンしてワールドステートを更新する"""
        _log("VISION", f"机上スキャン実行要求 (use_gemini={use_gemini}, image_path={image_path})", "STEP")
        start_t = time.perf_counter()

        success, new_state, msg = self.vision.scan_and_sync_world_state(
            image_path=image_path,
            use_gemini=use_gemini
        )

        elapsed = time.perf_counter() - start_t
        total_obj = new_state.get("total_objects", len(new_state.get("objects", [])))
        level = "OK" if success else "WARN"
        _log("VISION", f"スキャン完了 ({elapsed:.2f}秒) | 検出物体数: {total_obj}個 | {msg}", level)

        if total_obj > 0:
            for obj in new_state.get("objects", []):
                _log("VISION", f"  └ [ID #{obj.get('id')}] {obj.get('name')} 座標=(X={obj.get('x_mm'):.1f}, Y={obj.get('y_mm'):.1f}, Z={obj.get('z_mm'):.1f})mm", "DEBUG")

        return success, new_state, msg

    def get_current_world_state(self) -> Dict[str, Any]:
        """現在保存されているワールドステートJSONを取得する"""
        if os.path.exists(self.world_state_path):
            try:
                with open(self.world_state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log("WORLD_STATE", f"ワールドステート読み込み失敗: {e}", "WARN")
        return {"total_objects": 0, "objects": [], "workspace": {"place_area_xy_mm": [180.0, 160.0]}}

    def process_instruction(
        self,
        user_text: str,
        auto_precheck: bool = True
    ) -> Dict[str, Any]:
        """
        ユーザーの自然言語指示を処理し、会話応答とタスク計画を立案する。
        """
        world_state = self.get_current_world_state()
        _log("ORCH", "--------------------------------------------------", "INFO")
        _log("ORCH", f"ユーザー指示を受信: '{user_text}'", "STEP")

        # 1. LLM プランニング
        current_model = getattr(self.planner, "model_name", "Gemini")
        _log("LLM", f"自然言語推論開始 [使用Geminiモデル: {current_model}]", "INFO")
        start_llm = time.perf_counter()

        plan_result = self.planner.plan(user_instruction=user_text, world_state=world_state)
        elapsed_llm = time.perf_counter() - start_llm

        reply_text = plan_result.get("reply_text", "了解いたしました！")
        thought = plan_result.get("thought", "")
        tasks = plan_result.get("tasks", [])

        _log("LLM", f"LLM 応答完了 ({elapsed_llm:.2f}秒) | 立案サブタスク: {len(tasks)}件", "OK")
        if thought:
            _log("LLM_THOUGHT", f"思考ログ: {thought}", "DEBUG")
        _log("LLM_REPLY", f"発話テキスト: \"{reply_text}\"", "INFO")

        response = {
            "reply_text": reply_text,
            "thought": thought,
            "tasks": tasks,
            "ik_feasible": True,
            "preview_csvs": [],
            "error": plan_result.get("error")
        }

        # 物理動作が不要な場合 (雑談・質問など)
        if not tasks:
            _log("ORCH", "物理動作を伴わない雑談・応答タスクとして完了", "INFO")
            _log("ORCH", "--------------------------------------------------", "INFO")
            self.pending_plan = None
            return response

        # 2. IK 事前検証 ＆ プレビュー用モーションCSV出力
        if auto_precheck:
            _log("PRECHECK", f"全 {len(tasks)} 件のサブタスクに対する幾何学的 IK 事前検証を開始", "STEP")
            feasible, preview_csvs, ik_msg = self.precheck_and_export_preview(tasks, world_state)
            response["ik_feasible"] = feasible
            response["preview_csvs"] = preview_csvs

            if not feasible:
                _log("PRECHECK", f"❌ IK 事前検証 失敗: {ik_msg}", "ERR")
                response["reply_text"] += f"\n⚠️ 【幾何計算警告】一部の配置目標がアームの可動限界外です: {ik_msg}"
                self.pending_plan = None
                _log("ORCH", "--------------------------------------------------", "INFO")
                return response
            else:
                _log("PRECHECK", f"✅ 全タスクの幾何学 IK 成立を確認 (プレビューCSV: {len(preview_csvs)}件生成)", "OK")

        # 承認待ちプランとしてキャッシュ
        self.pending_plan = {
            "instruction": user_text,
            "tasks": tasks,
            "world_state": world_state,
            "preview_csvs": response["preview_csvs"],
            "timestamp": time.time()
        }
        _log("ORCH", "承認待ちプランをキャッシュしました。ユーザー実行承認を待機します。", "OK")
        _log("ORCH", "--------------------------------------------------", "INFO")

        return response

    def precheck_and_export_preview(
        self,
        tasks_list: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> Tuple[bool, List[str], str]:
        """
        全タスクの逆運動学 (IK) 到達性を検証し、motions/ 配下に検証用CSVを保存する。
        """
        objects = world_state.get("objects", [])
        preview_csvs = []

        for idx, task in enumerate(tasks_list, start=1):
            tid = task.get("target_id")
            _log("IK_CHECK", f"--- [ステップ {idx}/{len(tasks_list)}] 検証開始 (Target ID: {tid}) ---", "INFO")

            t_obj = next((o for o in objects if o.get("id") == tid), None)
            if t_obj is None:
                desc = task.get("description", "")
                t_obj = next((o for o in objects if o.get("name") in desc), None)

            if t_obj is None:
                err_msg = f"ステップ {idx}: 対象物体 (ID: {tid}) が見つかりません。"
                _log("IK_CHECK", f"❌ {err_msg}", "ERR")
                return False, preview_csvs, err_msg

            # 物理座標抽出
            phys = t_obj.get("physical", {})
            pos_xy = phys.get("position_xy_mm", [t_obj.get("x_mm", 200.0), t_obj.get("y_mm", 0.0)])
            size_mm = phys.get("size_mm", [30.0, 30.0])
            angle_deg = phys.get("angle_deg", t_obj.get("angle_deg", 0.0))

            x_mm, y_mm = float(pos_xy[0]), float(pos_xy[1])
            minor_mm = float(size_mm[1])
            dist_pick = math.hypot(x_mm, y_mm)

            _log("IK_CHECK", f"  ├ Pick対象: [{t_obj.get('name')}] 目標=(X={x_mm:.1f}, Y={y_mm:.1f})mm | 距離={dist_pick:.1f}mm | 傾き={angle_deg:.1f}°", "DEBUG")

            # Pick IK 検証
            ik_grasp, ik_pick_wp, _ = solve_ik_tabletop_grasp(
                x_phys_mm=x_mm, y_phys_mm=y_mm, z_phys_mm=10.0,
                angle_deg=angle_deg, obj_thickness_mm=minor_mm,
                gripper_open_rad=GRIPPER_OPEN_RAD, enable_sag_compensation=True,
                verbose=True  # kinematics側の詳細トレースをターミナルに展開
            )
            if ik_grasp is None or ik_pick_wp is None:
                fail_msg = f"ステップ {idx}: Pick目標 ({x_mm:.1f}, {y_mm:.1f}, 距離={dist_pick:.1f}mm) への到達解がありません。"
                _log("IK_CHECK", f"  └ ❌ Pick IK 失敗: {fail_msg}", "ERR")
                return False, preview_csvs, fail_msg
            _log("IK_CHECK", "  ├ ✅ Pick IK 解探索 成功", "OK")

            # Place 座標の計算
            place_data = task.get("place", {})
            pl_r_cm = float(place_data.get("r", 0.20)) * 100.0
            pl_th_deg = float(place_data.get("theta_deg", 0.0))
            pl_z_mm = float(place_data.get("z", 0.020)) * 1000.0
            pl_roll_deg = float(place_data.get("wrist_roll_deg", place_data.get("roll_deg", 0.0)))

            pl_r_mm = pl_r_cm * 10.0
            pl_th_rad = math.radians(pl_th_deg)
            pl_x_mm = pl_r_mm * math.cos(pl_th_rad)
            pl_y_mm = pl_r_mm * math.sin(pl_th_rad)

            # Place IK 検証 (机上リリース高度として最低 15.0mm を確保)
            safe_pl_z_mm = max(15.0, float(pl_z_mm))
            dist_place = math.hypot(pl_x_mm, pl_y_mm)

            _log("IK_CHECK", f"  ├ Place目標: 目標=(X={pl_x_mm:.1f}, Y={pl_y_mm:.1f}, Z={safe_pl_z_mm:.1f})mm | 距離={dist_place:.1f}mm | ロール={pl_roll_deg:.1f}°", "DEBUG")

            ik_place_tgt, ik_place_wp, _ = solve_ik_tabletop_place(
                x_phys_mm=pl_x_mm, y_phys_mm=pl_y_mm, z_phys_mm=safe_pl_z_mm,
                place_angle_deg=pl_roll_deg, enable_sag_compensation=True,
                verbose=True  # kinematics側の詳細トレースをターミナルに展開
            )
            if ik_place_tgt is None or ik_place_wp is None:
                fail_msg = f"ステップ {idx}: Place目標 (X={pl_x_mm:.1f}, Y={pl_y_mm:.1f}, 距離={dist_place:.1f}mm) への到達解がありません。"
                _log("IK_CHECK", f"  └ ❌ Place IK 失敗: {fail_msg}", "ERR")
                return False, preview_csvs, fail_msg
            _log("IK_CHECK", "  ├ ✅ Place IK 解探索 成功", "OK")

            # プレビュー用モーション CSV 生成
            safe_name = t_obj.get("name", f"obj_{tid}").replace(" ", "_")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(MOTIONS_DIR, f"task_step{idx}_{safe_name}_{timestamp_str}.csv")

            _log("IK_CHECK", f"  └ 📄 プレビュー CSV 生成処理中 -> {os.path.basename(csv_path)}", "DEBUG")
            _, _, saved_csv = self.executor.execute_single_task_with_retry(
                target_obj=t_obj,
                place_x_mm=pl_x_mm,
                place_y_mm=pl_y_mm,
                place_z_mm=pl_z_mm,
                place_angle_deg=pl_roll_deg,
                export_csv=True,
                csv_path=csv_path,
                dry_run=True
            )
            if saved_csv:
                preview_csvs.append(saved_csv)
                _log("IK_CHECK", f"  └ ✅ プレビュー CSV 出力完了: {os.path.basename(saved_csv)}", "OK")

        return True, preview_csvs, "全タスクのIK解成立を確認しました。"

    def execute_approved_plan(
        self,
        on_progress: Optional[Any] = None,
        rescan_after: bool = True
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        ユーザー承認済みのプランを実機 (またはドライラン) でシーケンス実行し、
        完了後に机上を再スキャンしてワールドステートを更新する。
        """
        if not self.pending_plan:
            _log("EXEC", "承認待ちのタスクプランが存在しません", "WARN")
            return False, ["⚠️ 承認待ちのタスクプランが存在しません。"], self.get_current_world_state()

        tasks = self.pending_plan["tasks"]
        world_state = self.pending_plan["world_state"]

        _log("EXEC", "==================================================", "STEP")
        _log("EXEC", f"承認済みプランの実行シーケンスを開始 (全 {len(tasks)} 件)", "STEP")
        start_exec = time.perf_counter()

        # 1. 物理搬送シーケンスの実行
        all_ok, logs, _ = self.executor.execute_plan_sequence(
            tasks_list=tasks,
            world_state=world_state,
            on_step_progress=on_progress,
            export_csv=False
        )

        elapsed_exec = time.perf_counter() - start_exec
        self.pending_plan = None
        _log("EXEC", f"搬送シーケンス完了 ({elapsed_exec:.2f}秒) | 結果: {'成功' if all_ok else '失敗'}", "OK" if all_ok else "ERR")

        # 2. クローズドループ: タスク完了後の机上再スキャン
        updated_state = self.get_current_world_state()
        if rescan_after:
            _log("EXEC", "クローズドループ: タスク完了に伴い机上物体配置を再スキャン中...", "STEP")
            scan_ok, new_state, scan_msg = self.refresh_vision(use_gemini=False)
            logs.append(f"📸 {scan_msg}")
            if scan_ok:
                updated_state = new_state
                _log("EXEC", "再スキャン完了: ワールドステートを最新に更新しました", "OK")

        _log("EXEC", "==================================================", "STEP")
        return all_ok, logs, updated_state

    def trigger_abort(self):
        """緊急停止 (キルスイッチ) をアーム Worker へ発令"""
        self.pending_plan = None
        self.executor.trigger_abort()
        _log("ABORT", "🛑 全システム緊急停止指令を発令しました。", "WARN")


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [TEST] core/orchestrator.py 単体動作 ＆ 調停機能検証")
    print("=" * 70)

    # モック初期化
    orchestrator = TachikomaOrchestrator()

    # 1. 雑談指示テスト (物理タスクなし)
    print("\n1. 雑談・挨拶応答テスト:")
    resp1 = orchestrator.process_instruction("タチコマ、調子はどう？")
    print(f"   タチコマ応答: {resp1['reply_text']}")
    print(f"   タスク数: {len(resp1['tasks'])}")
    assert len(resp1["tasks"]) == 0, "雑談でタスクが生成されてはいけません"
    print("   ✅ 雑談応答テスト: PASS")

    # 2. 物理搬送指示テスト (タスク立案・IK事前検証・CSV出力)
    print("\n2. 搬送タスク立案 ＆ IK事前検証 ＆ CSV出力テスト:")
    mock_test_state = {
        "timestamp": time.time(),
        "updated_at": "2026-10-29 15:00:00",
        "total_objects": 1,
        "objects": [
            {
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
        ],
        "workspace": {"place_area_xy_mm": [180.0, -120.0]}
    }
    with open(WORLD_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(mock_test_state, f, ensure_ascii=False, indent=2)

    resp2 = orchestrator.process_instruction("赤い木製ブロックを右奥に片付けて！")
    print(f"   タチコマ応答: {resp2['reply_text']}")
    print(f"   立案タスク数: {len(resp2['tasks'])}")
    print(f"   IK到達性: {resp2['ik_feasible']}")
    print(f"   出力プレビューCSV: {resp2['preview_csvs']}")

    assert resp2["ik_feasible"], "IK事前検証が成立していません"
    assert len(resp2["preview_csvs"]) > 0, "プレビュー用CSVが出力されていません"
    print("   ✅ タスク立案 ＆ IK検証 ＆ CSVエクスポート: PASS")

    # 3. 承認実行 ＆ クローズドループ再スキャンテスト
    print("\n3. 承認実行シーケンス ＆ 再スキャン検証:")
    done, exec_logs, final_state = orchestrator.execute_approved_plan(rescan_after=False)
    print(f"   実行結果: {done}")
    for log in exec_logs:
        print(f"     - {log}")
    assert done, "承認プランの実行に失敗しました"
    print("   ✅ 承認実行シーケンス: PASS")

    print("\n" + "=" * 70)
    print("🎉 core/orchestrator.py の全単体テストが正常に完了しました！")
    print("=" * 70)