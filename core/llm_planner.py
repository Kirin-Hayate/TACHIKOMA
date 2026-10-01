"""
==============================================================================
TACHIKOMA 自然言語タスクプランナー (core/llm_planner.py)
==============================================================================
【役割】
ユーザーの自然言語指示を入力し、
1. 挨拶や雑談への自然な応答 (reply_text)
2. 机上ワールドステート (current_world_state.json) に基づく把持対象同定と極座標変換 (tasks)
を両立して旧版と完全互換の辞書形式で出力します。

【旧版からの進化点】
- 静的な方位角推測ではなく、カメラ認識＋VLM同定された実際の物体位置 [X, Y]、
  色、外観特徴、アスペクト比を照合して精密な物理座標 [r, theta, z] を自動決定。
- 呼び出しインターフェース (plan メソッド、返り値スキーマ) は旧版と完全互換。

【耐障害フェイルオーバー】
- 複数のGeminiモデル候補を保持し、1つのモデルにつき最大2回試行。
- 404（廃止）や503（一時ビジー）等のエラー発生時は、次の候補モデルへ自動フェイルオーバー。
==============================================================================
"""

import os
import sys
import json
import time
import math
import traceback
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

from config.workspace_config import (
    R_MIN_METERS,
    R_MAX_METERS,
    DEFAULT_Z_TCP
)

WORLD_STATE_PATH = os.path.join(BASE_DIR, "config", "current_world_state.json")


class LLMTaskPlanner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        candidate_models: Optional[List[str]] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY が見つかりません。.env ファイルを確認してください。"
            )
        self.client = genai.Client(api_key=self.api_key)

        # multimodal_tagger.py と同様の優先度リスト
        if candidate_models is None:
            self.candidate_models = [
                "gemini-3.8-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemini-3.7-flash",
                "gemini-3.6-flash"
            ]
        else:
            self.candidate_models = candidate_models

    def _load_current_world_state(self) -> Dict[str, Any]:
        """最新の机上認識 JSON を読み込む"""
        if os.path.exists(WORLD_STATE_PATH):
            try:
                with open(WORLD_STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ ワールドステート読み込み失敗: {e}")
        return {"total_objects": 0, "objects": [], "workspace": {"place_area_xy_mm": [180.0, 160.0]}}

    def plan(self, user_instruction: str, world_state: Optional[Dict[str, Any]] = None) -> dict:
        """
        自然言語指示を入力し、タスクプラン (旧版完全互換辞書) を出力する。
        """
        if world_state is None:
            world_state = self._load_current_world_state()

        system_instruction = (
            "あなたの名前は思考戦車『TACHIKOMA（タチコマ）』です。\n"
            "礼儀正しく、しかし親しみと愛嬌を持って応答します。一人称は『当機』を使用してください。\n\n"
            "【機能概要】\n"
            "あなたには単眼カメラおよびVLMによってリアルタイム認識された机上の物体リスト (world_state) が与えられます。\n"
            "ユーザーからの自然言語指示を解釈し、雑談・挨拶なのか、それとも物体の搬送 (Pick & Place) なのかを判断してください。\n\n"
            "【推論ルール】\n"
            "1. 挨拶・雑談・感謝などの場合:\n"
            "   - 感情豊かに礼儀正しく応答してください。\n"
            "   - tasks リストは必ず空リスト [] にしてください。\n\n"
            "2. 物体の搬送指示の場合:\n"
            "   - ユーザーの曖昧な表現（例: '木製のやつ', '細長いもの', '右にあるもの', '消しゴム'）から、\n"
            "     objects 内の display_name, category, color, description, spatial, physical を総合照合して最も適切な物体を1つ特定してください。\n"
            "   - 物体の物理座標 [X_mm, Y_mm] を以下の計算式でロボット極座標に変換して pick に設定してください:\n"
            "       r = hypot(X_mm, Y_mm) / 1000.0  (単位: m)\n"
            "       theta_deg = degrees(atan2(Y_mm, X_mm))  (単位: 度、正が右/時計回り、負が左/反時計回り)\n"
            "       z = DEFAULT_Z_TCP (通常 0.02m)\n"
            "   - place には workspace.place_area_xy_mm (デフォルト [180.0, 160.0]) の極座標、またはユーザー指定位置を設定してください。\n"
            "   - 指定の物体が見当たらない場合は tasks を [] とし、見つからない旨を reply_text で優しく伝えてください。"
        )

        prompt = f"""
【現在の机上ワールドステート (実機カメラ認識結果)】
{json.dumps(world_state, ensure_ascii=False, indent=2)}

【ロボット可動仕様】
- 可動半径 r: {R_MIN_METERS}m 〜 {R_MAX_METERS}m
- 旋回角 theta_deg: -90.0度 〜 +90.0度
- 高さ z: 基準値 {DEFAULT_Z_TCP}m

【ユーザー入力】
"{user_instruction}"

【出力 JSON フォーマット要件】
{{
  "thought": "指示の解釈、対象物体の特定根拠、座標計算の過程",
  "reply_text": "タチコマらしい丁寧かつ愛嬌のある発話メッセージ (一人称: 当機)",
  "tasks": [
    {{
      "type": "pick_and_place",
      "target_id": 0,
      "description": "タスク要約 (例: 手前右の細長い白い定規を排出トレイへ搬送)",
      "pick": {{"r": 0.243, "theta_deg": 21.6, "z": {DEFAULT_Z_TCP}}},
      "place": {{"r": 0.241, "theta_deg": 41.6, "z": {DEFAULT_Z_TCP}}}
    }}
  ]
}}
※ 搬送不要な会話・挨拶の場合は tasks を必ず [] にしてください。
"""

        # ----------------------------------------------------------------------
        # 多重モデル ＆ 2回リトライ・フォールバックループ
        # ----------------------------------------------------------------------
        last_error = None
        for model_name in self.candidate_models:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            temperature=0.2
                        )
                    )

                    plan_data = json.loads(response.text)
                    if isinstance(plan_data, list):
                        plan_data = {
                            "thought": "シーケンス抽出完了",
                            "reply_text": "了解いたしました！指示のシーケンスを実行いたしますね。",
                            "tasks": plan_data
                        }

                    # 座標値バリデーション ＆ 安全クランプ
                    raw_tasks = plan_data.get("tasks") or []
                    valid_tasks = []

                    for task in raw_tasks:
                        if task.get("type") == "pick_and_place" and "pick" in task and "place" in task:
                            for key in ["pick", "place"]:
                                coord = task[key]
                                coord["r"] = float(max(R_MIN_METERS, min(R_MAX_METERS, coord.get("r", 0.25))))
                                coord["theta_deg"] = float(max(-90.0, min(90.0, coord.get("theta_deg", 0.0))))
                                coord["z"] = float(max(0.005, min(0.100, coord.get("z", DEFAULT_Z_TCP))))
                            valid_tasks.append(task)

                    plan_data["tasks"] = valid_tasks
                    return plan_data

                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    # 404 (廃止モデル) の場合は再試行せず即座に次のモデルへ
                    if "404" in err_msg or "NOT_FOUND" in err_msg:
                        break
                    # 一時ビジー (503 等) の場合は少し待ってから同モデルで再試行
                    if attempt == 0 and ("503" in err_msg or "UNAVAILABLE" in err_msg):
                        time.sleep(1.0)
                    else:
                        break

        # 全モデルが失敗した場合のエラーハンドリング
        if last_error:
            traceback.print_exc()
        return {
            "thought": "全候補モデルでの推論試行に失敗しました。",
            "reply_text": "通信エラーが発生したようです……もう一度指示をいただけますでしょうか！",
            "tasks": [],
            "error": str(last_error) if last_error else "Unknown error"
        }