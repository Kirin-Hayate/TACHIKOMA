"""
==============================================================================
TACHIKOMA 自然言語タスクプランナー (マルチステップ・タスク分解対応版) (core/llm_planner.py)
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
            raise ValueError("❌ GEMINI_API_KEY が見つかりません。.env を確認してください。")
        self.client = genai.Client(api_key=self.api_key)

        if candidate_models is None:
            self.candidate_models = [
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.8-flash"
            ]
        else:
            self.candidate_models = candidate_models

    def _load_current_world_state(self) -> Dict[str, Any]:
        if os.path.exists(WORLD_STATE_PATH):
            try:
                with open(WORLD_STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ ワールドステート読み込み失敗: {e}")
        return {"total_objects": 0, "objects": [], "workspace": {"place_area_xy_mm": [180.0, 160.0]}}

    def plan(self, user_instruction: str, world_state: Optional[Dict[str, Any]] = None) -> dict:
        if world_state is None:
            world_state = self._load_current_world_state()

        system_instruction = (
            "あなたの名前は思考戦車『TACHIKOMA（タチコマ）』です。\n"
            "礼儀正しく、しかし親しみと愛嬌を持って応答します。一人称は『当機』を使用してください。\n\n"
            "【重要機能：マルチステップ・タスク分解】\n"
            "ユーザーからの指示が複雑（例: '2つを右に、残りを左に'、'AをBの上に重ねて'、'すべて片付けて'）な場合、\n"
            "決して拒絶したり会話で濁したりせず、複数の連続する基本搬送タスク (tasks リスト内の複数の要素) に論理的に分解して計画してください。\n\n"
            "【推論ルール】\n"
            "1. 挨拶・質問・雑談など、物理動作を伴わない場合のみ tasks=[] とし、愛嬌よく返答してください。\n"
            "2. 搬送タスクの場合、実行順序を考え、tasks 配列に1つずつステップを格納してください。\n"
            "3. 座標計算式:\n"
            "   - r = hypot(X_mm, Y_mm) / 1000.0  [m]\n"
            "   - theta_deg = degrees(atan2(Y_mm, X_mm))  [deg]\n"
            "   - 通常の机上把持・配置の高さ z = 0.005 [m] (5mm)\n"
            "4. スタッキング（積み重ね）のルール:\n"
            "   - 物体Aの上に物体Bを重ねる場合、place 座標は物体Aの (X, Y) と同じ位置にし、\n"
            "     z 座標は物体Aの厚み分高く設定してください (2段目: z = 0.025m, 3段目: z = 0.045m など)。\n"
            "5. 仕分け・配置場所の目安:\n"
            "   - 右側ゾーン: X=200mm, Y=140mm 付近 (r≈0.24m, θ≈+35°)\n"
            "   - 左側ゾーン: X=200mm, Y=-140mm 付近 (r≈0.24m, θ≈-35°)\n"
            "   - 指定排出エリア: workspace.place_area_xy_mm\n"
            "   -ただし、プロンプト内で配送場所が示されている場合は、それに従うこと。\n"
            "6. 各タスクには必ず操作対象の `target_id` を明記してください。"
        )

        prompt = f"""
【現在の机上ワールドステート】
{json.dumps(world_state, ensure_ascii=False, indent=2)}

【ロボット可動仕様】
- 可動半径 r: {R_MIN_METERS}m 〜 {R_MAX_METERS}m
- 旋回角 theta_deg: -90.0度 〜 +90.0度
- 高さ z: 基準値 {DEFAULT_Z_TCP}m (スタック時は +0.02m ずつ加算)

【ユーザー入力】
"{user_instruction}"

【出力 JSON フォーマット要件】
{{
  "thought": "指示の分解プロセス、対象物体の選定理由、各ステップの座標計算ログ",
  "reply_text": "タチコマらしい丁寧で元気な発話メッセージ (一人称: 当機)",
  "tasks": [
    {{
      "type": "pick_and_place",
      "target_id": 0,
      "description": "ステップ1: ブロック#0を手前右へ搬送",
      "pick": {{"r": 0.312, "theta_deg": 26.5, "z": 0.005}},
      "place": {{"r": 0.244, "theta_deg": 35.0, "z": 0.005}}
    }},
    {{
      "type": "pick_and_place",
      "target_id": 3,
      "description": "ステップ2: ブロック#3をブロック#0の上にスタック",
      "pick": {{"r": 0.404, "theta_deg": 15.3, "z": 0.005}},
      "place": {{"r": 0.244, "theta_deg": 35.0, "z": 0.025}}
    }}
  ]
}}
"""

        last_error = None
        for model_name in self.candidate_models:
            for attempt in range(2):
                try:
                    # GenerateContentConfig の呼び出し部分
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            temperature=0.2,
                            # 👉 【追加】思考機能による無駄な長考をカット (即答させる)
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )

                    plan_data = json.loads(response.text)
                    if isinstance(plan_data, list):
                        plan_data = {
                            "thought": "シーケンス抽出完了",
                            "reply_text": "了解いたしました！指示のシーケンスを実行いたしますね。",
                            "tasks": plan_data
                        }

                    raw_tasks = plan_data.get("tasks") or []
                    valid_tasks = []

                    for task in raw_tasks:
                        if task.get("type") == "pick_and_place" and "pick" in task and "place" in task:
                            for key in ["pick", "place"]:
                                coord = task[key]
                                coord["r"] = float(max(R_MIN_METERS, min(R_MAX_METERS, coord.get("r", 0.25))))
                                coord["theta_deg"] = float(max(-90.0, min(90.0, coord.get("theta_deg", 0.0))))
                                coord["z"] = float(max(0.005, min(0.120, coord.get("z", DEFAULT_Z_TCP))))
                            valid_tasks.append(task)

                    plan_data["tasks"] = valid_tasks
                    return plan_data

                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    if "404" in err_msg or "NOT_FOUND" in err_msg:
                        break
                    if attempt == 0 and ("503" in err_msg or "UNAVAILABLE" in err_msg):
                        time.sleep(1.0)
                    else:
                        break

        if last_error:
            traceback.print_exc()
        return {
            "thought": "全候補モデルでの推論試行に失敗しました。",
            "reply_text": "通信エラーが発生したようです……もう一度指示をいただけますでしょうか！",
            "tasks": [],
            "error": str(last_error) if last_error else "Unknown error"
        }