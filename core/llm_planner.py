"""
==============================================================================
LLM タスクプランナー (core/llm_planner.py)
==============================================================================
【役割】
単一または複数の連続搬送指示を解析し、タスクのリスト（sequence）を出力します。
==============================================================================
"""

import os
import sys
import json
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

from config.workspace_config import (
    LOCATION_ANGLES,
    CENTER_RAW,
    MIN_RAW,
    MAX_RAW,
    MAX_RIGHT_DEG,
    MAX_LEFT_DEG,
    location_to_raw
)


class LLMTaskPlanner:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY が見つかりません。\n"
                ".env ファイルを確認してください。"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.known_locations = list(LOCATION_ANGLES.keys())

    def plan(self, user_instruction: str) -> dict:
        system_instruction = (
            "あなたの名前は支援ロボットTACHIKOMAです。\n"
            "極めて優秀な実務支援ユニットです。なお、攻殻機動隊とは関係ないです。\n\n"
            "【ペルソナ・口調ルール】\n"
            "・一人称は『当機』\n"
            "・感情の起伏は一切見せず、極めて論理的、平坦、かつ簡潔な軍事・システム報告調を用いてください。\n"
            "・余計な雑談や装飾は省き、ステータスや座標、処理結果を明瞭に伝達してください。\n"
            "・物理動作が不要な場合は tasks を空リスト [] に設定してください。"
        )

        prompt = f"""
【ハードウェア仕様（ID1 台座旋回サーボ）】
- 角度規約: 正面が 0度、右方向が ＋（プラス）、左方向が −（マイナス）
- 基準中心 (0度): Raw値 {CENTER_RAW}
- 物理限界（右限界 / 最大時計回り）: +{MAX_RIGHT_DEG:.1f}度 (Raw値 {MIN_RAW})
- 物理限界（左限界 / 最大反時計回り）: {MAX_LEFT_DEG:.1f}度 (Raw値 {MAX_RAW})
- 既知の地点・方向リスト: {json.dumps(self.known_locations, ensure_ascii=False)}

【ユーザーの指示】
"{user_instruction}"

【出力形式要件】
以下の JSON オブジェクトのフォーマットを厳密に返してください。
複数の搬送指示がある場合は、tasks リストに順番通りに格納してください。
搬送不要な対話・質問の場合は tasks を [] としてください。

{{
  "thought": "座標・幾何パラメータおよび一連のタスク遷移の解析ログ",
  "reply_text": "オペレーターとしてのシステム応答メッセージ",
  "tasks": [
    {{
      "type": "pick_and_place",
      "pick_location": "掴む位置の名称または角度",
      "place_location": "置く位置の名称または角度"
    }}
  ]
}}
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            raw_text = response.text
            parsed = json.loads(raw_text)

            # 万が一トップレベルがリストで返ってきた場合のラップ救済
            if isinstance(parsed, list):
                plan_data = {"thought": "シーケンス抽出完了", "reply_text": "了解。指示されたシーケンスを実行します。", "tasks": parsed}
            else:
                plan_data = parsed

            raw_tasks = plan_data.get("tasks") or []
            valid_tasks = []

            for t in raw_tasks:
                if isinstance(t, dict) and t.get("type") == "pick_and_place":
                    p_loc = t.get("pick_location")
                    d_loc = t.get("place_location")
                    if p_loc is not None and d_loc is not None:
                        t["theta_pick_raw"] = location_to_raw(p_loc)
                        t["theta_place_raw"] = location_to_raw(d_loc)
                        valid_tasks.append(t)

            plan_data["tasks"] = valid_tasks
            return plan_data

        except Exception as e:
            print("\n🚨 [デバッグ情報] 例外が発生しました:")
            traceback.print_exc()
            return {
                "thought": "例外検知: パラメータの解析に失敗しました。",
                "reply_text": "エラーを検知。入力を正しく解釈できませんでした。",
                "tasks": [],
                "error": str(e)
            }