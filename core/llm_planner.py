"""
==============================================================================
TACHIKOMA 自然言語タスクプランナー (core/llm_planner.py)
==============================================================================
【役割】
ユーザーの自然言語指示（例: 「正面の手前にあるものを右奥へ運んで」）を LLM (Gemini) に入力し、
1. 搬送タスクの要否判定
2. 掴み位置 (pick) と配置位置 (place) の極座標パラメータ [r, theta_deg, z] の抽出
3. 抽出結果の JSON パースおよび物理可動限界内の安全クランプ
を行い、構造化された実行シーケンスとして返却します。

【使用ライブラリ】
google-genai SDK (gemini-3.1-flash-lite)
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

# .env ファイルから API キーを読み込む
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

from config.workspace_config import (
    LOCATION_ANGLES,
    DISTANCE_PRESETS,
    R_MIN_METERS,
    R_MAX_METERS,
    DEFAULT_Z_TCP
)


class LLMTaskPlanner:
    def __init__(self, api_key=None):
        """Gemini API クライアントを初期化"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY が見つかりません。.env ファイルを確認してください。"
            )
        self.client = genai.Client(api_key=self.api_key)

    def plan(self, user_instruction: str) -> dict:
        """
        ユーザーの自然言語指示を解析し、極座標パラメータを含むタスクシーケンスを出力する
        """
        system_instruction = (
            "あなたの名前は支援ロボットTACHIKOMAです。\n"
            "極めて優秀な実務支援ユニットとして、論理的、平坦、かつ簡潔な軍事・システム報告調で応答してください。\n"
            "一人称は『当機』を使用し、感情表現や雑談は含めないでください。\n"
            "ユーザーの指示から、掴む位置(pick)と置く位置(place)の円筒極座標 [r: 半径(m), theta_deg: 角度(度), z: 高さ(m)] を特定してください。\n"
            "物理的な搬送を伴わない対話や挨拶の場合は、tasks を空リスト [] に設定してください。"
        )

        prompt = f"""
【ハードウェア座標規約】
- 水平距離 r [m]: 旋回軸中心からの距離。可動限界は {R_MIN_METERS}m 〜 {R_MAX_METERS}m。
  代表値: 手前/近い({DISTANCE_PRESETS['手前']}m), 通常/指定なし({DISTANCE_PRESETS['通常']}m), 奥/遠い({DISTANCE_PRESETS['奥']}m)
- 旋回角度 theta_deg [度]: 正面が 0.0度、時計回り(右)が ＋、反時計回り(左)が −。
  代表値: 正面(0.0), 右前方(+20.0), 右側(+45.0), 左前方(-20.0), 左側(-45.0)
- 高さ z [m]: 机上面からの爪先端高さ。指定がなければデフォルト値 {DEFAULT_Z_TCP}m。

【ユーザー指示】
"{user_instruction}"

【出力 JSON フォーマット要件】
以下の構造の JSON を厳密に返してください。
{{
  "thought": "幾何パラメータの解釈および一連のタスク遷移の解析ログ",
  "reply_text": "オペレーターに対するシステム応答メッセージ",
  "tasks": [
    {{
      "type": "pick_and_place",
      "description": "タスクの簡単な要約 (例: 正面手前 ➔ 右奥へ搬送)",
      "pick": {{"r": 0.20, "theta_deg": 0.0, "z": {DEFAULT_Z_TCP}}},
      "place": {{"r": 0.30, "theta_deg": 30.0, "z": {DEFAULT_Z_TCP}}}
    }}
  ]
}}
"""

        try:
            # Gemini モデルを呼び出し
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            plan_data = json.loads(response.text)
            if isinstance(plan_data, list):
                plan_data = {
                    "thought": "シーケンス抽出完了",
                    "reply_text": "了解。指定された搬送シーケンスを実行します。",
                    "tasks": plan_data
                }

            # 抽出された座標値のバリデーションと安全クランプ
            raw_tasks = plan_data.get("tasks") or []
            valid_tasks = []

            for task in raw_tasks:
                if task.get("type") == "pick_and_place" and "pick" in task and "place" in task:
                    for key in ["pick", "place"]:
                        coord = task[key]
                        # 半径 r を安全範囲にクランプ
                        coord["r"] = float(max(R_MIN_METERS, min(R_MAX_METERS, coord.get("r", 0.25))))
                        # 旋回角 theta を ±60° にクランプ
                        coord["theta_deg"] = float(max(-60.0, min(60.0, coord.get("theta_deg", 0.0))))
                        # 高さ z を安全範囲にクランプ
                        coord["z"] = float(max(0.010, min(0.100, coord.get("z", DEFAULT_Z_TCP))))
                    valid_tasks.append(task)

            plan_data["tasks"] = valid_tasks
            return plan_data

        except Exception as e:
            traceback.print_exc()
            return {
                "thought": "例外検知: 幾何パラメータの解析に失敗しました。",
                "reply_text": "エラー。入力を正しく解釈できませんでした。",
                "tasks": [],
                "error": str(e)
            }