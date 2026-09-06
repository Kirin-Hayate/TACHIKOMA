"""
==============================================================================
TACHIKOMA 自然言語タスクプランナー (core/llm_planner.py)
==============================================================================
【役割】
ユーザーの自然言語指示を入力し、
1. 挨拶や雑談への自然な応答 (reply_text)
2. 搬送タスクがある場合の極座標 [r, theta_deg, z] 抽出 (tasks)
を両立して JSON 形式で出力します。
旋回角 theta_deg は最大 ±90.0° のフルレンジに対応します。
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
    DISTANCE_PRESETS,
    R_MIN_METERS,
    R_MAX_METERS,
    DEFAULT_Z_TCP
) 


class LLMTaskPlanner:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") 
        if not self.api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY が見つかりません。.env ファイルを確認してください。" 
            )
        self.client = genai.Client(api_key=self.api_key) 

    def plan(self, user_instruction: str) -> dict:
        system_instruction = (
            "あなたの名前は『TACHIKOMA（タチコマ）』です。\n"
            "礼儀正しく応答します。\n"
            "一人称は『当機』を使用してください。\n\n"
            "【応答方針】\n"
            "1. 挨拶・雑談・お祝い・質問など、物理的なモノの移動を伴わない指示の場合:\n"
            "   - 相手の言葉を受け止めて、端的でわかりやすく、感情を排して会話してください。\n"
            "   - tasks リストは必ず空リスト [] にしてください。\n"
            "2. モノの移動・把持・配置指示（Pick & Place）の場合:\n"
            "   - 了解した旨を快く報告し、tasks に極座標 [r, theta_deg, z] を設定してください。"
        )

        prompt = f"""
【ハードウェア座標仕様】
- 水平距離 r [m]: 旋回中心からの距離。可動限界は {R_MIN_METERS}m 〜 {R_MAX_METERS}m。
  代表値: 手前/近い({DISTANCE_PRESETS['手前']}m), 通常({DISTANCE_PRESETS['通常']}m), 奥/遠い({DISTANCE_PRESETS['奥']}m)
- 旋回角度 theta_deg [度]: 正面が 0.0度、時計回り(右)が ＋、反時計回り(左)が −。
  可動限界: -90.0度 〜 +90.0度
  代表値:
    正面: 0.0度
    右前方: +20.0度, 右側: +45.0度, 最も右/右端/真右: +90.0度
    左前方: -20.0度, 左側: -45.0度, 最も左/左端/真左: -90.0度
- 高さ z [m]: 机上面からの爪先端高さ。指定がなければデフォルト値 {DEFAULT_Z_TCP}m。

【ユーザー入力】
"{user_instruction}"

【出力 JSON フォーマット要件】
{{
  "thought": "会話の意図や座標解析のログ",
  "reply_text": "端的でわかりやすく、感情を排した会話応答メッセージ",
  "tasks": [
    {{
      "type": "pick_and_place",
      "description": "タスク要約 (例: 最も左から最も右へ搬送)",
      "pick": {{"r": 0.25, "theta_deg": -90.0, "z": {DEFAULT_Z_TCP}}},
      "place": {{"r": 0.25, "theta_deg": 90.0, "z": {DEFAULT_Z_TCP}}}
    }}
  ]
}}
※ 搬送不要な会話・挨拶の場合は tasks を必ず [] にしてください。
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
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
                    "reply_text": "了解しました！指示のシーケンスを実行しますね。",
                    "tasks": plan_data
                }

            # 座標値のバリデーションと安全クランプ (±90° 対応)
            raw_tasks = plan_data.get("tasks") or [] 
            valid_tasks = []

            for task in raw_tasks: 
                if task.get("type") == "pick_and_place" and "pick" in task and "place" in task: 
                    for key in ["pick", "place"]: 
                        coord = task[key] 
                        coord["r"] = float(max(R_MIN_METERS, min(R_MAX_METERS, coord.get("r", 0.25)))) 
                        # 旋回角 theta を ±90.0° にクランプ
                        coord["theta_deg"] = float(max(-90.0, min(90.0, coord.get("theta_deg", 0.0))))
                        coord["z"] = float(max(0.010, min(0.100, coord.get("z", DEFAULT_Z_TCP)))) 
                    valid_tasks.append(task) 

            plan_data["tasks"] = valid_tasks 
            return plan_data 

        except Exception as e:
            traceback.print_exc() 
            return {
                "thought": "モデル呼び出しまたはパース例外が発生しました。",
                "reply_text": "通信エラーが発生したみたいです……もう一度話しかけてみてください！",
                "tasks": [],
                "error": str(e)
            } 