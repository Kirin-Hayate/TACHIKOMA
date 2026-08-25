"""
==============================================================================
LLM タスクプランナー (core/llm_planner.py)
==============================================================================
【役割】
ユーザーの自然言語指示を解釈し、搬送タスクの「掴む位置」と「置く位置」
(地点名、方向、または角度) を抽出した JSON プランを生成します。
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

from config.workspace_config import LOCATION_ANGLES, location_to_raw


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
        """
        自然言語指示から搬送パラメータを抽出
        """
        system_instruction = (
            "あなたは公安9課に配属されているオペレーター用女性型アンドロイド（プロッサー）です。\n"
            "電脳通信および端末操作をミリ秒単位で処理する、極めて優秀な実務支援ユニットです。\n\n"
            "【ペルソナ・口調ルール】\n"
            "・一人称は『当機』または『オペレーター』。\n"
            "・感情の起伏は一切見せず、極めて論理的、平坦、かつ簡潔な軍事・システム報告調（〜を確認、〜を実行します、スタンバイ完了）を用いてください。\n"
            "・余計な雑談や装飾は省き、ステータスや座標、処理結果を明瞭に伝達してください。\n"
            "・ユーザーからの自然言語指示を解釈し、マニピュレータが実行すべき\n"
            "  「掴む位置 (pick)」と「置く位置 (place)」を特定してください。"
        )

        prompt = f"""
【認識可能な既知の地点・方向の例】
{json.dumps(self.known_locations, ensure_ascii=False)}

【ユーザーの指示】
"{user_instruction}"

【出力形式要件】
以下の JSON フォーマットのみを返してください。
pick_location, place_location には、既知の地点名（例: "A", "右", "10時方向" など）や、
文脈から読み取れる角度（数値や "30度" など）を入れてください。

{{
  "thought": "座標・幾何パラメータの解析ログ",
  "reply_text": "オペレーターとしてのシステム応答メッセージ",
  "task": {{
    "type": "pick_and_place",
    "pick_location": "掴む位置の名称または角度",
    "place_location": "置く位置の名称または角度"
  }}
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

            plan_data = json.loads(response.text)
            
            task = plan_data.get("task", {})
            pick_loc = task.get("pick_location", "中央")
            place_loc = task.get("place_location", "中央")
            
            plan_data["raw_params"] = {
                "theta_pick_raw": location_to_raw(pick_loc),
                "theta_place_raw": location_to_raw(place_loc)
            }

            return plan_data

        except Exception as e:
            print("\n🚨 [デバッグ情報] 例外が発生しました:")
            traceback.print_exc()
            return {
                "thought": "例外検知: パラメータの解析に失敗しました。",
                "reply_text": "エラーを検知。座標パラメータの抽出に失敗しました。再入力を要求します。",
                "task": {"type": "pick_and_place", "pick_location": "中央", "place_location": "中央"},
                "raw_params": {"theta_pick_raw": 2048, "theta_place_raw": 2048},
                "error": str(e)
            }


# ==============================================================================
# 単体テスト用メイン処理
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print(" 🤖 パラメトリック LLM プランナー 単体テスト (オペレーターモード)")
    print("==================================================")

    planner = LLMTaskPlanner()

    test_prompts = [
        "右側にある積み木を、正面の真ん中に移動させて！",
        "10時方向のやつを2時方向へ運んでくれる？",
        "A地点からC地点へ運んで！"
    ]

    for prompt in test_prompts:
        print(f"\n🗣️ 指示: {prompt}")
        result = planner.plan(prompt)
        print(f"💬 報告: {result.get('reply_text')}")
        print(f"💡 ログ: {result.get('thought')}")
        task = result.get("task") or {}
        raws = result.get("raw_params") or {}
        print(f"🎯 目標値: [{task.get('pick_location')}] ➔ [{task.get('place_location')}]")
        print(f"   (ID1 Raw変換値: 掴み={raws.get('theta_pick_raw')} ➔ 配置={raws.get('theta_place_raw')})")