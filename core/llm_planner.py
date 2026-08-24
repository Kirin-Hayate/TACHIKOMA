"""
==============================================================================
LLM タスクプランナー (core/llm_planner.py)
==============================================================================
【役割】
1. .env から GEMINI_API_KEY を自動で読み込みます。
2. motions/motions.json に登録された動作ファイル一覧を読み取ります。
3. ユーザーの自然言語指示（例: 「AからBに運んで、そのあとCに持ってって！」）を
   Gemini API (gemini-3.5-flash-Lite) に送り、実行すべきCSVファイルと回数のリスト (JSON) を生成します。
==============================================================================
"""

import os
import sys
import json
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env ファイルの環境変数を自動読み込み
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)


class LLMTaskPlanner:
    def __init__(self, motions_json_path, api_key=None):
        self.motions_json_path = motions_json_path
        
        # APIキーの取得確認
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY が見つかりません。\n"
                "プロジェクト直下に .env ファイルを作成し、\n"
                "GEMINI_API_KEY=AIzaSy... と記述してください。"
            )

        # Gemini クライアントの初期化
        self.client = genai.Client(api_key=self.api_key)
        self.available_motions = self._load_motions()

    def _load_motions(self):
        if not os.path.exists(self.motions_json_path):
            raise FileNotFoundError(f"❌ モーション定義ファイルが見つかりません: {self.motions_json_path}")
        
        with open(self.motions_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def plan(self, user_instruction: str) -> dict:
        system_instruction = (
            "あなたは、ひとのくらしをサポートする手伝いロボット「タチコマ」のアクションプランナーです。\n"
            "ユーザーからの自然言語指示を理解し、手持ちのモーション一覧から\n"
            "最適な組み合わせと実行回数のシーケンスを作成してください。"
            "【口調ルール】\n"
            "・極めて冷静、論理的かつ簡潔に応答してください。\n"
            "・一人称は『本システム』または『当機』。\n"
            "・『了解。シーケンスを構成します』『ミッションパラメータを確認』などの軍事・システム用語を使用してください。"
        )

        prompt = f"""
【利用可能なモーション一覧 (motions.json)】
{json.dumps(self.available_motions, ensure_ascii=False, indent=2)}

【ユーザーの指示】
"{user_instruction}"

【出力形式要件】
以下の JSON フォーマットのみを返してください:
{{
  "thought": "選択した理由",
  "reply_text": "タチコマ口調の返答メッセージ",
  "sequence": [
    {{"file": "ファイル名.csv", "repeat": 1}}
  ]
}}
"""

        try:
            # モデルはgemini-3.1-flash-lite
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )

            # レスポンス文字列の確認
            raw_text = response.text
            plan_data = json.loads(raw_text)
            return plan_data

        except Exception as e:
            # 🔍 エラーの詳細をコンソールに出力
            print("\n🚨 [デバッグ情報] 例外が発生しました:")
            traceback.print_exc()
            
            return {
                "thought": "エラーが発生しました",
                "reply_text": "プラン生成に失敗。もう一度指示を与えてください。",
                "sequence": [],
                "error": str(e)
            }


if __name__ == "__main__":
    motions_path = os.path.join(BASE_DIR, "motions", "motions.json")

    print("==================================================")
    print(" 🤖 LLM タスクプランナー 単体テスト")
    print("==================================================")

    try:
        planner = LLMTaskPlanner(motions_path)
        test_prompt = "A地点の積み木をB地点に運んで！"
        print(f"🗣️ 入力指示: {test_prompt}\n")
        print("🧠 プラン生成中...")

        result = planner.plan(test_prompt)

        print("\n--- 📝 生成されたプラン ---")
        print(f"💬 返答: {result.get('reply_text')}")
        print(f"💡 思考: {result.get('thought')}")
        print("📋 実行シーケンス:")
        for idx, item in enumerate(result.get("sequence", []), start=1):
            print(f"   {idx}. {item.get('file')}  (再生回数: {item.get('repeat')}回)")

    except Exception as e:
        print(f"\n❌ エラー: {e}")