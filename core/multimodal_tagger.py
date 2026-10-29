"""
==============================================================================
TACHIKOMA マルチモーダル物体同定モジュール (core/multimodal_tagger.py)
==============================================================================
【モジュールの役割と構成】
本モジュールは、OpenCV で切り出した複数の物体クロップ画像から最適なグリッド台紙を動的に生成し、
Gemini API (クラウド) または Qwen2.5-VL (ローカル Ollama) へ「1リクエスト」で一括送信して、
各物体のセマンティック情報（カテゴリ名、主要色、特徴表現）を高速に同定します。

1. 動的コラージュ画像生成レイヤー
   - build_adaptive_collage: 検出数 N に応じた最適縦横比 (行x列) を算出し、各タイルの左上に
     インデックス番号 (#0, #1, ...) を焼き込んだ単一の合成画像を生成

2. クラウド VLM レイヤー (Gemini API)
   - query_gemini: Pydantic 構造化スキーマを用いた高速 JSON 出力と複数モデル (Flash/Lite) への
     自動フェイルオーバー・指数バックオフリトライ

3. ローカル VLM レイヤー (Ollama / Qwen2.5-VL)
   - query_qwen: Base64 エンコードしたコラージュ画像をローカル HTTP API 経由で推論し、
     完全オフラインで JSON パースして属性を抽出

4. 統合ディスパッチャレイヤー
   - identify_items: 起動引数に応じて Gemini -> Qwen への自動フォールバックまたは排他実行を統括
==============================================================================
"""

import os
import sys
import time
import json
import math
import re
import base64
import threading
from contextlib import contextmanager
import urllib.request
import urllib.error
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# プロジェクトルートと環境変数
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Google GenAI SDK
from google import genai
from google.genai import types


# ==============================================================================
# 1. Pydantic 出力スキーマ定義
# ==============================================================================
class IdentifiedItem(BaseModel):
    id: int = Field(description="画像内のタイル番号 (#0, #1 等の整数)")
    category: str = Field(description="物体の一般名 (例: pen, wooden block, eraser, mouse, ruler)")
    color: str = Field(description="物体の主要な色 (例: red, blue, silver, natural wood, black)")
    description: str = Field(default="", description="簡潔な特徴表現")

class CollageClassificationResult(BaseModel):
    items: List[IdentifiedItem] = Field(description="同定結果リスト")


# ==============================================================================
# 2. プログレスタイマー監視ユーティリティ
# ==============================================================================
@contextmanager
def progress_timer(label: str = "推論処理中"):
    """VLM 応答待ちの間に 1 秒刻みで経過秒数をコンソールに上書き表示"""
    stop_event = threading.Event()
    start_time = time.time()

    def _worker():
        while not stop_event.wait(1.0):
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r   ⏳ {label}... 経過: {elapsed:.1f} 秒")
            sys.stdout.flush()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        sys.stdout.write("\r" + " " * 48 + "\r")
        sys.stdout.flush()


# ==============================================================================
# 3. MultimodalTagger クラス
# ==============================================================================
class MultimodalTagger:
    def __init__(
        self,
        gemini_candidate_models: Optional[List[str]] = None,
        qwen_model_name: str = "qwen2.5vl:3b",
        ollama_api_url: str = "http://localhost:11434/api/generate",
        tile_size: int = 160
    ):
        self.tile_size = tile_size
        self.qwen_model_name = qwen_model_name
        self.ollama_api_url = ollama_api_url

        # デフォルトの試行順序
        if gemini_candidate_models is None:
            self.gemini_models = [
                "gemini-3.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemini-3.7-flash",
                "gemini-3.6-flash"
            ]
        else:
            self.gemini_models = gemini_candidate_models

        # Gemini クライアントの遅延初期化用
        self._gemini_client = None

    def _get_gemini_client(self) -> Optional[genai.Client]:
        if self._gemini_client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print(f"❌ '{ENV_PATH}' または環境変数内に 'GEMINI_API_KEY' が見つかりませんでした。")
                return None
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def build_adaptive_collage(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
        """
        検出数 N に応じた最適な縦横比の白い台紙画像を生成し、
        各物体の左上に '#0', '#1' と番号を焼き込んで連結する。
        """
        n = len(crops)
        if n == 0:
            return np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8), 0, 0

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        collage = np.full((rows * self.tile_size, cols * self.tile_size, 3), 255, dtype=np.uint8)

        for idx, crop in enumerate(crops):
            r, c = idx // cols, idx % cols
            y_off = r * self.tile_size
            x_off = c * self.tile_size

            pad = 8
            cell_w = self.tile_size - pad * 2
            cell_h = self.tile_size - pad * 2

            ch, cw = crop.shape[:2]
            scale = min(cell_w / cw, cell_h / ch)
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

            oy = y_off + pad + (cell_h - nh) // 2
            ox = x_off + pad + (cell_w - nw) // 2
            collage[oy:oy + nh, ox:ox + nw] = resized

            cv2.rectangle(collage, (x_off, y_off), (x_off + self.tile_size, y_off + self.tile_size), (210, 210, 210), 1)
            tag = f"#{idx}"
            cv2.rectangle(collage, (x_off + 4, y_off + 4), (x_off + 52, y_off + 28), (0, 0, 0), -1)
            cv2.putText(collage, tag, (x_off + 8, y_off + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return collage, rows, cols

    def query_gemini(self, collage_bgr: np.ndarray, num_items: int) -> Dict[int, Dict[str, str]]:
        """1枚のコラージュ画像を Gemini に送信し、耐障害リトライを経て構造化辞書を取得"""
        client = self._get_gemini_client()
        if client is None:
            return {}

        img_rgb = cv2.cvtColor(collage_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        prompt = (
            f"This composite image contains {num_items} tabletop object crops arranged in a grid (#0, #1, ...).\n"
            "Identify each item's canonical category name (e.g. 'pen', 'wooden block', 'eraser', 'mouse', 'ruler'), "
            "primary dominant color (e.g. 'red', 'blue', 'silver', 'black', 'natural wood'), "
            "and a concise visual description including texture, markings, or state (e.g. 'striped pattern', 'wooden grain', 'metallic finish').\n"  # 👈 ここを追加
            "Return structured JSON matching the schema."
        )

        for model_name in self.gemini_models:
            for attempt in range(2):
                try:
                    print(f"⚡ [Gemini: {model_name}] へ推論リクエスト中... (試行 {attempt + 1}/2)")
                    t0 = time.time()
                    with progress_timer(f"Gemini ({model_name}) 応答待機中"):
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[pil_image, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=CollageClassificationResult,
                                temperature=0.1
                            ),
                        )
                    print(f"✅ Gemini 推論完了 ({model_name}, 所要時間: {time.time() - t0:.2f} 秒)")
                    data = json.loads(response.text)
                    res = {}
                    for item in data.get("items", []):
                        res[item["id"]] = {
                            "category": item.get("category", "object"),
                            "color": item.get("color", ""),
                            "description": item.get("description", "")
                        }
                    return res
                except Exception as e:
                    err_msg = str(e)
                    print(f"⚠️ {model_name} エラー: {err_msg}")
                    if "503" in err_msg or "UNAVAILABLE" in err_msg:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break
        return {}

    def query_qwen(self, collage_bgr: np.ndarray, num_items: int) -> Dict[int, Dict[str, str]]:
        """コラージュ画像をローカル Ollama (qwen2.5vl:3b) に投げ、JSON 形式で一括同定"""
        success, buffer = cv2.imencode(".jpg", collage_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            return {}
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            f"This composite image shows {num_items} tabletop object crops labeled #0 to #{num_items - 1}.\n"
            "Identify each item's canonical category name (e.g. 'pen', 'wooden block', 'eraser', 'mouse', 'ruler') "
            "and primary color (e.g. 'red', 'blue', 'silver', 'black', 'natural wood').\n"
            "Output ONLY a raw JSON array matching this exact format:\n"
            '[{"id": 0, "category": "pen", "color": "red"}, {"id": 1, "category": "mouse", "color": "black"}]\n'
            "Do not include any explanation or markdown tags."
        )

        request_payload = {
            "model": self.qwen_model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512}
        }

        req_data = json.dumps(request_payload).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_api_url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        print(f"🦙 [Local Qwen: {self.qwen_model_name}] へ推論リクエスト中...")
        t0 = time.time()
        try:
            with progress_timer("Qwen 推論中"):
                with urllib.request.urlopen(req, timeout=90) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
            raw_text = res_body.get("response", "").strip()
            print(f"✅ Qwen 推論完了 (所要時間: {time.time() - t0:.2f} 秒)")

            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                items_data = json.loads(json_match.group(0))
                res = {}
                for item in items_data:
                    res[int(item.get("id", 0))] = {
                        "category": item.get("category", "object"),
                        "color": item.get("color", ""),
                        "description": ""
                    }
                return res
            else:
                print(f"⚠️ Qwen 出力パース失敗: {raw_text[:100]}...")
                return {}
        except Exception as e:
            print(f"⚠️ Qwen 推論エラー: {e}")
            return {}

    def identify_items(
        self,
        crops: List[np.ndarray],
        use_gemini: bool = True,
        use_qwen: bool = True
    ) -> Dict[int, Dict[str, str]]:
        """
        クロップ画像リストから動的コラージュを生成し、
        Gemini / Qwen / ハイブリッドフォールバックにより同定属性辞書を返却する。
        
        戻り値:
            {
               0: {"category": "pen", "color": "red", "display_name": "red pen", "description": "..."},
               ...
            }
        """
        if not crops:
            return {}

        collage_img, rows, cols = self.build_adaptive_collage(crops)
        num_items = len(crops)

        raw_attrs = {}
        if use_gemini:
            raw_attrs = self.query_gemini(collage_img, num_items)
            if not raw_attrs and use_qwen:
                print("🔄 Gemini の推論に失敗したため、ローカル Qwen に自動フォールバックします...")
                raw_attrs = self.query_qwen(collage_img, num_items)
        elif use_qwen:
            raw_attrs = self.query_qwen(collage_img, num_items)

        # 表示用文字列 (display_name) を成形して返却
        final_profiles = {}
        for idx in range(num_items):
            attr = raw_attrs.get(idx, {"category": "object", "color": "", "description": ""})
            color = attr.get("color", "").strip()
            category = attr.get("category", "object").strip()
            display_name = f"{color} {category}".strip() if color else category

            final_profiles[idx] = {
                "category": category,
                "color": color,
                "description": attr.get("description", ""),
                "display_name": display_name
            }

        return final_profiles