"""
==============================================================================
VLM 物体検出・バウンディングボックス抽出ツール (tools/detect_objects_vlm.py)
==============================================================================
【役割】
ローカルで稼働している Ollama (qwen2.5vl:3b) に対し、机面正射影画像を送信。
机上に置かれた物体（ジェンガ、ペン、ハサミなど）の名称と領域 (Bounding Box) を
JSON フォーマットで返答させ、画像上にバウンディングボックスを描画して確認する。

【事前準備】
1. Ollama が起動していること (ollama serve)
2. モデルがダウンロード済みであること (ollama run qwen2.5vl:3b)

【改善点】
1. 正射影画像を 448x448 にリサイズして送信し、推論トークン数・処理時間を大幅削減。
2. カメラから直接取得する場合も、VisionProjector で真上視点に直してから送信。
3. 初回モデルロードを考慮してタイムアウトを 180 秒に拡大。
==============================================================================
"""
import sys
import os
import json
import base64
import re
import urllib.request
import urllib.error
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"

# 入力画像保存先
CAPTURE_DIR = os.path.join(BASE_DIR, "data", "camera_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)
WARPED_IMAGE_PATH = os.path.join(CAPTURE_DIR, "topdown_warped.jpg")


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    """画像を適切な解像度(最大512px)にリサイズして JPEG/Base64 化"""
    h, w = image_bgr.shape[:2]
    # トークン削減のため、長辺を 512px 以下にリサイズ
    max_side = 512
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise ValueError("画像のエンコードに失敗しました。")
    return base64.b64encode(buffer).decode("utf-8")


def query_qwen_vl_for_objects(image_bgr: np.ndarray) -> str:
    """Qwen2.5-VL:3B に問い合わせて机上の物体検出を行う"""
    img_b64 = encode_image_to_base64(image_bgr)

    prompt_text = (
        "You are an accurate visual perception system for a tabletop robot arm.\n"
        "Detect the graspable small objects on the table (such as wooden blocks, jenga, pens, scissors, tape dispenser, mouse).\n"
        "Do NOT detect the four square ArUco markers at the corners.\n"
        "Output ONLY a valid JSON list. Format:\n"
        "[\n"
        '  {"name": "object_label", "box_2d": [ymin, xmin, ymax, xmax]}\n'
        "]\n"
        "The coordinates [ymin, xmin, ymax, xmax] must be normalized between 0 and 1000."
    )

    request_payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512
        }
    }

    req_data = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    print(f"⏳ Ollama ({MODEL_NAME}) へリクエスト送信中 (初回ロード時は少し時間がかかります)...")
    start_t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_t
            print(f"⚡ 推論完了 (所要時間: {elapsed:.2f} 秒)")
            return res_body.get("response", "")
    except urllib.error.URLError as e:
        print(f"❌ 通信エラー: {e}")
        return ""


def parse_vlm_response(raw_text: str):
    """モデル返答文から JSON 配列を抽出"""
    cleaned = re.sub(r"```json\s*", "", raw_text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
    target_str = match.group(0) if match else cleaned

    try:
        data = json.loads(target_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失敗: {e}")
        print(f"生レスポンス:\n{raw_text}")
    return []


def draw_detections(image_bgr: np.ndarray, detected_objects: list) -> np.ndarray:
    """正射影画像上に BBox と中心点を描画"""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    print("\n🔍 === 検出された物体一覧 ===")
    for i, obj in enumerate(detected_objects):
        name = obj.get("name", f"obj_{i}")
        box = obj.get("box_2d", None)
        if not box or len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box
        if max(ymin, xmin, ymax, xmax) <= 1000:
            px_ymin = int((ymin / 1000.0) * h)
            px_xmin = int((xmin / 1000.0) * w)
            px_ymax = int((ymax / 1000.0) * h)
            px_xmax = int((xmax / 1000.0) * w)
        else:
            px_ymin, px_xmin, px_ymax, px_xmax = int(ymin), int(xmin), int(ymax), int(xmax)

        center_u = int((px_xmin + px_xmax) / 2)
        center_v = int((px_ymin + px_ymax) / 2)

        print(f"  [{i+1}] {name:<12} | BBox: ({px_xmin}, {px_ymin}) -> ({px_xmax}, {px_ymax}) | 中心: ({center_u}, {center_v}) px")

        cv2.rectangle(annotated, (px_xmin, px_ymin), (px_xmax, px_ymax), (0, 255, 0), 2)
        cv2.circle(annotated, (center_u, center_v), 5, (0, 0, 255), -1)
        cv2.putText(annotated, name, (px_xmin, max(20, px_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, name, (px_xmin, max(20, px_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated


def main():
    print("==================================================")
    print(f" 🧠 Qwen2.5-VL 机上物体検出パイプライン ({MODEL_NAME})")
    print("==================================================")

    target_img = None

    # 保存済みの正射影画像が存在すれば優先使用
    if os.path.exists(WARPED_IMAGE_PATH):
        print(f"📁 保存済みの正射影画像をロード: {WARPED_IMAGE_PATH}")
        target_img = cv2.imread(WARPED_IMAGE_PATH)

    # 保存画像がない場合はカメラから正射影を動的生成
    if target_img is None:
        print("📷 カメラから正射影画像を生成中...")
        projector = VisionProjector()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        for _ in range(30):
            ret, frame = cap.read()
            if ret and projector.update_homography(frame):
                target_img = projector.warp_to_topdown(frame, out_w=500, out_h=500)
                break
            time.sleep(0.05)

        cap.release()

        if target_img is not None:
            cv2.imwrite(WARPED_IMAGE_PATH, target_img)
            print(f"💾 正射影画像を保存しました: {WARPED_IMAGE_PATH}")
        else:
            print("❌ 正射影画像の生成に失敗しました（マーカーが揃っていません）。")
            return

    # VLM による物体推論
    raw_response = query_qwen_vl_for_objects(target_img)
    if not raw_response:
        return

    detected_list = parse_vlm_response(raw_response)
    if not detected_list:
        print("⚠️ 物体が検出されませんでした。")
        return

    # 描画と表示
    result_img = draw_detections(target_img, detected_list)
    out_path = os.path.join(CAPTURE_DIR, "vlm_detected_result.jpg")
    cv2.imwrite(out_path, result_img)
    print(f"💾 結果画像を保存しました: {out_path}")

    cv2.imshow("VLM Object Detection Result", result_img)
    print("👉 ウィンドウをクリックして何かキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()