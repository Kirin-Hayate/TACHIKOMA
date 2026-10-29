import os
import sys
import time
import cv2
from flask import Flask, Response, render_template_string

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.joint_config import FOLLOWER_PORT, BAUDRATE

try:
    from core.bus_servo_controller import BusServoController
    HAS_BUS_SERVO = True
except ImportError:
    HAS_BUS_SERVO = False

app = Flask(__name__)

# カメラ初期化 (安定しているインデックス 0 を使用)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# アーム制御初期化
controller = None
if HAS_BUS_SERVO:
    try:
        controller = BusServoController(port=FOLLOWER_PORT, baudrate=BAUDRATE)
        if controller.connect():
            print(f"✅ アーム制御コントローラに接続成功: {FOLLOWER_PORT}")
        else:
            print("⚠️ アーム接続に失敗 (カメラ配信単体モードで待機)")
            controller = None
    except Exception as e:
        print(f"⚠️ アーム初期化例外: {e}")
        controller = None

# 現在のサーボ角度 (安全な可動初期値)
# ID 1: パン (台座旋回), ID 2: チルト (肩ピッチ)
current_pan = 2130
current_tilt = 2000

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>TACHIKOMA 遠隔花火観測所</title>
  <style>
    body { background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, sans-serif; text-align: center; margin: 0; padding: 12px; }
    h2 { font-size: 1.1rem; margin: 8px 0; color: #38bdf8; }
    .stream-container { width: 100%; max-width: 720px; margin: 0 auto; border-radius: 12px; overflow: hidden; background: #000; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }
    img { width: 100%; height: auto; display: block; }
    .pad { display: grid; grid-template-columns: repeat(3, 1fr); max-width: 260px; margin: 16px auto; gap: 8px; }
    button {
      padding: 14px 0; font-size: 1.1rem; font-weight: bold; border: none; border-radius: 8px;
      background: #1e293b; color: #f8fafc; cursor: pointer; transition: background 0.1s;
      -webkit-tap-highlight-color: transparent;
    }
    button:active { background: #334155; transform: scale(0.96); }
    .center-btn { background: #0284c7; }
    .status { font-size: 0.85rem; color: #94a3b8; margin-top: 8px; }
  </style>
</head>
<body>
  <h2>🎆 TACHIKOMA 遠隔花火観測所</h2>
  <div class="stream-container">
    <img src="/video_feed" alt="Live Stream">
  </div>

  <div class="pad">
    <div></div>
    <button onclick="sendMove('up')">▲ 上</button>
    <div></div>
    <button onclick="sendMove('left')">◀ 左</button>
    <button class="center-btn" onclick="sendMove('center')">● 戻す</button>
    <button onclick="sendMove('right')">右 ▶</button>
    <div></div>
    <button onclick="sendMove('down')">▼ 下</button>
    <div></div>
  </div>
  <div id="status" class="status">待機中</div>

  <script>
    function sendMove(action) {
      document.getElementById('status').innerText = '操作コマンド送信中...';
      fetch('/move/' + action)
        .then(res => res.json())
        .then(data => {
          document.getElementById('status').innerText = `Pan: ${data.pan} | Tilt: ${data.tilt}`;
        })
        .catch(err => {
          document.getElementById('status').innerText = '通信エラー';
        });
    }
  </script>
</body>
</html>
"""

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue
        # モバイル回線向けに画質 65% で JPEG 圧縮
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/move/<action>')
def move(action):
    global current_pan, current_tilt
    step = 100
    if action == 'left':
        current_pan = min(3200, current_pan + step)
    elif action == 'right':
        current_pan = max(1000, current_pan - step)
    elif action == 'up':
        current_tilt = min(2800, current_tilt + step)
    elif action == 'down':
        current_tilt = max(1200, current_tilt - step)
    elif action == 'center':
        current_pan, current_tilt = 2130, 2000

    if controller and hasattr(controller, 'driver') and controller.driver:
        # 指令値を 300ms でスムーズに送信
        controller.driver.write_position(1, current_pan, time_ms=300)
        controller.driver.write_position(2, current_tilt, time_ms=300)

    return {"pan": current_pan, "tilt": current_tilt}

if __name__ == '__main__':
    # ポート 8502 で待機
    app.run(host='0.0.0.0', port=8502, threaded=True)