"""
==============================================================================
机上デジタルツイン同期ツール (tools/sync_digital_twin.py)
==============================================================================
【役割】
1. カメラ映像から OpenCV 背景差分を用いて机上の全ジェンガを 30fps で検出。
2. 検出したピクセル位置・傾きをロボット物理座標系へ幾何射影。
3. MuJoCo シミュレータ空間の対応するブロック (jenga_block_0〜7) の位置・姿勢を
   リアルタイムに更新し、物理空間と仮想空間を完全同期させる。

【操作】
  - [B]     : 現在の机面を背景として記憶（背景差分更新）
  - [SPACE] : 4隅マーカーから正射影を再計算
  - [Q/ESC] : 終了

【修正点】
1. sim.launch() を明示的に呼び出して MuJoCo ビューアウィンドウを起動。
2. 物体座標を反映した後に sim.viewer.sync() を呼び出し、3D描画をリアルタイム同期。

【修正点】
1. MuJoCo ワールド座標の左右軸反転 (-mj_x) による左右連動の修正
2. OpenCV ウィンドウ / MuJoCo ウィンドウ双方でのキーイベント受付
==============================================================================
"""

import sys
import os
import time
import math
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from typing import List, Dict, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.kinematics import get_home_radians

# キャンバス設定
CANVAS_SIZE = 500
MARGIN = 50
INNER_SPAN_PX = 400

# ジェンガのサイズ定義 (MuJoCo half-size: 75x25x15 mm)
JENGA_HALF_Z = 0.0075  # 7.5mm (机面上の中心Z高さ)
MAX_SLOTS = 8

MIN_AREA_PX = 600
MAX_AREA_PX = 15000

# キーフラグ管理
REQ_SAVE_BG = False
REQ_RECALIB = False
REQ_QUIT = False


def create_marker_mask(size: int = CANVAS_SIZE, margin: int = MARGIN) -> np.ndarray:
    """4隅マーカーを除外する中央作業領域マスク"""
    mask = np.zeros((size, size), dtype=np.uint8)
    pad = margin + 15
    mask[pad:size - pad, pad:size - pad] = 255
    return mask


def pixel_to_robot_phys_xy(u: float, v: float, projector: VisionProjector) -> Tuple[float, float]:
    """正射影ピクセル (u, v) をロボット直交物理座標 (X_mm, Y_mm) に変換"""
    p0 = projector.marker_phys_xy[0]
    p1 = projector.marker_phys_xy[1]
    p2 = projector.marker_phys_xy[2]
    p3 = projector.marker_phys_xy[3]

    s = (u - MARGIN) / INNER_SPAN_PX
    t = (v - MARGIN) / INNER_SPAN_PX

    top = (1.0 - s) * p0 + s * p1
    bottom = (1.0 - s) * p2 + s * p3
    phys_xy = (1.0 - t) * top + t * bottom
    return float(phys_xy[0]), float(phys_xy[1])


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    """Z軸周りのヨー回転角 (rad) を MuJoCo クォータニオン [w, x, y, z] に変換"""
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def extract_objects(warped_img: np.ndarray, bg_gray: Optional[np.ndarray], valid_mask: np.ndarray):
    """OpenCV による回転矩形抽出"""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if bg_gray is not None:
        diff = cv2.absdiff(blurred, bg_gray)
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 6
        )

    thresh = cv2.bitwise_and(thresh, thresh, mask=valid_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w < h:
            w, h = h, w
            angle += 90.0

        while angle > 90.0:
            angle -= 180.0
        while angle <= -90.0:
            angle += 180.0

        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)

        detected.append({
            "u": cx,
            "v": cy,
            "angle_deg": angle,
            "box_pts": box_pts
        })

    return detected, thresh


def custom_sim_key_callback(keycode: int):
    """MuJoCo ウィンドウ側のキーボードイベントもフックする"""
    global REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT
    if keycode in (66, 98):  # B, b
        REQ_SAVE_BG = True
        print("\n📸 [MuJoCo窓経由] 背景画像を記憶要求を受け付けました。")
    elif keycode == 32:  # Space
        REQ_RECALIB = True
        print("\n🔄 [MuJoCo窓経由] キャリブレーション更新要求を受け付けました。")
    elif keycode in (81, 113, 256):  # Q, q, ESC
        REQ_QUIT = True
        print("\n🛑 [MuJoCo窓経由] 終了要求を受け付けました。")


def main():
    global REQ_SAVE_BG, REQ_RECALIB, REQ_QUIT
    print("==================================================")
    print(" 🌐 TACHIKOMA リアルタイム・デジタルツイン同期")
    print("==================================================")
    print("【キーボード操作（どちらのウィンドウでも有効）】")
    print("  [B]     : 現在の机面を背景として記憶（背景差分更新）")
    print("  [SPACE] : 4隅マーカーから正射影を再計算")
    print("  [Q/ESC] : 終了")
    print("--------------------------------------------------")

    # 1. 3Dシミュレータ初期化
    sim = MujocoSimViewer()
    home_rad = get_home_radians()
    try:
        sim.update_joints_rad(home_rad)
    except Exception:
        pass

    # MuJoCo ビューアの起動（カスタムキーコールバックを登録）
    sim.viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=custom_sim_key_callback
    )

    # ブロック各ボディのアドレス解決
    block_qpos_addrs = []
    for i in range(MAX_SLOTS):
        bname = f"jenga_block_{i}"
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid != -1:
            jnt_adr = sim.model.body_jntadr[bid]
            if jnt_adr != -1:
                qpos_adr = sim.model.jnt_qposadr[jnt_adr]
                block_qpos_addrs.append(qpos_adr)
            else:
                block_qpos_addrs.append(None)
        else:
            block_qpos_addrs.append(None)

    valid_slots = sum(1 for a in block_qpos_addrs if a is not None)
    print(f"✅ MuJoCo 内に {valid_slots} 個のジェンガスロットを検出しました。")

    # 2. カメラ & プロジェクター初期化
    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    valid_mask = create_marker_mask(CANVAS_SIZE, MARGIN)
    bg_gray = None

    cv2.namedWindow("Digital Twin Camera Tracker")

    try:
        while sim.is_running() and not REQ_QUIT:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=CANVAS_SIZE, out_h=CANVAS_SIZE)
            if warped is None:
                cv2.imshow("Digital Twin Camera Tracker", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in [ord('q'), ord('Q'), 27]:
                    break
                continue

            # 背景記憶の要求処理
            if REQ_SAVE_BG:
                cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)
                REQ_SAVE_BG = False
                print("📸 背景画像を記憶しました。")

            # 再キャリブレーション要求処理
            if REQ_RECALIB:
                projector.update_homography(frame)
                REQ_RECALIB = False
                print("🔄 正射影キャリブレーションを更新しました。")

            # 物体検出
            detected_objs, _ = extract_objects(warped, bg_gray, valid_mask)
            annotated = warped.copy()

            # --- MuJoCo シーン同期処理 ---
            for i in range(MAX_SLOTS):
                qadr = block_qpos_addrs[i]
                if qadr is None:
                    continue

                if i < len(detected_objs):
                    obj = detected_objs[i]
                    x_mm, y_mm = pixel_to_robot_phys_xy(obj["u"], obj["v"], projector)

                    # ========================================================
                    # 左右反転の修正:
                    # ロボット座標系の正面: -Y (mj_y = -x_mm)
                    # ロボット座標系の右側: +X (mj_x = -y_mm / 1000.0 に反転)
                    # ========================================================
                    mj_x = -y_mm / 1000.0
                    mj_y = -x_mm / 1000.0
                    mj_z = JENGA_HALF_Z

                    # 回転方向の反転整合（画像系から右手系 MuJoCo への符号反転）
                    yaw_rad = math.radians(-obj["angle_deg"])
                    quat = euler_yaw_to_quat(yaw_rad)

                    sim.data.qpos[qadr:qadr + 3] = [mj_x, mj_y, mj_z]
                    sim.data.qpos[qadr + 3:qadr + 7] = quat

                    cv2.drawContours(annotated, [obj["box_pts"]], 0, (0, 255, 0), 2)
                    cv2.circle(annotated, (int(obj["u"]), int(obj["v"])), 4, (0, 0, 255), -1)
                    cv2.putText(annotated, f"#{i}: ({int(x_mm)},{int(y_mm)})",
                                (int(obj["u"]) - 35, int(obj["v"]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    # 検出されていないスロットは机の下へ
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]
                    sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            mode_text = "Background Diff" if bg_gray is not None else "Adaptive (Press B to calibrate BG)"
            cv2.putText(annotated, f"Sync: {len(detected_objs)} blocks | {mode_text}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Digital Twin Camera Tracker", annotated)

            # OpenCV 側のキー入力
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('b'), ord('B')]:
                REQ_SAVE_BG = True
            elif key == 32:  # SPACE
                REQ_RECALIB = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()