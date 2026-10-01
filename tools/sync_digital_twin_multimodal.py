"""
==============================================================================
マルチモーダル・デジタルツイン同期 ＆ 机上ワールドステート生成システム
(tools/sync_digital_twin_multimodal.py)
==============================================================================
【機能】
1. カメラ画像から全物体を OpenCV で検出し、物理座標・傾き・ミリ寸法・アスペクト比を計測[cite: 4]。
2. [C] キー入力で全物体をコラージュ台紙化し、VLM (Gemini / Qwen) へ一括推論リクエスト[cite: 4]。
   - 物体カテゴリ、主要色、外観詳細（質感・特徴・状態）を動的同定[cite: 5]。
3. 実行時引数に応じた推論エンジンの切り替え・自動フォールバック[cite: 4]：
   - --gemini        : Gemini API によるクラウド一括同定[cite: 4]
   - --qwen          : ローカル Ollama (qwen2.5vl:3b) による完全オフライン同定[cite: 4]
   - --gemini --qwen : Gemini 試行 ➔ 全モデル失敗時に Qwen へ自動フォールバック[cite: 4]
4. 計測されたミリ寸法に合わせて MuJoCo 上の直方体形状を動的変形し、3D ラベルを重畳表示[cite: 4]。
5. 【LLM タスクプランナー連携】
   - 検出した物理幾何情報と VLM の外観・属性推論結果を統合[cite: 4, 5]。
   - 自然言語指示解釈用の構造化ファイル (config/current_world_state.json) を自動生成・保存。

【操作】
  - [C]     : 全物体を一括同定、MuJoCo 反映 ＆ ワールドステート JSON を出力[cite: 4]
  - [B]     : 机面背景の記憶（高精度差分リフレッシュ）[cite: 4]
  - [SPACE] : ArUco マーカーに基づく正射影キャリブレーションの再計算[cite: 4]
  - [Q/ESC] : 終了[cite: 4]
==============================================================================
"""

import sys
import os
import math
import argparse
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from typing import Dict
import time
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.vision_projector import VisionProjector
from core.sim_viewer import MujocoSimViewer
from core.kinematics import get_home_radians
from core.tabletop_detector import TabletopDetector
from core.multimodal_tagger import MultimodalTagger

MAX_SLOTS = 16
DEFAULT_OBJ_HEIGHT_M = 0.015
HALF_Z = DEFAULT_OBJ_HEIGHT_M / 2.0

REQ_SAVE_BG = False
REQ_RECALIB = False
REQ_CLASSIFY = False
REQ_QUIT = False


def custom_sim_key_callback(keycode: int):
    global REQ_SAVE_BG, REQ_RECALIB, REQ_CLASSIFY, REQ_QUIT
    if keycode in (66, 98):  # B, b
        REQ_SAVE_BG = True
    elif keycode in (67, 99):  # C, c
        REQ_CLASSIFY = True
    elif keycode == 32:  # Space
        REQ_RECALIB = True
    elif keycode in (81, 113, 256):  # Q, q, ESC
        REQ_QUIT = True


def euler_yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def draw_aruco_markers_in_mujoco(sim, projector, marker_size_m: float = 0.04):
    if sim.viewer is None:
        return
    half_s = marker_size_m / 2.0
    half_th = 0.0002

    for marker_idx in range(4):
        phys_x, phys_y = projector.marker_phys_xy[marker_idx]
        mj_x = -phys_y / 1000.0
        mj_y = -phys_x / 1000.0

        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([half_s, half_s, half_th], dtype=np.float64),
                pos=np.array([mj_x, mj_y, half_th], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.9, 0.9, 0.9, 0.95], dtype=np.float32)
            )
            sim.viewer.user_scn.ngeom += 1

        if sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
            ng = sim.viewer.user_scn.ngeom
            lbl_pos = np.array([mj_x, mj_y, 0.025], dtype=np.float64)
            mujoco.mjv_initGeom(
                sim.viewer.user_scn.geoms[ng],
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=np.zeros(3),
                pos=lbl_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.3, 0.8, 1.0, 1.0], dtype=np.float32)
            )
            sim.viewer.user_scn.geoms[ng].label = f"ArUco #{marker_idx}".encode("utf-8")
            sim.viewer.user_scn.ngeom += 1


def main():
    global REQ_SAVE_BG, REQ_RECALIB, REQ_CLASSIFY, REQ_QUIT

    parser = argparse.ArgumentParser(description="Tachikoma Multimodal Digital Twin")
    parser.add_argument("--gemini", action="store_true", help="Gemini API による認識")
    parser.add_argument("--qwen", action="store_true", help="ローカル Qwen による認識")
    args = parser.parse_args()

    use_gemini = args.gemini
    use_qwen = args.qwen
    if not use_gemini and not use_qwen:
        use_gemini = True
        use_qwen = True

    print("==================================================")
    print(" 🌐 マルチモーダル・デジタルツイン同期システム (モジュール統合版)")
    print("==================================================")

    # 1. MuJoCo ビューア初期化
    sim = MujocoSimViewer()
    try:
        sim.update_joints_rad(get_home_radians())
    except Exception:
        pass
    sim.viewer = mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=custom_sim_key_callback
    )

    # スロットのアドレス解決
    slot_info = []
    for i in range(MAX_SLOTS):
        bname = f"obj_block_{i}"
        gname = f"geom_obj_{i}"
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, bname)
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if bid == -1:
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"jenga_block_{i}")
            if bid != -1:
                gid = sim.model.body_geomadr[bid]

        qpos_adr = None
        if bid != -1:
            jnt_adr = sim.model.body_jntadr[bid]
            if jnt_adr != -1:
                qpos_adr = sim.model.jnt_qposadr[jnt_adr]
        slot_info.append({"bid": bid, "gid": gid, "qpos_adr": qpos_adr})

    # 2. カメラ & モジュール群のインスタンス化
    projector = VisionProjector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 👉 モジュールの初期化
    detector = TabletopDetector(projector)
    tagger = MultimodalTagger()

    object_profiles: Dict[int, dict] = {}
    cv2.namedWindow("Digital Twin Multi-modal Profiler")

    try:
        while sim.is_running() and not REQ_QUIT:
            ret, frame = cap.read()
            if not ret:
                break

            if projector.homography_mat is None:
                projector.update_homography(frame)

            warped = projector.warp_to_topdown(frame, out_w=500, out_h=500)
            if warped is None:
                cv2.imshow("Digital Twin Multi-modal Profiler", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                    break
                continue

            # 背景更新
            if REQ_SAVE_BG:
                detector.update_background(warped)
                object_profiles.clear()
                REQ_SAVE_BG = False

            # キャリブレーション更新
            if REQ_RECALIB:
                projector.update_homography(frame)
                REQ_RECALIB = False
                print("🔄 正射影キャリブレーションを更新しました。")

            # 1. 物体検出 (幾何計測)
            detected_objs, _ = detector.detect_objects(warped)

            # 2. VLM 一括同定
            if REQ_CLASSIFY:
                REQ_CLASSIFY = False
                if detected_objs:
                    crops = [obj["crop"] for obj in detected_objs if obj["crop"].size > 0]
                    object_profiles = tagger.identify_items(crops, use_gemini=use_gemini, use_qwen=use_qwen)
                    print("\n📦 === 物体同定結果一覧 ===")
                    for idx, obj in enumerate(detected_objs):
                        name = object_profiles.get(idx, {}).get("display_name", "object")
                        print(f"  [#{idx}] {name} ({obj['size_mm'][0]}x{obj['size_mm'][1]}mm)")

                    # ==============================================================
                    # 👉 【ここを追加】LLM用の机上ワールドステート JSON を生成・保存
                    # ==============================================================
                    world_objects = []
                    for idx, obj in enumerate(detected_objs):
                        prof = object_profiles.get(idx, {})
                        x_mm, y_mm = obj["phys_xy"]
                        major_mm, minor_mm = obj["size_mm"]
                        dist_to_base = math.hypot(x_mm, y_mm)

                        world_objects.append({
                            "id": idx,
                            "display_name": prof.get("display_name", "object"),
                            "category": prof.get("category", "object"),
                            "color": prof.get("color", ""),
                            "description": prof.get("description", ""),  # VLMが返した外観詳細
                            "physical": {
                                "position_xy_mm": [round(x_mm, 1), round(y_mm, 1)],
                                "size_mm": [round(major_mm, 1), round(minor_mm, 1)],
                                "angle_deg": round(obj["angle_deg"], 1),
                                "aspect_ratio": round(major_mm / max(1.0, minor_mm), 2)
                            },
                            "spatial": {
                                "relative_position": ("手前" if x_mm < 250 else "奥") + ("左" if y_mm < -50 else "右" if y_mm > 50 else "中央"),
                                "distance_to_base_mm": round(dist_to_base, 1)
                            }
                        })

                    world_state = {
                        "timestamp": time.time(),
                        "total_objects": len(world_objects),
                        "objects": world_objects,
                        "workspace": {
                            "place_area_xy_mm": [180.0, 160.0]
                        }
                    }

                    # 保存 (次のタスクプランナー LLM が読み込む用)
                    output_json_path = os.path.join(BASE_DIR, "config", "current_world_state.json")
                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(world_state, f, ensure_ascii=False, indent=2)
                    print(f"📄 [World State] LLM 用の机上状態ファイルを保存しました: {output_json_path}")

            # 3. OpenCV 描画
            annotated = detector.draw_annotations(warped, detected_objs, object_profiles)

            # 4. MuJoCo への反映
            if sim.viewer is not None:
                sim.viewer.user_scn.ngeom = 0

            draw_aruco_markers_in_mujoco(sim, projector, marker_size_m=0.04)

            for i in range(MAX_SLOTS):
                sinfo = slot_info[i]
                qadr, gid = sinfo["qpos_adr"], sinfo["gid"]
                if qadr is None or gid == -1:
                    continue

                if i < len(detected_objs):
                    obj = detected_objs[i]
                    x_mm, y_mm = obj["phys_xy"]
                    major_mm, minor_mm = obj["size_mm"]

                    mj_x = -y_mm / 1000.0
                    mj_y = -x_mm / 1000.0
                    yaw_rad = math.radians(-obj["angle_deg"])

                    sim.data.qpos[qadr:qadr + 3] = [mj_x, mj_y, HALF_Z]
                    sim.data.qpos[qadr + 3:qadr + 7] = euler_yaw_to_quat(yaw_rad)

                    # 直方体サイズ動的変更
                    half_x = max(0.005, (major_mm / 1000.0) / 2.0)
                    half_y = max(0.005, (minor_mm / 1000.0) / 2.0)
                    sim.model.geom_size[gid] = [half_x, half_y, HALF_Z]

                    # 3Dラベル描画
                    prof = object_profiles.get(i, {})
                    label_name = prof.get("display_name", f"#{i}")

                    if sim.viewer is not None and sim.viewer.user_scn.ngeom < sim.viewer.user_scn.maxgeom:
                        ngeom = sim.viewer.user_scn.ngeom
                        label_pos = np.array([mj_x, mj_y, HALF_Z + 0.03], dtype=np.float64)
                        mujoco.mjv_initGeom(
                            sim.viewer.user_scn.geoms[ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.zeros(3),
                            pos=label_pos,
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1.0, 1.0, 0.2, 1.0], dtype=np.float32)
                        )
                        sim.viewer.user_scn.geoms[ngeom].label = label_name.encode("utf-8")
                        sim.viewer.user_scn.ngeom += 1
                else:
                    sim.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]
                    sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

            mujoco.mj_forward(sim.model, sim.data)
            if sim.viewer is not None:
                sim.viewer.sync()

            mode_text = "Diff" if detector.bg_gray is not None else "Adaptive"
            cv2.putText(annotated, f"Detected: {len(detected_objs)} | Mode: {mode_text} | Press [C] to Identify",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Digital Twin Multi-modal Profiler", annotated)

            k = cv2.waitKey(1) & 0xFF
            if k in [ord('q'), ord('Q'), 27]:
                break
            elif k in [ord('b'), ord('B')]:
                REQ_SAVE_BG = True
            elif k in [ord('c'), ord('C')]:
                REQ_CLASSIFY = True
            elif k == 32:  # SPACE
                REQ_RECALIB = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()