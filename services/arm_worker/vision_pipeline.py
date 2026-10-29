"""
==============================================================================
TACHIKOMA 視覚パイプライン ＆ 机上ワールドステート同期モジュール
(services/arm_worker/vision_pipeline.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムの Layer 3 (専門サービス層: 🦾物理制御Worker) に位置し、
机上カメラ画像から物体認識・幾何計測・VLM同定を行い、世界地図 (current_world_state.json)
の原子的更新と、Discord UI 向けスナップショット生成・アノテーション描画を一括提供します。

【所属レイヤー】
Layer 3: 専門サービス層 (物理制御 Worker: arm_worker)

【主な機能と責務】
1. カメラハードウェア抽象化 (drivers.camera_driver.CameraDriver 連携):
   - 排他ロック制御により、Discordスナップショットと物体認識処理の競合を防止。
   - 露光安定化フレーム読み捨てを経てクリアな最新フレームを取得。
2. 幾何正射影・物体検出・セマンティック同定:
   - core.vision_projector.VisionProjector による ArUco マーカー検出と真上視点透視変換 (Warp)。
   - core.tabletop_detector.TabletopDetector による机上輪郭抽出、重心座標、ミリ寸法、把持角度の算出。
   - core.multimodal_tagger.MultimodalTagger による Gemini/Qwen VLM 動的コラージュ一括同定。
3. 机上ワールドステート JSON (config/current_world_state.json) の安全・原子的保存:
   - 要求された「いつ更新したデータか?」を示す UNIXエポック秒 (timestamp) および
     ISO 8601 形式の人間可読文字列 (updated_at) をヘッダーに格納。
   - 一時ファイル書き出し後の rename によるアトミック（原子的）保存で破損・競合を防止。
4. ノートPC / 開発機対応 (モック・フェイルファスト):
   - 実機カメラや ArUco マーカーが存在しない環境でもクラッシュせず、
     安全なエラーハンドリングまたはテスト用疑似データを返却。
==============================================================================
"""

import os
import sys
import time
import json
import math
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 依存モジュールのインポート
from drivers.camera_driver import CameraDriver
from config.model_config import get_candidate_models
from core.vision_projector import VisionProjector
from core.tabletop_detector import TabletopDetector
from core.multimodal_tagger import MultimodalTagger

# 保存先パスおよび定数設定
WORLD_STATE_PATH = os.path.join(BASE_DIR, "config", "current_world_state.json")
DEFAULT_PLACE_X_MM = 180.0
DEFAULT_PLACE_Y_MM = 160.0


class VisionPipeline:
    """
    机上の視覚認識とワールドステート同期を統括するサービスパイプライン
    """
    def __init__(
        self,
        camera_driver: Optional[CameraDriver] = None,
        world_state_path: str = WORLD_STATE_PATH,
        enable_vlm: bool = True
    ):
        """
        Args:
            camera_driver: 抽象化カメラインスタンス (None時は新規生成)
            world_state_path: ワールドステートJSONの保存先
            enable_vlm: VLM (Gemini/Qwen) による物体セマンティック同定を有効化するか
        """
        self.camera = camera_driver or CameraDriver(device_index=0, width=1280, height=720, warmup_frames=5)
        self.world_state_path = world_state_path
        self.enable_vlm = enable_vlm

        # コア認識モジュール初期化
        self.projector = VisionProjector()
        self.detector = TabletopDetector(self.projector)

        # モデル設定一元管理 (config/model_config.py) から候補モデルを取得して Tagger に渡す
        candidate_models = get_candidate_models()
        self.tagger = MultimodalTagger(gemini_candidate_models=candidate_models)

    def capture_annotated_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Discord Bot 送信用に、現在の生カメラ視界（または正射影重畳画像）を取得して
        JPEG バイト列を返す。
        """
        frame, err = self.camera.capture_frame()
        if frame is None:
            return self.camera.capture_jpeg_bytes()

        # マーカーが検出できればマーカー枠を描画
        annotated = frame.copy()
        try:
            centers = self.projector.detect_markers(frame)
            for mid, pt in centers.items():
                u, v = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (u, v), 6, (0, 255, 255), -1)
                cv2.putText(annotated, f"ArUco #{mid}", (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception:
            pass

        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return b"", "スナップショットのJPEGエンコードに失敗しました。"
        return buf.tobytes(), None

    def scan_and_sync_world_state(
        self,
        image_path: Optional[str] = None,   # 👈 追加: テスト用画像ファイルパス
        use_gemini: bool = True,
        use_qwen: bool = False
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        カメラ（または指定静止画）から机面をスキャンし、正射影変換・物体検出・VLM同定を実行して
        current_world_state.json を原子的（アトミック）に更新・保存する。
        """
        now_ts = time.time()
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        base_empty_state = {
            "timestamp": now_ts,
            "updated_at": now_iso,
            "total_objects": 0,
            "objects": [],
            "workspace": {
                "place_area_xy_mm": [DEFAULT_PLACE_X_MM, DEFAULT_PLACE_Y_MM]
            }
        }

        # 1. フレームの取得 (テスト用画像ファイル指定時はそれを読み込む)
        if image_path is not None:
            if not os.path.exists(image_path):
                self._save_world_state_atomic(base_empty_state)
                msg = f"⚠️ 指定されたテスト画像が見つかりません: {image_path}"
                return False, base_empty_state, msg
            frame = cv2.imread(image_path)
            if frame is None:
                self._save_world_state_atomic(base_empty_state)
                msg = f"⚠️ テスト画像の読み込みに失敗しました (破損の可能性): {image_path}"
                return False, base_empty_state, msg
            print(f"🖼️ [TEST MODE] 静止画ファイルからテスト実行中: {image_path} (解像度: {frame.shape[1]}x{frame.shape[0]})")
        else:
            # 通常モード: 実機カメラからキャプチャ
            frame, err = self.camera.capture_frame()
            if frame is None:
                self._save_world_state_atomic(base_empty_state)
                msg = f"⚠️ カメラフレーム取得失敗: {err}"
                return False, base_empty_state, msg

        # 2. ホモグラフィ更新 ＆ 正射影変換 (Top-down Warp)
        if self.projector.homography_mat is None:
            updated = self.projector.update_homography(frame)
            if not updated:
                # マーカー不足時も最新スキャン日時付きで空ステートを保存
                self._save_world_state_atomic(base_empty_state)
                msg = "⚠️ ArUcoマーカー (ID 0〜3) が不足しているため、ホモグラフィ更新に失敗しました。"
                return False, base_empty_state, msg

        warped = self.projector.warp_to_topdown(frame, out_w=500, out_h=500)
        if warped is None:
            self._save_world_state_atomic(base_empty_state)
            msg = "⚠️ 正射影変換 (Warp) に失敗しました。"
            return False, base_empty_state, msg

        # 3. 机上物体・幾何輪郭検出 (OpenCV)
        detected_objs, _ = self.detector.detect_objects(warped)
        if not detected_objs:
            self._save_world_state_atomic(base_empty_state)
            return True, base_empty_state, "ℹ️ 机上に物体は検出されませんでした (空のワールドステートを保存)。"

        # 4. VLM による物体同定 (Gemini / Qwen)
        crops = [obj["crop"] for obj in detected_objs if obj["crop"].size > 0]
        object_profiles = {}
        if self.enable_vlm and crops:
            try:
                object_profiles = self.tagger.identify_items(crops, use_gemini=use_gemini, use_qwen=use_qwen)
            except Exception as e:
                print(f"⚠️ VLM同定中にエラーが発生しました (幾何情報のみで継続します): {e}")

        # 5. ワールドステート JSON データの構築
        world_objects = []
        for idx, obj in enumerate(detected_objs):
            prof = object_profiles.get(idx, {})
            x_mm, y_mm = obj["phys_xy"]
            major_mm, minor_mm = obj["size_mm"]
            dist_to_base = math.hypot(x_mm, y_mm)
            display_name = prof.get("display_name", f"object_{idx}")

            # 相対位置表現
            fwd_str = "手前" if x_mm < 250 else "奥"
            lat_str = "左" if y_mm < -50 else "右" if y_mm > 50 else "中央"

            world_objects.append({
                "id": idx,
                "name": display_name,
                "display_name": display_name,
                "category": prof.get("category", "object"),
                "color": prof.get("color", "不明"),
                "description": prof.get("description", ""),
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": 5.0,  # 机上把持の基準高さ (5mm)
                "angle_deg": obj["angle_deg"],
                "physical": {
                    "position_xy_mm": [x_mm, y_mm],
                    "size_mm": [major_mm, minor_mm],
                    "angle_deg": obj["angle_deg"],
                    "aspect_ratio": round(major_mm / max(1.0, minor_mm), 2)
                },
                "spatial": {
                    "relative_position": fwd_str + lat_str,
                    "distance_to_base_mm": round(dist_to_base, 1)
                }
            })

        world_state = {
            "timestamp": now_ts,
            "updated_at": now_iso,
            "total_objects": len(world_objects),
            "objects": world_objects,
            "workspace": {
                "place_area_xy_mm": [DEFAULT_PLACE_X_MM, DEFAULT_PLACE_Y_MM]
            }
        }

        # 6. JSON の原子的書き込み (Atomic Write)
        self._save_world_state_atomic(world_state)
        success_msg = f"✅ 机上物体スキャン完了: {len(world_objects)} 個の物体を認識・保存しました (更新日時: {now_iso})。"
        return True, world_state, success_msg

    def _save_world_state_atomic(self, state_data: Dict[str, Any]):
        """
        競合やクラッシュによる破損を防ぐため、一時ファイルに完全出力してから
        rename により原子的 (atomic) に置換保存する。
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.world_state_path)), exist_ok=True)
        dir_name = os.path.dirname(os.path.abspath(self.world_state_path))
        
        # 同一ファイルシステム上で一時ファイルを作成
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(state_data, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name

        # アトミックに置き換え
        os.replace(temp_name, self.world_state_path)


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VisionPipeline 単体テスト")
    parser.add_argument("--image", type=str, default=None, help="テスト用机面写真のパス (例: test_desk.jpg)")
    args = parser.parse_args()

    print("=" * 70)
    print("🧪 [TEST] services/arm_worker/vision_pipeline.py 単体動作検証")
    print("=" * 70)

    # 修正後: VLM (Gemini) を有効化
    pipeline = VisionPipeline(enable_vlm=True)

    print("\n机上認識 ＆ ワールドステートJSON同期テスト:")
    success, state, msg = pipeline.scan_and_sync_world_state(
        image_path=args.image,
        use_gemini=True
    )
    
    print(f"実行結果: {msg}")
    print(f"成否: {success}")
    print(f"検出物体数: {state.get('total_objects')} 個")
    print(f"更新日時: {state.get('updated_at')}")
    
    if state.get("objects"):
        print("\n--- 検出された物体一覧 ---")
        for obj in state["objects"]:
            print(f"・[ID #{obj['id']}] {obj['name']} | 座標: ({obj['x_mm']:.1f}, {obj['y_mm']:.1f}, {obj['z_mm']:.1f}) | 角度: {obj['angle_deg']:.1f}°")

    print("\n" + "=" * 70)