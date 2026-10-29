"""
==============================================================================
TACHIKOMA カメラデバイス抽象化ドライバ (drivers/camera_driver.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムの Layer 4 (ハードウェア・OS抽象化層) に位置し、
USB Webカメラ (/dev/video* または DirectShow) へのアクセスを一元管理・抽象化します。

【所属レイヤー】
Layer 4: ハードウェア・OS抽象化層 (Camera Driver)

【設計方針と責務】
1. 排他的・スレッドセーフなカメラアクセス:
   - threading.Lock により、Discord Bot のスナップショット撮影、MJPEGストリーミング、
     および VisionPipeline の机上物体スキャンが同時にカメラを取り合ってデバイスビジー
     (Device or resource busy) や競合クラッシュを起こすことを完全に防止。
2. オンデマンド省電力＆即時解放:
   - カメラを常時開きっぱなしにせず、撮影要求時にのみオープンし、露出安定のための
     数フレーム読み捨て後にキャプチャして即座に release() する安全設計。
3. クロスプラットフォーム自動適応:
   - Linux (HeroBox/Xubuntu) では cv2.CAP_V4L2、Windows (開発ノートPC) では cv2.CAP_DSHOW
     を自動選択。
4. ノートPC / デバイス不在時のフェイルファスト＆モック画像生成:
   - カメラ未接続時や権限不足時でも例外でクラッシュせず、状況を明記したエラー案内
     JPEG画像（プレースホルダー）を生成して返却。
==============================================================================
"""

import os
import sys
import time
import glob
import platform
import threading
from typing import Optional, Tuple, Union
import cv2
import numpy as np

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# カメラ設定 (環境変数または既定値)
DEFAULT_CAMERA_INDEX = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
DEFAULT_CAMERA_PATH = os.getenv("CAMERA_DEVICE_PATH", f"/dev/video{DEFAULT_CAMERA_INDEX}")
IS_LINUX = platform.system().lower() == "linux"


class CameraDriver:
    """
    スレッドセーフなUSBカメラアクセスを提供するドライバクラス
    """
    _global_lock = threading.Lock()

    def __init__(
        self,
        device_index: int = DEFAULT_CAMERA_INDEX,
        device_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        warmup_frames: int = 5
    ):
        """
        Args:
            device_index: カメラデバイスインデックス (例: 0)
            device_path: Linux /dev/video* パス (未指定時は device_index から生成)
            width: キャプチャ解像度 (幅)
            height: キャプチャ解像度 (高さ)
            warmup_frames: 露出・ホワイトバランス安定のための読み捨てフレーム数
        """
        self.device_index = device_index
        self.device_path = device_path or (f"/dev/video{device_index}" if IS_LINUX else None)
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames

    def is_device_available(self) -> Tuple[bool, str]:
        """
        カメラデバイスが物理的・権限的にアクセス可能かを事前検査する
        """
        if IS_LINUX:
            target_path = self.device_path or f"/dev/video{self.device_index}"
            if not os.path.exists(target_path):
                return False, f"カメラデバイス '{target_path}' が存在しません。"
            if not os.access(target_path, os.R_OK | os.W_OK):
                return False, f"カメラデバイス '{target_path}' のアクセス権限がありません (Permission denied)。"
            return True, f"カメラデバイス '{target_path}' はアクセス可能です。"
        else:
            # Windows / macOS はインデックスによる判定
            return True, f"カメラ (Index: {self.device_index}) の確認準備完了 (非Linux環境)"

    def create_error_image(
        self,
        message: str,
        sub_message: str = "Camera Offline / Fail-Fast Mode",
        w: int = 640,
        h: int = 480
    ) -> np.ndarray:
        """
        カメラ接続不可時やエラー時に返すプレースホルダー画像を生成する
        """
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 35)  # ダークグレー背景

        # 警告ヘッダー
        cv2.putText(img, "CAMERA OFFLINE", (int(w * 0.15), int(h * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
        cv2.putText(img, sub_message, (int(w * 0.15), int(h * 0.48)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # エラー詳細
        short_msg = (message or "Unknown Error")[:50]
        cv2.putText(img, short_msg, (int(w * 0.08), int(h * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 180, 255), 1)
        return img

    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        カメラを1回だけオープンし、露光安定化の読み捨てを経て最新の1フレーム (BGR numpy配列) を取得して解放する。

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: (取得フレーム, エラーメッセージ)
        """
        # 事前存在確認
        ok, reason = self.is_device_available()
        if not ok:
            return None, reason

        with self._global_lock:
            cap = None
            try:
                # プラットフォームに合わせたバックエンドの選択
                if IS_LINUX:
                    target = self.device_path or self.device_index
                    cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
                else:
                    cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

                if not cap.isOpened():
                    # フォールバック (デフォルトバックエンドで再試行)
                    cap = cv2.VideoCapture(self.device_index)

                if not cap.isOpened():
                    return None, f"カメラ (Index: {self.device_index}) のオープンに失敗しました (他プロセスが使用中の可能性)。"

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                # 露出・ホワイトバランス安定のための読み捨て
                for _ in range(self.warmup_frames):
                    cap.read()
                    time.sleep(0.02)

                ret, frame = cap.read()
                if not ret or frame is None:
                    return None, "フレームの読み取り (cap.read) に失敗しました。"

                return frame, None

            except Exception as e:
                return None, f"キャプチャ実行中例外: {e}"
            finally:
                if cap is not None:
                    cap.release()

    def capture_jpeg_bytes(self, quality: int = 80) -> Tuple[bytes, Optional[str]]:
        """
        Discord Bot 送信用や MJPEG 配信向けに、JPEGエンコード済みのバイト列を取得する。
        カメラが利用できない場合はエラープレースホルダー画像をJPEG化して返す。

        Returns:
            Tuple[bytes, Optional[str]]: (JPEGバイト列, エラーメッセージ)
        """
        frame, err = self.capture_frame()
        if frame is None:
            err_img = self.create_error_image(err or "Failed to capture")
            ok, buf = cv2.imencode(".jpg", err_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return buf.tobytes() if ok else b"", err

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            err_img = self.create_error_image("JPEG Encoding Error")
            _, buf_err = cv2.imencode(".jpg", err_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return buf_err.tobytes(), "フレームのJPEGエンコードに失敗しました。"

        return buf.tobytes(), None


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [TEST] drivers/camera_driver.py 単体動作検証")
    print(f"🖥️ 検出プラットフォーム: {platform.system()} (Linux判定: {IS_LINUX})")
    print("=" * 70)

    driver = CameraDriver(device_index=0, width=640, height=480, warmup_frames=2)

    # 1. デバイス利用可否チェック
    print("\n1. カメラデバイスの事前チェック:")
    avail, status_msg = driver.is_device_available()
    print(f"   利用可能判定: {avail}")
    print(f"   詳細メッセージ: {status_msg}")

    # 2. フレームキャプチャ試行
    print("\n2. フレームキャプチャ試行:")
    t0 = time.time()
    frame, err = driver.capture_frame()
    elapsed = time.time() - t0
    if frame is not None:
        print(f"   ✅ キャプチャ成功! 解像度: {frame.shape[1]}x{frame.shape[0]}, 所要時間: {elapsed:.2f}秒")
    else:
        print(f"   ⚠️ 実機カメラ未接続 (またはオープン失敗): {err}")
        print("   💻 フェイルファスト機能が正常に動作しています。")

    # 3. JPEG バイト列取得 & モック画像フォールバック確認
    print("\n3. JPEG バイト列取得テスト:")
    jpg_bytes, err = driver.capture_jpeg_bytes()
    print(f"   JPEGサイズ: {len(jpg_bytes)} bytes")
    if err:
        print(f"   エラーハンドリング結果: {err} (※プレースホルダー画像を生成)")
    assert len(jpg_bytes) > 0, "JPEGバイト列の生成に失敗しました"
    print("   ✅ JPEGデータ生成確認: PASS")

    # 4. エラー画像単体生成テスト
    print("\n4. プレースホルダー画像描画テスト:")
    dummy_img = driver.create_error_image("Device Not Found /dev/video0")
    assert dummy_img.shape == (480, 640, 3), "エラー画像サイズが不正です"
    print(f"   ✅ エラー画像生成確認: {dummy_img.shape} PASS")

    print("\n" + "=" * 70)
    print("🎉 drivers/camera_driver.py の全単体テストが正常に完了しました！")
    print("=" * 70)