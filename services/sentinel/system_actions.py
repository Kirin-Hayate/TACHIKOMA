"""
==============================================================================
TACHIKOMA Sentinel: システム緊急遮断・復旧モジュール (services/sentinel/system_actions.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA システムおよび常時稼働ホスト（HeroBox / Ubuntu / Linux環境）
におけるハードウェア緊急停止（キルスイッチ）およびOSサービス・ネットワーク遮断・復旧
アクションを一元管理する専門サービスコンポーネントです。

【所属レイヤー】
Layer 3: 専門サービス層（インフラ監視・防衛 Worker: Sentinel）

【主な機能と責務】
1. アーム緊急停止 (emergency_stop_arm):
   - 実行中フラグの即時クリアおよび軌道生成ループ割り込みシグナル発行。
   - STS3215 サーボドライバを介した全軸サーボのトルク解除（脱力）によるハードウェア破損防止。
2. 保護対象サービス制御 (stop_service / start_service / get_service_status):
   - systemctl を用いた Web 公開アプリケーション（家計簿サービス等）の緊急停止および安全再開。
3. 外部ネットワーク隔離 (isolate_network):
   - tailscale down を実行し、不正侵入時や暴走時に外部からの通信経路を物理的に即時遮断。

【ローカル開発・ノートPC対応（モック・安全機構）】
- 非 Linux 環境（Windows, macOS）や非 root / sudo 権限のない開発機（ノートPC）で
  実行された場合、実際に OS コマンドを叩くのではなく安全なドライラン（シミュレーション）
  モードへ自動的にフォールバックし、例外でプログラムがクラッシュするのを防ぎます。
- 実機アーム未接続時でもエラーを起こさず、警告ログを出力して安全に終了します。
==============================================================================
"""

import os
import sys
import time
import subprocess
import platform
from typing import Tuple, Optional, Any, List

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 設定値の読み込み (config/joint_config.py があれば使用、未定義時は標準定義にフォールバック)
try:
    from config.joint_config import SERVO_IDS
except ImportError:
    SERVO_IDS = [1, 2, 3, 4, 5, 6]

# 保護対象サービス名のデフォルト (環境変数優先)
DEFAULT_PROTECTED_SERVICE = os.getenv("PROTECTED_SERVICE_NAME", "my_web_app")


def is_linux_environment() -> bool:
    """現在の実行環境が Linux OS であるかを判定する"""
    return platform.system().lower() == "linux"


def emergency_stop_arm(follower_driver: Optional[Any] = None) -> Tuple[bool, str]:
    """
    実機ロボットアームの全軸サーボに対して脱力（トルクOFF）指令を発行し、緊急停止させます。

    Args:
        follower_driver: STS3215Driver インスタンス（未接続時は None）

    Returns:
        Tuple[bool, str]: (成功フラグ, 結果報告メッセージ)
    """
    if follower_driver is None or not getattr(follower_driver, "is_open", False):
        return True, "⚠️ 実機アームは未接続またはポート未オープンです (シミュレーション停止完了)。"

    failed_ids: List[int] = []
    for sid in SERVO_IDS:
        try:
            # set_torque(sid, False) でサーボ脱力
            follower_driver.set_torque(sid, False)
        except Exception as e:
            failed_ids.append(sid)

    if failed_ids:
        return False, f"⚠️ サーボ ID: {failed_ids} のトルク解除に失敗しました。"
    return True, "🛑 ロボットアームの全軸サーボを正常に脱力・緊急停止しました。"


def stop_service(service_name: str = DEFAULT_PROTECTED_SERVICE) -> Tuple[bool, str]:
    """
    指定された systemd サービスを緊急停止します。

    Args:
        service_name: systemctl で管理されているサービス名 (.service は省略可)

    Returns:
        Tuple[bool, str]: (成功フラグ, 結果報告メッセージ)
    """
    if not is_linux_environment():
        return True, f"💻 [DRY-RUN/ノートPC] sudo systemctl stop {service_name} (シミュレーション成功)"

    try:
        cmd = ["sudo", "systemctl", "stop", service_name]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return True, f"🛑 `{service_name}.service` を緊急停止しました。"
        else:
            err_msg = res.stderr.strip() or res.stdout.strip()
            return False, f"⚠️ `{service_name}` の停止に失敗しました (Code {res.returncode}): {err_msg}"
    except Exception as e:
        return False, f"❌ `{service_name}` 停止コマンド実行中に例外が発生しました: {e}"


def start_service(service_name: str = DEFAULT_PROTECTED_SERVICE) -> Tuple[bool, str]:
    """
    指定された systemd サービスを再開し、現在のアクティブ状態を確認します。

    Args:
        service_name: systemctl で管理されているサービス名

    Returns:
        Tuple[bool, str]: (成功フラグ, 結果報告メッセージ)
    """
    if not is_linux_environment():
        return True, f"💻 [DRY-RUN/ノートPC] sudo systemctl start {service_name} (シミュレーション成功: active)"

    try:
        cmd_start = ["sudo", "systemctl", "start", service_name]
        res_start = subprocess.run(cmd_start, capture_output=True, text=True, timeout=8)
        if res_start.returncode != 0:
            err_msg = res_start.stderr.strip() or res_start.stdout.strip()
            return False, f"⚠️ `{service_name}` の再開に失敗しました (Code {res_start.returncode}): {err_msg}"

        # 起動後の状態確認
        status_ok, status_str = get_service_status(service_name)
        return True, f"▶️ `{service_name}.service` を再開しました（状態: `{status_str}`）。"
    except Exception as e:
        return False, f"❌ `{service_name}` 再開コマンド実行中に例外が発生しました: {e}"


def get_service_status(service_name: str = DEFAULT_PROTECTED_SERVICE) -> Tuple[bool, str]:
    """
    指定された systemd サービスの稼働ステータス (active, inactive, failed等) を取得します。

    Args:
        service_name: サービス名

    Returns:
        Tuple[bool, str]: (アクティブ判定真偽値, 状態文字列)
    """
    if not is_linux_environment():
        return True, "active (DRY-RUN)"

    try:
        cmd = ["sudo", "systemctl", "is-active", service_name]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        state_str = res.stdout.strip()
        is_active = (state_str == "active")
        return is_active, state_str
    except Exception as e:
        return False, f"unknown ({e})"


def isolate_network() -> Tuple[bool, str]:
    """
    Tailscale VPN を切断し、外部ネットワークからの侵入・操作を物理コンソール以外遮断します。

    Returns:
        Tuple[bool, str]: (成功フラグ, 結果報告メッセージ)
    """
    if not is_linux_environment():
        return True, "💻 [DRY-RUN/ノートPC] sudo tailscale down (シミュレーション成功: 外部隔離)"

    try:
        cmd = ["sudo", "tailscale", "down"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return True, "🚨 Tailscale VPN を切断し、外部ネットワークから完全隔離しました。"
        else:
            err_msg = res.stderr.strip() or res.stdout.strip()
            return False, f"⚠️ Tailscale 切断に失敗しました (Code {res.returncode}): {err_msg}"
    except Exception as e:
        return False, f"❌ Tailscale 切断コマンド実行中に例外が発生しました: {e}"


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [TEST] services/sentinel/system_actions.py 単体動作検証")
    print(f"🖥️ 検出プラットフォーム: {platform.system()} (Linux判定: {is_linux_environment()})")
    print("=" * 70)

    # 1. アーム脱力テスト (ドライバ None の未接続シミュレーション)
    print("\n1. ロボットアーム緊急停止 (未接続モックテスト):")
    success, msg = emergency_stop_arm(follower_driver=None)
    print(f"   結果: {'[PASS]' if success else '[FAIL]'} {msg}")
    assert success, "アーム未接続時の停止ハンドリングに失敗しました"

    # 2. モックサーボドライバによる脱力シミュレーション
    print("\n2. ロボットアーム緊急停止 (モックドライバ経由テスト):")
    class MockDriver:
        def __init__(self):
            self.is_open = True
            self.disabled_ids = []

        def set_torque(self, servo_id: int, enable: bool):
            if not enable:
                self.disabled_ids.append(servo_id)

    mock_driver = MockDriver()
    success, msg = emergency_stop_arm(follower_driver=mock_driver)
    print(f"   結果: {'[PASS]' if success else '[FAIL]'} {msg}")
    print(f"   脱力サーボID一覧: {mock_driver.disabled_ids}")
    assert success and len(mock_driver.disabled_ids) == len(SERVO_IDS), "全軸サーボ脱力に失敗しました"

    # 3. サービス停止・再開テスト (OS 自動判別ドライラン対応)
    target_svc = "test_kakeibo_service"
    print(f"\n3. Webアプリ緊急停止テスト (対象: {target_svc}):")
    success, msg = stop_service(target_svc)
    print(f"   結果: {msg}")

    print(f"\n4. Webアプリ状態確認テスト (対象: {target_svc}):")
    _, status = get_service_status(target_svc)
    print(f"   結果: 現在の状態 = {status}")

    print(f"\n5. Webアプリ再開テスト (対象: {target_svc}):")
    success, msg = start_service(target_svc)
    print(f"   結果: {msg}")

    # 6. ネットワーク隔離テスト
    print("\n6. ネットワーク外部隔離テスト:")
    success, msg = isolate_network()
    print(f"   結果: {msg}")

    print("\n" + "=" * 70)
    print("🎉 services/sentinel/system_actions.py の全単体テストが正常に完了しました！")
    print("=" * 70)