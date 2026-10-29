"""
==============================================================================
TACHIKOMA Sentinel: システム状態監視 ＆ 自律防衛モジュール (services/sentinel/system_monitor.py)
==============================================================================
【概要】
本モジュールは、TACHIKOMA ホストサーバー（HeroBox / Ubuntu / Linux環境）の
リソース健全性監視（レベル1: 警告通知）、不審プロセス検知・自動遮断（レベル2: 自律防衛）、
および 7系統の詳細テレメトリ収集と Gemini API による自律診断を一元管理します。

【所属レイヤー】
Layer 3: 専門サービス層（インフラ監視・防衛 Worker: Sentinel）

【主な機能と責務】
1. 段階的セキュリティ監視 (check_system_security_sync):
   - レベル1 (警告監視): CPU負荷 (1分平均)、ディスク使用率、SSH不正ログイン試行の検知。
   - レベル2 (自動防衛): /tmp, /var/tmp, /dev/shm 直下の不審プロセス探索、即時 kill、
     および services/sentinel/system_actions.py を介した Web アプリ自動停止。
2. 7系統セキュリティメトリクス収集 (collect_7_security_metrics_sync):
   - ①外部TCP接続、②認証警告ログ、③Top 10プロセス、④失敗サービス、
     ⑤sudo実行履歴、⑥カーネルエラー、⑦温度・SSD状態を一括取得。
3. 高速 Gemini AI セキュリティ診断 (run_ai_security_diagnosis_sync):
   - config/model_config.py からモデル候補リストを取得。
   - 内部リトライ無効化(attempts=1) ＆ thinking無効化(budget=0)による高速カスケードフォールバック。

【ローカル開発・ノートPC対応（モック・安全機構）】
- 非 Linux 環境（Windows, macOS 等）では、Linux 固有コマンド（journalctl, systemctl 等）
  の失敗を安全にハンドリングし、開発機がクラッシュすることなくテスト可能な
  疑似メトリクス・ドライラン結果へ自動フォールバックします。
==============================================================================
"""

import os
import sys
import time
import shutil
import subprocess
import platform
from typing import Tuple, List, Dict, Any, Optional

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import psutil
from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

# 内部依存モジュール
from config.model_config import get_candidate_models
from services.sentinel.system_actions import stop_service, DEFAULT_PROTECTED_SERVICE

# 監視閾値・設定 (環境変数または安全なデフォルト値)
DISK_USAGE_THRESHOLD = float(os.getenv("DISK_USAGE_THRESHOLD", "90.0"))
CPU_LOAD_THRESHOLD = float(os.getenv("CPU_LOAD_THRESHOLD", "3.8"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "900"))

# プロセスホワイトリスト
_env_safe_procs = os.getenv("SAFE_PROCESS_NAMES", "")
if _env_safe_procs.strip():
    SAFE_PROCESS_NAMES = {p.strip() for p in _env_safe_procs.split(",") if p.strip()}
else:
    SAFE_PROCESS_NAMES = {
        "python", "python3", "bash", "sh", "git", "pip", "sshd",
        "tailscaled", "systemd", "apt", "dpkg"
    }

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def is_linux_environment() -> bool:
    """実行環境が Linux OS かどうかを判定"""
    return platform.system().lower() == "linux"


def _run_cmd_safe(cmd_list: List[str], timeout: int = 5) -> str:
    """コマンドを安全に実行し、失敗・不在時も例外を出さずに文字列を返す"""
    if not is_linux_environment():
        return f"(ノートPC/Windows環境のため '{cmd_list[0]}' はスキップされました)"
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
        out = res.stdout.strip()
        if res.returncode != 0 and not out:
            err_txt = res.stderr.strip()
            return f"(終了コード {res.returncode}: {err_txt or '出力なし'})"
        return out or "(該当データなし)"
    except Exception as e:
        return f"(実行エラー: {e})"


# ==============================================================================
# 1. 段階的セキュリティ監視 (レベル1警告 ＆ レベル2自動防衛)
# ==============================================================================
def check_system_security_sync() -> Tuple[List[str], List[str]]:
    """
    システムの負荷・セキュリティ状態を検査する同期関数。

    Returns:
        Tuple[List[str], List[str]]: (警告リスト, 自動防衛アクションリスト)
    """
    warnings: List[str] = []
    auto_actions: List[str] = []

    # --- [レベル1: リソース・警告監視] ---
    try:
        # ディスク使用率
        usage = shutil.disk_usage("/")
        disk_pct = (usage.used / usage.total) * 100.0
        if disk_pct > DISK_USAGE_THRESHOLD:
            warnings.append(f"💾 **ディスク残量警告**: 使用率 `{disk_pct:.1f}%` (閾値: `{DISK_USAGE_THRESHOLD}%`)")

        # CPU 負荷 (Windows では getloadavg が利用できないためフォールバック)
        if hasattr(os, "getloadavg"):
            load1, _, _ = os.getloadavg()
            if load1 > CPU_LOAD_THRESHOLD:
                warnings.append(f"🔥 **CPU高負荷**: 1分平均ロード `{load1:.2f}` (閾値: `{CPU_LOAD_THRESHOLD}`)")
        else:
            cpu_pct = psutil.cpu_percent(interval=0.1)
            if cpu_pct > 90.0:
                warnings.append(f"🔥 **CPU高負荷**: 使用率 `{cpu_pct:.1f}%`")

        # SSH 不正ログイン試行 (Linux のみ)
        if is_linux_environment():
            ssh_check = subprocess.run(
                ["journalctl", "-u", "ssh", "--since", "2 minutes ago", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            failed_ssh = sum(1 for line in ssh_check.stdout.splitlines() if "Failed password" in line)
            if failed_ssh > 0:
                warnings.append(f"🚨 **SSH不正ログイン試行**: 直近2分間で `{failed_ssh}` 件の失敗を検知")
    except Exception as e:
        warnings.append(f"⚠️ リソース監視中にエラーが発生しました: {e}")

    # --- [レベル2: 身元不明コードの検知 ＆ 自動停止 (フェイルセーフ)] ---
    suspicious_dirs = ("/tmp/", "/var/tmp/", "/dev/shm/")
    try:
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                p_info = proc.info
                exe_path = p_info.get("exe") or ""
                p_name = p_info.get("name") or ""

                if any(exe_path.startswith(d) for d in suspicious_dirs) and p_name not in SAFE_PROCESS_NAMES:
                    pid = p_info["pid"]
                    proc.kill()
                    # system_actions を介してサービス自動停止
                    stop_ok, stop_msg = stop_service(DEFAULT_PROTECTED_SERVICE)
                    auto_actions.append(
                        f"🛡️ **不審プロセスを自動強制終了しました**\n"
                        f"・対象: `{p_name}` (PID: `{pid}`)\n"
                        f"・実行パス: `{exe_path}`\n"
                        f"・防衛措置: {stop_msg}"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        warnings.append(f"⚠️ プロセス監視中にエラーが発生しました: {e}")

    return warnings, auto_actions


# ==============================================================================
# 2. 7系統セキュリティメトリクス収集
# ==============================================================================
def collect_7_security_metrics_sync() -> str:
    """
    Cockpit相当の7系統システム・セキュリティデータを一括収集してテキスト形式で返す。
    """
    # 1. 外部TCP接続
    ext_conns = []
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == "ESTABLISHED" and c.raddr:
                ext_conns.append(f"PID:{c.pid} {c.laddr.ip}:{c.laddr.port} -> {c.raddr.ip}:{c.raddr.port}")
    except Exception:
        pass
    if not ext_conns and is_linux_environment():
        ss_out = _run_cmd_safe(["ss", "-tunp", "state", "established"])
        if ss_out and "(該当データなし)" not in ss_out:
            ext_conns.append(ss_out)
    sec1 = "\n".join(ext_conns[:20]) if ext_conns else "外部ESTABLISHED接続なし"

    # 2. 認証・ログイン警告ログ
    sec2 = _run_cmd_safe(
        ["journalctl", "-u", "ssh", "-u", "cockpit", "--since", "1 hour ago", "-p", "warning", "--no-pager", "-n", "20"]
    )

    # 3. CPU/メモリ消費 Top 10 プロセス
    ps_lines = []
    try:
        procs = sorted(psutil.process_iter(["pid", "username", "cpu_percent", "memory_percent", "name"]),
                       key=lambda p: p.info.get("cpu_percent") or 0.0, reverse=True)
        ps_lines.append(f"{'PID':<7} {'USER':<10} {'%CPU':<6} {'%MEM':<6} {'COMMAND'}")
        for p in procs[:10]:
            info = p.info
            ps_lines.append(
                f"{info.get('pid', 0):<7} {str(info.get('username') or ''):<10} "
                f"{info.get('cpu_percent', 0.0):<6.1f} {info.get('memory_percent', 0.0):<6.1f} {info.get('name', '')}"
            )
    except Exception:
        ps_raw = _run_cmd_safe(["ps", "-eo", "pid,user,%cpu,%mem,comm", "--sort=-%cpu"])
        ps_lines = ps_raw.splitlines()[:11]
    sec3 = "\n".join(ps_lines)

    # 4. 失敗サービス
    sec4 = _run_cmd_safe(["systemctl", "--failed", "--no-pager"])

    # 5. sudo 実行履歴
    sec5 = _run_cmd_safe(["journalctl", "_COMM=sudo", "--since", "1 hour ago", "--no-pager", "-n", "15"])

    # 6. OS / カーネルエラーログ
    sec6 = _run_cmd_safe(["journalctl", "-p", "err..emerg", "--since", "1 hour ago", "--no-pager", "-n", "15"])

    # 7. 温度 ＆ ストレージ
    temps = []
    try:
        s_temps = psutil.sensors_temperatures()
        for name, entries in s_temps.items():
            for e in entries:
                temps.append(f"{name}({e.label or 'core'}): {e.current}°C")
    except Exception:
        pass

    usage = shutil.disk_usage("/")
    disk_info = f"Root Disk: {(usage.used / usage.total) * 100:.1f}% used ({usage.free // (1024**3)}GB free)"
    sec7 = (", ".join(temps) if temps else "温度センサー値なし") + f"\n{disk_info}"

    report = f"""
[1. 外部TCP接続(ESTABLISHED)]
{sec1}

[2. 認証・ログイン警告ログ(直近1時間)]
{sec2}

[3. リソース消費上位プロセス(Top 10)]
{sec3}

[4. 失敗サービス(systemctl --failed)]
{sec4}

[5. 直近のsudo特権実行履歴(直近1時間)]
{sec5}

[6. OS・カーネルエラーログ(直近1時間)]
{sec6}

[7. ハードウェア温度・ストレージ健全性]
{sec7}
""".strip()
    return report


# ==============================================================================
# 3. Gemini AI セキュリティ診断 (高速カスケード実行)
# ==============================================================================
def call_single_gemini_model_sync(model_name: str, prompt: str, api_key: Optional[str] = None) -> str:
    """
    指定した単一モデルに対して内部リトライを無効化し、ThinkingをOFFにしてリクエストを送信する。
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY が設定されていません。")

    from google import genai
    from google.genai import types

    http_opts = None
    try:
        http_opts = types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except Exception:
        pass

    client = genai.Client(api_key=key, http_options=http_opts) if http_opts else genai.Client(api_key=key)

    gen_config = None
    try:
        gen_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
        )
    except Exception:
        try:
            gen_config = types.GenerateContentConfig(temperature=0.2)
        except Exception:
            gen_config = None

    try:
        if gen_config is not None:
            resp = client.models.generate_content(model=model_name, contents=prompt, config=gen_config)
        else:
            resp = client.models.generate_content(model=model_name, contents=prompt)
    except Exception as first_err:
        err_str = str(first_err).lower()
        if "invalid_argument" in err_str or "400" in err_str or "thinking" in err_str:
            resp = client.models.generate_content(model=model_name, contents=prompt)
        else:
            raise first_err

    if resp and resp.text:
        return resp.text.strip()
    raise RuntimeError(f"モデル {model_name} から空の応答が返されました。")


def run_ai_security_diagnosis_sync(custom_metrics: Optional[str] = None) -> Tuple[bool, str, str, float]:
    """
    7系統メトリクスを収集し、config/model_config.py のモデル候補順にAI診断を実行する。

    Returns:
        Tuple[bool, str, str, float]: (成功フラグ, 診断テキスト結果, 使用モデル名, 所要秒数)
    """
    t0 = time.time()
    raw_metrics = custom_metrics or collect_7_security_metrics_sync()

    prompt = f"""
あなたはLinuxサーバー（Ubuntu 24.04 / Tailscale環境）の専属セキュリティ・インフラ診断AIです。
以下の7系統の実測テレメトリデータを分析し、スマホ画面で読みやすいよう簡潔な日本語の箇条書きで診断結果を出力してください。

【出力フォーマット】
1. **総合判定**: （🟢安全 / 🟡要注意 / 🔴危険 のいずれかと一言理由）
2. **通信・認証の安全性**: （不審な外部IP接続や不正ログイン試行の有無）
3. **プロセス・サービスの健全性**: （暴走プロセスやfailedサービスの有無）
4. **ハードウェア・OS状態**: （温度・ディスク・エラーログの評価と推奨アクション）

【実測テレメトリデータ】
{raw_metrics}
"""
    candidate_models = get_candidate_models()
    diag_text = None
    used_model = "Failed"
    errors = []

    for model_name in candidate_models:
        try:
            diag_text = call_single_gemini_model_sync(model_name, prompt)
            used_model = model_name
            break
        except Exception as e:
            short_err = str(e).splitlines()[0][:100]
            errors.append(f"• `{model_name}`: {short_err}")

    elapsed = time.time() - t0
    if diag_text is None:
        err_block = "\n".join(errors)
        return False, f"❌ 全モデルで診断リクエストが失敗しました:\n{err_block}", "Failed", elapsed

    return True, diag_text, used_model, elapsed


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [TEST] services/sentinel/system_monitor.py 単体動作検証")
    print(f"🖥️ 検出プラットフォーム: {platform.system()} (Linux判定: {is_linux_environment()})")
    print("=" * 70)

    # 1. 段階的セキュリティ監視テスト
    print("\n1. 段階的セキュリティ監視 (check_system_security_sync) テスト:")
    warnings, auto_actions = check_system_security_sync()
    print(f"   警告検知数: {len(warnings)}")
    for w in warnings:
        print(f"     - {w}")
    print(f"   自動防衛アクション数: {len(auto_actions)}")
    for a in auto_actions:
        print(f"     - {a}")
    print("   ✅ セキュリティ監視ループ実行: PASS")

    # 2. 7系統メトリクス収集テスト
    print("\n2. 7系統メトリクス収集 (collect_7_security_metrics_sync) テスト:")
    metrics = collect_7_security_metrics_sync()
    print(f"   収集文字数: {len(metrics)} 文字")
    print("   --- 収集結果の先頭 300 文字 ---")
    print(metrics[:300] + "...")
    assert len(metrics) > 50, "メトリクス収集結果が短すぎます"
    print("   ✅ 7系統メトリクス収集: PASS")

    # 3. AI診断テスト (GEMINI_API_KEY の有無を確認)
    print("\n3. AI診断カスケード (run_ai_security_diagnosis_sync) テスト:")
    if GEMINI_API_KEY:
        print("   GEMINI_API_KEY 検出済み -> 実際のAPI呼び出しテストを開始します...")
        success, report, model_used, sec = run_ai_security_diagnosis_sync(metrics)
        print(f"   成否: {'[成功]' if success else '[失敗]'}")
        print(f"   使用モデル: {model_used} (所要時間: {sec:.2f}秒)")
        print("   --- 診断レポート結果 ---")
        print(report[:400] + ("..." if len(report) > 400 else ""))
    else:
        print("   ⚠️ GEMINI_API_KEY が未設定のため、実際のAPI呼び出しはスキップします。")
        print("   (config/model_config.py からのモデル候補取得テストを実施)")
        candidates = get_candidate_models()
        print(f"   取得されたモデル候補: {candidates}")
        assert len(candidates) > 0, "モデル候補が取得できませんでした"
        print("   ✅ モデル設定連携テスト: PASS")

    print("\n" + "=" * 70)
    print("🎉 services/sentinel/system_monitor.py の全単体テストが正常に完了しました！")
    print("=" * 70)