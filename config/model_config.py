"""
==============================================================================
TACHIKOMA モデル設定・一元管理モジュール (config/model_config.py)
==============================================================================
【概要】
本モジュールは、システム全体（LLMTaskPlanner, Orchestrator, 各種解析スクリプト等）で
使用する Google Gemini API の推論モデル名およびフェイルオーバー優先順序を一元管理します。

【設計方針】
1. モデル名のハードコード撲滅:
   各機能モジュールが個別にモデル名を記述することを禁止し、本ファイルで定義された
   リストまたはヘルパー関数を参照させることで、将来のバージョン更新やエイリアス対応を
   本ファイルの書き換えのみで完結させます。

2. 段階的フォールバック（二段構え・多層フォールバック）:
   - 第1候補: 最軽量・最高速の Flash-Lite（日常的な雑談、高速なタスク分解、トークン節約）
   - 第2候補: 高性能・安定版の Flash（複雑な指示、Liteが制限またはエラー時のフォールバック）
   - 第3・第4候補: エイリアス解決に失敗した際や特定バージョン固定時のセーフティネット

3. 環境変数による動的オーバーライド:
   .env に GEMINI_PRIMARY_MODEL や GEMINI_CANDIDATE_MODELS が指定されている場合は
   それを最優先で読み込み、柔軟な実験や差し替えを可能にします。
==============================================================================
"""

import os
from typing import List

# ==============================================================================
# デフォルトモデル候補リスト (現在実動品・安定版の優先順)
# ==============================================================================
# 1. まずは最軽量・高速応答の Flash-Lite
# 2. 失敗時・レートリミット(429)時のフォールバック用標準 Flash
# 3. バージョン固定のセーフティネット
DEFAULT_CANDIDATE_MODELS: List[str] = [
    "gemini-3.5-flash-lite",  # 第1優先: RPD 500枠、最高速
    "gemini-3.1-flash-lite",  # 第2優先: RPD 500枠、フォールバック用
    "gemini-3.6-flash",
]

# 単一モデル利用時（デフォルト推奨モデル）
DEFAULT_PRIMARY_MODEL: str = DEFAULT_CANDIDATE_MODELS[0]


def get_candidate_models() -> List[str]:
    """
    推論試行に使用する Gemini モデル候補のリストを取得します。
    環境変数 `GEMINI_CANDIDATE_MODELS` (カンマ区切り) が設定されている場合はそれを最優先します。

    Returns:
        List[str]: 試行優先順に並んだモデル名のリスト
    """
    env_models = os.getenv("GEMINI_CANDIDATE_MODELS", "").strip()
    if env_models:
        # カンマ区切りの文字列をリスト化して前後の空白を除去
        models = [m.strip() for m in env_models.split(",") if m.strip()]
        if models:
            return models
    return list(DEFAULT_CANDIDATE_MODELS)


def get_primary_model() -> str:
    """
    単一モデル指定が必要な場合に使用する推奨モデル名を取得します。
    環境変数 `GEMINI_PRIMARY_MODEL` が設定されている場合はそれを優先します。

    Returns:
        str: プライマリモデル名
    """
    env_primary = os.getenv("GEMINI_PRIMARY_MODEL", "").strip()
    if env_primary:
        return env_primary
    return DEFAULT_PRIMARY_MODEL


# ==============================================================================
# ノートPC単体テスト・検証用ブロック
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [TEST] config/model_config.py 単体動作確認")
    print("=" * 70)

    # 1. デフォルト状態の取得確認
    primary = get_primary_model()
    candidates = get_candidate_models()
    print(f"✅ プライマリモデル: {primary}")
    print(f"✅ 候補モデルリスト ({len(candidates)}件):")
    for i, model in enumerate(candidates, 1):
        print(f"   [{i}] {model}")

    # 2. 環境変数オーバーライドのシミュレーションテスト
    print("\n--- 環境変数オーバーライド動作テスト ---")
    test_custom_models = "custom-flash-lite, custom-flash"
    os.environ["GEMINI_CANDIDATE_MODELS"] = test_custom_models
    overridden_candidates = get_candidate_models()
    print(f"設定環境変数: GEMINI_CANDIDATE_MODELS='{test_custom_models}'")
    print(f"取得結果: {overridden_candidates}")

    assert overridden_candidates == ["custom-flash-lite", "custom-flash"], "❌ 環境変数オーバーライドに失敗しました。"
    print("✅ 環境変数による候補モデルの動的変更テスト: PASS")

    # クリーンアップ
    del os.environ["GEMINI_CANDIDATE_MODELS"]
    print("\n🎉 すべてのテストが正常に通過しました！")
    print("=" * 70)