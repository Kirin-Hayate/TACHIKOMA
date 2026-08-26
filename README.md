# Project TACHIKOMA: LLM & Voice-Enabled Robot Arm System

SO-ARM100 をベースに、攻殻機動隊に登場する「タチコマ」のような、**自然言語・音声対話で人間の活動を支援する知能型ロボットアームシステム**を開発するプロジェクトです。

実機リーダー・フォロワーによるテレオペレーション、MuJoCo による 3D デジタルツイン（シミュレータ）、そして Gemini API を用いたタスクプランニング（自然言語による動作シーケンスの自動生成とプレビュー・実機実行）を統合しています。

---

## 主な機能と特徴

1. **リアルタイム・テレオペレーション (`teleop_main.py`)**
   - リーダーアームの動作をフォロワー実機に高精度同期追従。
   - 動作データの CSV 記録および MuJoCo 3D 画面での同時プレビューに対応。
   - **起動時アプローチ補間:** 起動直後の姿勢差による急発進・過負荷を防止するコサイン S 字補間を搭載。

2. **安全なモーション再生・生成 (`replay_main.py` / `motion_generator.py`)**
   - 記録した CSV モーションファイルの連続再生および動的なモーション軌道生成。
   - コサイン補間（S字カーブ加減速）による急加速排除と、実機現在角度の自動取得による位置ズレ解消。
   - `現在位置 ➔ Home位置 ➔ 開始姿勢` の安全シーケンスおよび巻き戻し機能。

3. **自然言語タスクプランナー (`tachikoma_agent_1.py` / Step 1 実装完了)**
   - 「Aの積み木をBに運んでからCに持ってって！」といった自然言語の指示を LLM (Gemini) が解釈。
   - `motions/motions.json` に定義されたメタデータおよびワークスペース定義（`workspace_config.py`）をもとに実行プラン（JSON）を自動生成。
   - **デジタルツイン・プレビュー:** 実機を動かす前に MuJoCo 上で動作シーケンスを先行シミュレーションし、承認（Y/N）後に実機へ安全送信。

4. **ドライバ層スルーレートリミッター (`sts3215.py`)**
   - 1ステップあたりの最大変化量を制限し、不意な目標値ジャンプによる物理破損や暴走を根本から防止。

---

## ハードウェア & ソフトウェア環境

* **Robot Arm:** SO-ARM100 (Feetech STS3215 シリアルバスサーボ × 6軸)
* **通信仕様:** USB-Serial 双方向通信 (1,000,000 bps)
* **主要スタック:**
  - Python 3.10+
  - **制御・通信:** `pyserial`
  - **物理シミュレーション:** `mujoco`
  - **LLM プランニング:** `google-genai` (Gemini API)
  - **環境変数管理:** `python-dotenv`

---

## ディレクトリ構成

```text
tachikoma/
├── archive/                 # 過去のコード、バックアップデータ等を保管
├── assets/                  # アームの 3D モデル (URDF / MuJoCo XML) 等を保管
│   ├── assets/              # メッシュ・テクスチャファイル群
│   ├── so100_scene.xml      # MuJoCo シーン定義 XML
│   └── so100.urdf           # ロボットモデル URDF
├── config/
│   ├── joint_config.py      # ポート設定、ボーレート、各軸可動範囲・初期位置定義
│   └── workspace_config.py  # ワークスペース座標・作業エリア定義
├── core/
│   ├── kinematics.py        # 運動学計算・キャリブレーション
│   ├── llm_planner.py       # Gemini API を用いた自然言語 ➔ JSON タスクプランナー
│   ├── motion_generator.py  # 補間・動作シーケンス自動生成
│   ├── sim_viewer.py        # MuJoCo 3D シミュレータ描画ビューア
│   └── sts3215.py           # STS3215 サーボドライバ (スルーレートリミッター内蔵)
├── motions/
│   ├── motions.json         # モーションのメタデータ定義（LLM 参照用辞書）
│   └── *.csv                # 記録・生成された各動作データファイル
├── src/
│   ├── replay_main.py       # モーション自動再生スクリプト
│   ├── tachikoma_agent_1.py # 自然言語対話・プレビュー・実機承認実行エージェント
│   └── teleop_main.py       # リアルタイム遠隔操作・記録統合スクリプト
├── tests/                   # 単体テスト・動作確認用スクリプト群
├── .env                     # 環境変数設定（API キー等 / .gitignore 対象）
├── .gitignore
├── README.md
└── requirements.txt         # 依存ライブラリ一覧