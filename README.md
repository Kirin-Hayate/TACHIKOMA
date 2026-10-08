# Project TACHIKOMA: Vision-Guided Autonomous & LLM-Enabled Robot Arm System

SO-ARM100 をベースに、攻殻機動隊に登場する「タチコマ」のような、**高度な自律動作・デジタルツイン・自然言語対話で人間の活動を支援する知能型ロボットアームシステム**を開発するプロジェクトです 。

単眼カメラと ArUco マーカーによる机上ビジョン認識、SO-ARM100 特有の非対称爪（固定爪・可動爪）を考慮した把持幾何モデル、2次元連成物理たわみ補正、STS3215 バスサーボによる把持フィードバック、MuJoCo による 3D デジタルツイン、マルチモーダル VLM による物体属性同定、LLM マルチステップタスク計画、そして完全自律連続 Pick & Place パイプラインを統合しています 。

---

## 主な機能と特徴

### 1. ビジョン・マルチモーダル・LLM 統合対話エージェント (`src/tachikoma_agent_3.py`)
- **VLM セマンティック物体同定:**
  - 机上の物体を OpenCV で輪郭検出し、個別クロップ画像を VLM（Gemini API 主軸 / ローカル Qwen フォールバック）へ送信 。
  - カテゴリ、色、外観特徴、自然言語名称（例: `natural wood wooden block`）を一括同定し、`config/current_world_state.json` へ動的構造化 。
- **マルチステップ・タスク分解 ＆ 高速推論 (`core/llm_planner.py`):**
  - 自然言語の複合指示（例:「2つを右ゾーンに、残りを左に移動して」「すべて積み重ねて」）を複数の Pick & Place サブタスクへ自動分解 。
  - スタッキング動作時は目標配置 $Z$ 座標に厚みを自動累積 。
  - `thinking_budget=0` 設定による長考バイパスにより、数秒以内の高速ターンアラウンドを実現。
- **動的 3D デジタルツイン描画 (`user_scn` 直接注入):**
  - 静的 XML スロット定義に依存せず、MuJoCo の動的描画バッファ（`user_scn`）へ検出物体の 3D 直方体（実測寸法・実測ヨー角反映）と頭上黄色テキストラベル（`[#ID] 名称`）、四隅の ArUco マーカー板を直接レンダリング 。
- **机上自動再認識ループ (動的リフレッシュ):**
  - 各サブタスクの実行完了ごとに机上を自動再撮影・再同定 。物体数の増減や位置ズレによる ID ドリフトを防止し、最新の物理配置を常にワールドステートへ反映 。
- **クロスプラットフォーム ＆ 自動ヘッドレス適応:**
  - 実行環境（Windows ノート PC / Linux ミニ PC・SSH 接続）およびディスプレイ環境を自動検知 。
  - ディスプレイのない環境や `--headless` 指定時は、3D レンダリングを安全にバイパスし、逆運動学（IK）計算と実機制御のみを高スループット・低リソースで実行 。

### 2. 完全自律連続 Pick & Place システム (`tools/run_auto_pick_and_place.py`, `src/tachikoma_agent_3.py`)
- **自律タスクスケジューリング:**
  - 机上に散在する複数の物体を検出し、ロボットベースからの距離や到達性に基づいて最適な把持順序を自動決定 。
- **6段階インテリジェント・リトライ（戦略的摂動アプローチ）:**
  - 空振りを検知した際、毎回直立 Home 姿勢へ戻る時間ロスを排除 。
  - 物体直上の上空退避点（+50mm）に留まったまま爪を開き、深掘り・長手シフト・ランダムジッターを動的に切り替えて 0.5〜0.8 秒間隔で直接再試行（最大 6 回）。
  - 連続失敗した物体は自動で一時スキップされ、他の物体を優先処理 。
- **ハードウェア把持フィードバック:**
  - グリッパサーボ（ID 6）の物理現在値を直接ポーリングし、爪完全閉止値（実測: 1889 count）との差分から「空振り（落下）」を確実に検知。
- **デジタルツイン安定化（Home スナップショット方式）:**
  - アーム稼働中に自身がカメラに映り込んで巨大な誤検出物体が発生する現象を防止 。
  - アームが直立・待機中（IDLE）のみワークスペースの配置を更新し、動作中はスナップショットを固定表示 。把持成功時には該当物体が机上から即座に消滅する物理表現を実現 。

### 3. 把持・配置幾何学 ＆ 逆運動学 (IK) ソルバー (`core/kinematics.py`)
- **固定爪着地モデル ＆ 偏心補正:**
  - SO-ARM100 の非対称爪構造に対応し、物体外縁から固定爪を逃がして着地させた上で可動爪で抱え込む幾何計算 。
  - 長手・短手の任意オフセット調整（`MANUAL_OFFSET_MAJOR_MM` / `MINOR_MM`）に対応。
- **2次元完全連成・自重たわみ補正モデル:**
  - アームの水平リーチ $r$ と旋回角 $\theta$ から、実機アームの自重沈み込み量 $\Delta z$ を2次多項式で推計して目標 Z 座標を動的補正 。
- **適応進入ピッチ角探索:**
  - ワークスペースの近傍から物理限界域（最大リーチ 410mm）まで、真上からの降下（80°）〜斜め進入（5°）を自動探索 。
- **2段階安全 Home 復帰シーケンス:**
  - 把持・配置完了後、手首ピッチ（ID 4）を先行して引き上げて重力モーメントを低減し、サーボの物理到達を監視しながら確実に直立姿勢へ復帰 。

### 4. デュアル実行 ＆ 実機把持検証ツール (`tools/run_real_pick_test.py`)
- **シミュレータ先行プレビュー:**
  - カメラ認識結果に基づき、把持から配置・復帰までの一連の軌道を MuJoCo 上で事前プレビュー。
  - コンソールで承認（`y`）を入力した時のみ実機サーボを同期駆動する安全ゲート機構。

### 5. 対話型テレオペ ＆ リアルタイム座標モニタ (`tools/interactive_arm_teleop.py`)
- **キーボード手動ティーチング:**
  - `W/S/A/D/R/F`（直交 X/Y/Z）、`U/J`（ピッチ）、`O/K`（ロール）、`C/V`（爪開閉）で実機とシミュレータを同時操作 。
  - 現在の「直交座標 $(X, Y, Z)$」「極座標 $(r, \theta, z)$」「たわみ補正量」「サーボ Raw 生値」をコンソールへリアルタイム表示 。

### 6. リアルタイム追従＆モーション再生システム
- **テレオペレーション (`src/teleop_main.py`):** リーダーアームの動作をフォロワー実機に高精度同期追従 。CSV 記録対応 。
- **モーション再生 (`src/replay_main.py`):** 記録された軌道をコサイン S 字加減速で滑らかに再生 。
- **ドライバ層スルーレートリミッター (`core/sts3215.py`):** 1ステップあたりの最大変化量を制限し、不意な目標値ジャンプによる物理破損を根本防止 。

---

## ハードウェア & ソフトウェア環境

* **Robot Arm:** SO-ARM100 (Feetech STS3215 シリアルバスサーボ × 6軸) 
* **Vision System:** トップダウン RGB カメラ (1280×720, DirectShow / V4L2) + ArUco マーカー (DICT_4X4_50, Homography 投影)
* **通信仕様:** USB-Serial 双方向通信 (1,000,000 bps)
* **主要スタック:**
  - Python 3.10+ 
  - **コンピュータビジョン:** `opencv-python`, `opencv-contrib-python` 
  - **物理シミュレーション:** `mujoco` 
  - **サーボ通信・制御:** `pyserial` 
  - **数値計算:** `numpy` 
  - **LLM / VLM:** `google-genai` (Gemini 2.5 Flash), `ollama` (Qwen2.5-VL)

---

## ディレクトリ構成

```text
tachikoma/
├── assets/                  # アームの 3D モデル (URDF / MuJoCo XML) 
│   ├── assets/              # STL メッシュ・テクスチャファイル群 
│   ├── so100_scene.xml      # MuJoCo シーン定義 XML 
│   └── so100.urdf           # ロボットモデル URDF 
├── config/
│   ├── current_world_state.json # 動的生成される現在の机上物体配置 JSON 
│   ├── joint_config.py      # サーボ ID、通信レート、可動リミット、OS別ポート自動判定
│   └── workspace_config.py  # ワークスペース座標・エリア定義 
├── core/
│   ├── bus_servo_controller.py # 実機サーボバス統合制御インターフェース 
│   ├── kinematics.py        # 順・逆運動学、たわみ補正、非対称把持・配置幾何計算 
│   ├── llm_planner.py       # Gemini API 自然言語マルチステップタスクプランナー
│   ├── motion_generator.py  # 補間・動作シーケンス自動生成 
│   ├── multimodal_tagger.py # VLM (Gemini / Qwen) による物体属性一括同定 
│   ├── sim_viewer.py        # MuJoCo 3D シミュレータ描画ビューア 
│   ├── sts3215.py           # STS3215 サーボドライバ (スルーレート制限・バッファクリア内蔵) 
│   ├── tabletop_detector.py # 背景差分・輪郭抽出による机上物体幾何計測 
│   ├── trajectory_executor.py # S字加減速軌道実行 ＆ 到達監視付き2段階安全Home復帰 
│   └── vision_projector.py  # ArUco ホモグラフィ変換・机面平面射影 
├── motions/                 # 記録された CSV モーションデータ 
├── src/                     # 対話エージェント・テレオペ・モーション再生実行系 
│   ├── replay_main.py       # モーション自動再生スクリプト 
│   ├── tachikoma_agent_1.py # 自然言語対話・プレビュー・実機承認実行エージェント 
│   ├── tachikoma_agent_3.py # VLM一括同定・LLMマルチステップ・自律再認識対話エージェント 
│   └── teleop_main.py       # リアルタイム遠隔操作・記録統合スクリプト 
├── tools/                   # 実機テスト・自律実行・調整ツール群 
│   ├── interactive_arm_teleop.py   # キーボード対話テレオペ & リアルタイム座標モニタ 
│   ├── run_auto_pick_and_place.py  # 完全自律連続 Pick & Place 自動化システム 
│   └── run_real_pick_test.py       # 単体物体 Pick & Place プレビュー＆検証ツール 
├── .env                     # 環境変数設定 (GEMINI_API_KEY 等) 
├── .gitignore
├── README.md
└── requirements.txt         # 依存ライブラリ一覧 
```

---

## セットアップ手順

### 1. 依存ライブラリのインストール
Python 3.10 以上の仮想環境を作成し、必要なパッケージを導入します 。

```bash
git clone [https://github.com/your-username/tachikoma.git](https://github.com/your-username/tachikoma.git)
cd tachikoma

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 環境変数設定
プロジェクトルートに `.env` ファイルを作成し、API キーを設定します 。

```bash
GEMINI_API_KEY="your-gemini-api-key-here"
```

### 3. Linux 実行環境（ミニ PC / Ubuntu）の固有設定
Linux（Xubuntu 等）を実行ホストとする場合は、シリアルポートおよびカメラへのアクセス権限を付与します。

```bash
sudo usermod -a -G dialout,video $USER
```

USB シリアルデバイスの認識名を固定するため、`/etc/udev/rules.d/99-tachikoma.rules` を作成してルールを定義します。

```text
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="tachikoma_arm", MODE="0666"
```

設定反映後、udev ルールを再読込します。
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 実行方法

### 1. ビジョン・LLM 統合対話エージェント (`src/tachikoma_agent_3.py`)

#### ① シミュレーションプレビューのみ（実機なしテスト）
机上の物体認識、VLM 同定、LLM タスク立案、MuJoCo 3D プレビューまでを確認できます 。
```bash
python src/tachikoma_agent_3.py
```

#### ② 実機フォロワー接続モード
プレビュー確認後、コンソールで `y` を入力すると実機アームが搬送を実行します 。
```bash
python src/tachikoma_agent_3.py --arm
```

#### ③ ヘッドレスモード（ミニ PC / 組み込み実行）
MuJoCo 3D ウィンドウを起動せず、純粋な数値 IK 計算と実機アーム駆動のみを高効率に実行します 。
```bash
python src/tachikoma_agent_3.py --arm --headless
```

#### CLI オプション一覧
| オプション | 型 | 説明 |
| :--- | :--- | :--- |
| `--arm` | flag | 実機フォロワーアームへのシリアル接続を有効化  |
| `--headless` | flag | MuJoCo 3D ビューアを起動せずヘッドレス実行（IK 計算のみ）  |
| `--no-headless` | flag | ヘッドレス環境下でも強制的に 3D ビューアウィンドウの起動を試行  |
| `--port` | string | シリアルポートを手動上書き（例: `/dev/ttyUSB0`, `COM4`）  |
| `--cam-index` | int | 使用するカメラのデバイスインデックス（既定: `0`）  |

### 2. 完全自律連続 Pick & Place 自動化 (`tools/run_auto_pick_and_place.py`)
机上にあるすべての物体を自動で検出し、把持順序の最適化、空振り時の 6 段階リトライを行いながら一掃する自律ループです 。
```bash
python tools/run_auto_pick_and_place.py
```

### 3. キーボード対話テレオペレーション (`tools/interactive_arm_teleop.py`)
手動でアームの IK 軌道とサーボ位置を検証します 。
```bash
python tools/interactive_arm_teleop.py
```
- `W / S`: $X$ 軸（前後移動）
- `A / D`: $Y$ 軸（左右移動）
- `R / F`: $Z$ 軸（上下移動）
- `U / J`: 手首ピッチ角変更
- `O / K`: 手首ロール角変更
- `C / V`: グリッパー開閉

---

## 開発・デプロイ運用方針

- **デュアル環境ハイブリッド運用:**
  - **ノート PC (Windows):** 対話型開発、3D シミュレータプレビュー、機能追加、アルゴリズムチューニング。
  - **ミニ PC (Xubuntu / CHUWI HeroBox 等):** ロボットアーム常設ホスト、ヘッドレス実行専用アプライアンス。
- **Git ブランチ戦略:**
  - `dev` ブランチ: 実機検証前の新規機能や実験的コードを同期・検証。
  - `main` ブランチ: 実機検証完了済みの安定版コードのみを統合・常駐同期。
- **計算リソース配分:**
  - 高負荷な 3D レンダリングはヘッドレス化によりエッジ側でカット 。
  - 複雑なセマンティック理解や言語推論（VLM / LLM）はクラウド API に集約し、エッジ側はミリ秒単位の IK 計算およびシリアルバス制御に専念 。