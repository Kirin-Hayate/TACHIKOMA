# Project TACHIKOMA: Vision-Guided Autonomous & LLM-Enabled Robot Arm System

SO-ARM100 をベースに、攻殻機動隊に登場する「タチコマ」のような、**高度な自律動作・デジタルツイン・自然言語対話・遠隔システム防衛で人間の活動を支援する知能型ロボットアームシステム**を開発するプロジェクトです。

単眼カメラと ArUco マーカーによる机上ビジョン認識、SO-ARM100 特有の非対称爪（固定爪・可動爪）を考慮した把持幾何モデル、2次元連成物理たわみ補正、STS3215 バスサーボによる把持フィードバック、MuJoCo による 3D デジタルツイン、マルチモーダル VLM による物体属性同定、LLM マルチステップタスク計画、完全自律連続 Pick & Place パイプライン、そして Discord 経由の遠隔操作・カメラ監視・AI セキュリティ診断・自律システム防衛機構を統合しています。

---

## 主な機能と特徴

### 1. ビジョン・マルチモーダル・LLM 統合対話エージェント (`src/tachikoma_agent_3.py`)
- **VLM セマンティック物体同定:**
  - 机上の物体を OpenCV で輪郭検出し、個別クロップ画像を VLM（Gemini API 主軸 / ローカル Qwen フォールバック）へ送信。
  - カテゴリ、色、外観特徴、自然言語名称（例: `natural wood wooden block`）を一括同定し、`config/current_world_state.json` へ動的構造化。
- **マルチステップ・タスク分解 ＆ 高速推論 (`core/llm_planner.py`):**
  - 自然言語の複合指示（例:「2つを右ゾーンに、残りを左に移動して」「すべて積み重ねて」）を複数の Pick & Place サブタスクへ自動分解。
  - スタッキング動作時は目標配置 $Z$ 座標に厚みを自動累積。
  - `thinking_budget=0` 設定による長考バイパスにより、数秒以内の高速ターンアラウンドを実現。
- **動的 3D デジタルツイン描画 (`user_scn` 直接注入):**
  - 静的 XML スロット定義に依存せず、MuJoCo の動的描画バッファ（`user_scn`）へ検出物体の 3D 直方体（実測寸法・実測ヨー角反映）と頭上黄色テキストラベル（`[#ID] 名称`）、四隅の ArUco マーカー板を直接レンダリング。
- **机上自動再認識ループ (動的リフレッシュ):**
  - 各サブタスクの実行完了ごとに机上を自動再撮影・再同定。物体数の増減や位置ズレによる ID ドリフトを防止し、最新の物理配置を常にワールドステートへ反映。
- **クロスプラットフォーム ＆ 自動ヘッドレス適応:**
  - 実行環境（Windows ノート PC / Linux ミニ PC・SSH 接続）およびディスプレイ環境を自動検知。
  - ディスプレイのない環境や `--headless` 指定時は、3D レンダリングを安全にバイパスし、逆運動学（IK）計算と実機制御のみを高スループット・低リソースで実行。

### 2. Discord 遠隔対話・カメラ配信・自律システム防衛エージェント (`src/tachikoma_discord_bot.py`)
- **自然言語マニピュレーション ＆ 物理コンテキスト動的注入 (Grounding):**
  - 指定された個人用 Discord サーバー（Guild）内からのみ安全に応答。
  - メッセージ受信ごとに「アーム接続状態」「カメラ認識・権限状態」「外付けストレージ（`lsblk`）」「USB機器（`lsusb`）」「机上物体の 3D 座標リスト」をリアルタイム走査してプロンプトへ注入し、ハルシネーションを防止。
  - 立案されたタスクを $[r, \theta, z, \text{Roll}]$ の 4 変数形式で Embed 表示し、実行承認ボタン（`TaskConfirmView`）と `asyncio.to_thread` による非同期サーボ駆動を実現。
- **排他制御カメラ撮影 ＆ Tailscale Serve 連携 MJPEG ライブ配信:**
  - **📸 スナップショット撮影:** ボタンワンタップで V4L2 カメラから 1 フレームを取得して即座にデバイスを解放（`cap.release()`）し、Discord チャットへ送信。
  - **📹 軽量 MJPEG ストリーミング:** バックグラウンド HTTP サーバー（`127.0.0.1:8088`）と `tailscale serve`（HTTPS 8443）を組み合わせ、Tailnet 内のスマホブラウザから低遅延でリアルタイム映像を確認可能。カメラ異常時は 1 回で即座にループを中断するフェイルファスト設計と `CAMERA OFFLINE` プレースホルダー画像生成を実装。
- **7系統テレメトリ収集 ＆ Gemini 高速カスケード AI セキュリティ診断:**
  - Cockpit 相当の 7 系統データ（①外部 ESTABLISHED 接続、②SSH/Cockpit 認証警告ログ、③CPU/メモリ消費上位プロセス、④失敗 systemd サービス、⑤直近の `sudo` 履歴、⑥OS 重大エラーログ、⑦CPU 温度・SSD 残量）を一括収集。
  - SDK 内部の自動リトライ（`tenacity`）を無効化（`attempts=1`）し、`thinking_budget=0`（Thinking OFF）を強制した上で複数モデル候補（`GEMINI_MODEL_CASCADE`）を即座にフォールバック試行。503 サーバー混雑時でも待ち時間ゼロで次モデルへ切り替え、数秒で診断レポートを生成。
  - 診断実行時は開始メッセージの即時確定返信と Discord の「入力中...（Typing）」表示により、スマホ画面での体感応答性を向上。
- **段階的防衛 (Tiered Defense) ＆ スマホ完結 3段グリッド緊急制御パネル (`SystemControlView`):**
  - **レベル 1（警告通知）:** 2 分周期でディスク使用率、CPU 負荷、SSH 不正ログイン試行を監視し、閾値超過時に警告 Embed を送信。
  - **レベル 2（自動迎撃・フェイルセーフ）:** `/tmp`, `/var/tmp`, `/dev/shm` 配下の身元不明プロセスを検知した瞬間、自動で該当プロセスを `kill` し、保護対象の Web サービスを自動停止して踏み台化を阻止。
  - **ワンタップ緊急制御 UI:** `[📸 撮影] [📹 ライブ] [🧠 AI診断]` / `[🛑 アーム] [🛑 Web] [▶️ Web]` / `[🔌 外部隔離 (Tailscale 切断)]` のコンパクト 3 段ボタンを永続化（Persistent View）して配置。

### 3. 完全自律連続 Pick & Place システム (`tools/run_auto_pick_and_place.py`, `src/tachikoma_agent_3.py`)
- **自律タスクスケジューリング:**
  - 机上に散在する複数の物体を検出し、ロボットベースからの距離や到達性に基づいて最適な把持順序を自動決定。
- **6段階インテリジェント・リトライ（戦略的摂動アプローチ）:**
  - 空振りを検知した際、毎回直立 Home 姿勢へ戻る時間ロスを排除。
  - 物体直上の上空退避点（+50mm）に留まったまま爪を開き、深掘り・長手シフト・ランダムジッターを動的に切り替えて 0.5〜0.8 秒間隔で直接再試行（最大 6 回）。
  - 連続失敗した物体は自動で一時スキップされ、他の物体を優先処理。
- **ハードウェア把持フィードバック:**
  - グリッパサーボ（ID 6）の物理現在値を直接ポーリングし、爪完全閉止値（実測: 1889 count）との差分から「空振り（落下）」を確実に検知。
- **デジタルツイン安定化（Home スナップショット方式）:**
  - アーム稼働中に自身がカメラに映り込んで巨大な誤検出物体が発生する現象を防止。
  - アームが直立・待機中（IDLE）のみワークスペースの配置を更新し、動作中はスナップショットを固定表示。把持成功時には該当物体が机上から即座に消滅する物理表現を実現。

### 4. 把持・配置幾何学 ＆ 逆運動学 (IK) ソルバー (`core/kinematics.py`)
- **固定爪着地モデル ＆ 偏心補正:**
  - SO-ARM100 の非対称爪構造に対応し、物体外縁から固定爪を逃がして着地させた上で可動爪で抱え込む幾何計算。
  - 長手・短手の任意オフセット調整（`MANUAL_OFFSET_MAJOR_MM` / `MINOR_MM`）に対応。
- **2次元完全連成・自重たわみ補正モデル:**
  - アームの水平リーチ $r$ と旋回角 $\theta$ から、実機アームの自重沈み込み量 $\Delta z$ を2次多項式で推計して目標 Z 座標を動的補正。
- **適応進入ピッチ角探索:**
  - ワークスペースの近傍から物理限界域（最大リーチ 410mm）まで、真上からの降下（80°）〜斜め進入（5°）を自動探索。
- **2段階安全 Home 復帰シーケンス:**
  - 把持・配置完了後、手首ピッチ（ID 4）を先行して引き上げて重力モーメントを低減し、サーボの物理到達を監視しながら確実に直立姿勢へ復帰。

### 5. デュアル実行 ＆ 実機把持検証ツール (`tools/run_real_pick_test.py`)
- **シミュレータ先行プレビュー:**
  - カメラ認識結果に基づき、把持から配置・復帰までの一連の軌道を MuJoCo 上で事前プレビュー。
  - コンソールで承認（`y`）を入力した時のみ実機サーボを同期駆動する安全ゲート機構。

### 6. 対話型テレオペ ＆ リアルタイム座標モニタ (`tools/interactive_arm_teleop.py`)
- **キーボード手動ティーチング:**
  - `W/S/A/D/R/F`（直交 X/Y/Z）、`U/J`（ピッチ）、`O/K`（ロール）、`C/V`（爪開閉）で実機とシミュレータを同時操作。
  - 現在の「直交座標 $(X, Y, Z)$」「極座標 $(r, \theta, z)$」「たわみ補正量」「サーボ Raw 生値」をコンソールへリアルタイム表示。

### 7. リアルタイム追従＆モーション再生システム
- **テレオペレーション (`src/teleop_main.py`):** リーダーアームの動作をフォロワー実機に高精度同期追従。CSV 記録対応。
- **モーション再生 (`src/replay_main.py`):** 記録された軌道をコサイン S 字加減速で滑らかに再生。
- **ドライバ層スルーレートリミッター (`core/sts3215.py`):** 1ステップあたりの最大変化量を制限し、不意な目標値ジャンプによる物理破損を根本防止。

---

## ハードウェア & ソフトウェア環境

* **Robot Arm:** SO-ARM100 (Feetech STS3215 シリアルバスサーボ × 6軸)
* **Vision System:** トップダウン RGB カメラ (1280×720 / 640×480, DirectShow / V4L2) + ArUco マーカー (DICT_4X4_50, Homography 投影)
* **Edge Host / Network:** CHUWI HeroBox (Intel Celeron J4125 / Xubuntu 24.04 LTS) + Tailscale Mesh VPN & Tailscale Serve
* **通信仕様:** USB-Serial 双方向通信 (1,000,000 bps)
* **主要スタック:**
  - Python 3.10+
  - **コンピュータビジョン:** `opencv-python`, `opencv-contrib-python`
  - **物理シミュレーション:** `mujoco`
  - **サーボ通信・制御:** `pyserial`
  - **数値計算:** `numpy`
  - **遠隔制御・システム監視:** `discord.py`, `psutil`, `python-dotenv`
  - **LLM / VLM:** `google-genai`, `ollama` (Qwen2.5-VL)

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
├── src/                     # 対話エージェント・Discord Bot・テレオペ・モーション再生実行系
│   ├── replay_main.py       # モーション自動再生スクリプト
│   ├── tachikoma_agent_1.py # 自然言語対話・プレビュー・実機承認実行エージェント
│   ├── tachikoma_agent_3.py # VLM一括同定・LLMマルチステップ・自律再認識対話エージェント
│   ├── tachikoma_discord_bot.py # Discord 遠隔対話・カメラ配信・AI診断・段階的防衛エージェント
│   └── teleop_main.py       # リアルタイム遠隔操作・記録統合スクリプト
├── tools/                   # 実機テスト・自律実行・調整ツール群
│   ├── interactive_arm_teleop.py   # キーボード対話テレオペ & リアルタイム座標モニタ
│   ├── run_auto_pick_and_place.py  # 完全自律連続 Pick & Place 自動化システム
│   └── run_real_pick_test.py       # 単体物体 Pick & Place プレビュー＆検証ツール
├── .env                     # 環境変数設定 (APIキー、Discord設定、監視閾値等)
├── .gitignore
├── README.md
└── requirements.txt         # 依存ライブラリ一覧
```

---

## セットアップ手順

### 1. 依存ライブラリのインストール
Python 3.10 以上の仮想環境を作成し、必要なパッケージを導入します。

```bash
git clone [https://github.com/your-username/tachikoma.git](https://github.com/your-username/tachikoma.git)
cd tachikoma

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install discord.py psutil python-dotenv opencv-python google-genai
```

### 2. 環境変数設定 (`.env`)
プロジェクトルートに `.env` ファイルを作成し、API キーや Discord サーバー設定、カメラ配信・監視パラメータを記述します。

```ini
# --- Gemini API 設定 ---
GEMINI_API_KEY="your-gemini-api-key-here"
GEMINI_MODEL_CASCADE="gemini-2.5-flash,gemini-2.5-flash-lite,gemini-1.5-flash"

# --- Discord Bot ＆ 自律防衛設定 ---
DISCORD_BOT_TOKEN="your-discord-bot-token-here"
DISCORD_GUILD_ID="123456789012345678"
DISCORD_ALERT_CHANNEL_ID="123456789012345678"
PROTECTED_SERVICE_NAME="my_web_app"
DISK_USAGE_THRESHOLD="90.0"
CPU_LOAD_THRESHOLD="3.8"
ALERT_COOLDOWN_SEC="900"

# --- カメラ・MJPEG ライブ配信設定 ---
CAMERA_DEVICE_INDEX="0"
CAMERA_DEVICE_PATH="/dev/video0"
STREAM_PORT="8088"
STREAM_FPS="3.0"
STREAM_EXTERNAL_URL="[https://your-machine.tailxxxx.ts.net:8443](https://your-machine.tailxxxx.ts.net:8443)"
```

### 3. Linux 実行環境（ミニ PC / Ubuntu）の固有設定
Linux（Xubuntu 等）を実行ホストとする場合は、シリアルポートおよびカメラへのアクセス権限を付与します。

```bash
sudo usermod -a -G dialout,video $USER
# 現在のシェルセッションへ即座にカメラ読み書き権限を付与する場合
sudo setfacl -m u:$USER:rw /dev/video*
```

USB シリアルデバイスの認識名を固定するため、`/etc/udev/rules.d/99-tachikoma.rules` を作成してルールを定義します。

```text
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="tachikoma_arm", MODE="0666"
```

設定反映後、udev ルールを再読込します。

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Tailscale Serve を用いて、ローカルの MJPEG ストリーム（ポート `8088`）を Tailnet 内の HTTPS（ポート `8443`）として公開する場合は以下を実行します。

```bash
sudo tailscale serve --bg --https=8443 [http://127.0.0.1:8088](http://127.0.0.1:8088)
```

---

## 実行方法

### 1. ビジョン・LLM 統合対話エージェント (`src/tachikoma_agent_3.py`)

#### ① シミュレーションプレビューのみ（実機なしテスト）
机上の物体認識、VLM 同定、LLM タスク立案、MuJoCo 3D プレビューまでを確認できます。
```bash
python src/tachikoma_agent_3.py
```

#### ② 実機フォロワー接続モード
プレビュー確認後、コンソールで `y` を入力すると実機アームが搬送を実行します。
```bash
python src/tachikoma_agent_3.py --arm
```

#### ③ ヘッドレスモード（ミニ PC / 組み込み実行）
MuJoCo 3D ウィンドウを起動せず、純粋な数値 IK 計算と実機アーム駆動のみを高効率に実行します。
```bash
python src/tachikoma_agent_3.py --arm --headless
```

#### CLI オプション一覧 (`tachikoma_agent_3.py`)
| オプション | 型 | 説明 |
| :--- | :--- | :--- |
| `--arm` | flag | 実機フォロワーアームへのシリアル接続を有効化 |
| `--headless` | flag | MuJoCo 3D ビューアを起動せずヘッドレス実行（IK 計算のみ） |
| `--no-headless` | flag | ヘッドレス環境下でも強制的に 3D ビューアウィンドウの起動を試行 |
| `--port` | string | シリアルポートを手動上書き（例: `/dev/ttyUSB0`, `COM4`） |
| `--cam-index` | int | 使用するカメラのデバイスインデックス（既定: `0`） |

### 2. Discord 遠隔制御・システム防衛エージェント (`src/tachikoma_discord_bot.py`)
Discord からの自然言語指示、カメラ撮影・ライブ配信、AI セキュリティ診断、および 2 分周期の段階的防衛モニタリングを起動します。

```bash
# 監視・シミュレーションモードで起動（MJPEG配信サーバー同時起動）
python src/tachikoma_discord_bot.py

# 実機フォロワーアーム接続を有効化して起動
python src/tachikoma_discord_bot.py --arm

# カメラ配信サーバーを無効化して起動
python src/tachikoma_discord_bot.py --no-stream
```
- Discord チャット欄で `!control` または `!status` と送信すると、システムメトリクスと接続ハードウェア一覧、および 3 段グリッドの緊急制御パネルが表示されます。

### 3. 完全自律連続 Pick & Place 自動化 (`tools/run_auto_pick_and_place.py`)
机上にあるすべての物体を自動で検出し、把持順序の最適化、空振り時の 6 段階リトライを行いながら一掃する自律ループです。
```bash
python tools/run_auto_pick_and_place.py
```

### 4. キーボード対話テレオペレーション (`tools/interactive_arm_teleop.py`)
手動でアームの IK 軌道とサーボ位置を検証します。
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
  - **ミニ PC (Xubuntu / CHUWI HeroBox 等):** ロボットアーム常設ホスト、ヘッドレス実行、Discord 常駐エージェントおよび Tailscale Exit Node / Serve によるセキュアなリモートゲートウェイ。
- **Git ブランチ戦略:**
  - `dev` ブランチ: 実機検証前の新規機能や実験的コードを同期・検証。
  - `main` ブランチ: 実機検証完了済みの安定版コードのみを統合・常駐同期。
- **計算リソース配分:**
  - 高負荷な 3D レンダリングはヘッドレス化によりエッジ側でカット。
  - 複雑なセマンティック理解、言語推論、システムログ解析（VLM / LLM）はクラウド API（高速カスケード＋Thinking 無効化）に集約し、エッジ側はミリ秒単位の IK 計算、シリアルバス制御、および軽量なプロセス・ハードウェア監視に専念。