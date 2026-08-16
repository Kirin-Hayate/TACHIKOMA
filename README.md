## Project TACHICOMA: Voice-Enabled Teleoperated Robot Arm

SO-ARM100をベースに、攻殻機動隊に登場する「タチコマ」のような、**音声操作で人間の活動を支援する自律・協調型ロボットアーム**を目指す開発プロジェクトです。

現在は、マスター・スレーブ方式による**テレオペレーション（遠隔操作）システムの構築・検証**をメインに進めています。

---

## Project Vision
* **音声と知能による支援:** 将来的には音声対話や自然言語処理を統合し、言葉で指示して動かせるロボットアームへ。
* **直感的な遠隔操作 (Teleoperation):** リーダー・フォロワー構成による高精度な双方向同期制御。

---

## Hardware & Setup
* **Robot Arm:** SO-ARM100 (3Dプリンタ製カスタム骨格 & Feetech STS3215 シリアルバスサーボ)
* **Control Unit:** Python 3.x / PySerial
* **Communication:** 双方向シリアル通信 (1Mbps)

---

## Repository Structure

```text
├── src/                  # メインの制御・同期スクリプト
│   └── teleop_sync.py    # 1:1 等倍マッピング＆無限回転対応の同期制御プログラムなど
├── tests/                # 動作テスト・単体検証用スクリプト
├── archive/               # 過去のバージョンや実験用コードのアーカイブ
└── docs/                 # 設計メモ・可動範囲データなど