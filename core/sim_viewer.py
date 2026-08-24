"""
==============================================================================
MuJoCo 3Dシミュレーションビューア管理モジュール (core/sim_viewer.py)
==============================================================================
【役割】
1. SO-ARM100 の 3D モデル (assets/so100_scene.xml) を読み込んで画面に表示する
2. 与えられた関節目標値 (Raw: 0〜4095) をラジアンに変換して 3D モデルの姿勢をリアルタイム更新する
3. キーボード操作（Space: 一時停止/再開, R: 最初から, L: ループ再生）の状態を管理する
==============================================================================
"""

import os
import mujoco
import mujoco.viewer
from core.kinematics import raw_to_radian
from config.joint_config import SERVO_IDS


class MujocoSimViewer:
    def __init__(self, xml_path=None):
        """
        MuJoCo モデルをロードして初期化する
        - xml_path: so100_scene.xml のファイルパス（None の場合はデフォルト位置を探索）
        """
        # プロジェクトルートからのデフォルトパスを取得
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "assets", "so100_scene.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo シーンモデルが見つかりません: {xml_path}")

        print(f"🤖 SO-ARM100 シーンモデルをロード: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 画面の視認性を高めるためのライト設定（影や輪郭を見やすくする）
        self.model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
        self.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        self.model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]

        # キーボード操作用の状態フラグ
        self.paused = False
        self.reset_requested = False
        self.loop_mode = False

        self.viewer = None

    def _key_callback(self, keycode):
        """
        キーボードが押されたときに自動で呼ばれる関数
        - Space (32)   : 一時停止 / 再開
        - R (82, 114)  : 最初からリプレイ
        - L (76, 108)  : ループ再生 ON / OFF
        """
        if keycode == 32:  # Spaceキー
            self.paused = not self.paused
            status = "⏸️ 一時停止" if self.paused else "▶️ 再生再開"
            print(f"\n[{status}]")
        elif keycode in (82, 114):  # 'R', 'r'
            self.reset_requested = True
            print("\n🔄 最初から再生します")
        elif keycode in (76, 108):  # 'L', 'l'
            self.loop_mode = not self.loop_mode
            loop_str = "ON" if self.loop_mode else "OFF"
            print(f"\n🔁 ループ再生: {loop_str}")

    def update_joints(self, target_positions):
        """
        各サーボの目標値 (Raw: 0〜4095) をラジアンに変換して MuJoCo の関節に代入し、描画を更新する
        - target_positions: {1: 2048, 2: 973, ...} のような各軸の目標値辞書
        """
        for i, sid in enumerate(SERVO_IDS):
            if sid in target_positions:
                target_val = target_positions[sid]
                # core/kinematics.py の計算式でラジアン角へ変換
                rad = raw_to_radian(sid, target_val)
                if i < self.model.nq:
                    self.data.qpos[i] = rad

        # 姿勢の順運動学計算を更新し、ビューアへ反映
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def launch(self):
        """
        MuJoCo のパッシブビューアを起動する（with 構文で使用可能）
        使い方:
            sim = MujocoSimViewer()
            with sim.launch():
                while sim.is_running():
                    sim.update_joints(targets)
        """
        print("==================================================")
        print(" 🚀 操作キー一覧:")
        print("   [Space] : 一時停止 / 再生")
        print("   [R]     : 最初からリプレイ")
        print("   [L]     : ループ再生 ON / OFF")
        print("==================================================")
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )
        return self.viewer

    def is_running(self):
        """ビューアのウィンドウが開いているか確認"""
        return self.viewer is not None and self.viewer.is_running()