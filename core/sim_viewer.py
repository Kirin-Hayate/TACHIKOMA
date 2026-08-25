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
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "assets", "so100_scene.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo シーンモデルが見つかりません: {xml_path}")

        print(f"🤖 SO-ARM100 シーンモデルをロード: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
        self.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        self.model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]

        # 状態フラグ
        self.paused = False
        self.reset_requested = False
        self.loop_mode = False
        self.playback_speed = 1.0  # ★ 再生速度倍率 (デフォルト: 1.0倍)

        self.viewer = None

    def _key_callback(self, keycode):
        """
        - Space (32)      : 一時停止 / 再開
        - R (82, 114)     : 最初からリプレイ
        - L (76, 108)     : ループ再生 ON / OFF
        - 1〜9 (49〜57)   : 再生速度倍率の変更 (1倍〜9倍速)
        - テンキー 1〜9 (321〜329) : テンキー倍率変更
        """
        if keycode == 32:  # Space
            self.paused = not self.paused
            status = "⏸️ 一時停止" if self.paused else "▶️ 再生再開"
            print(f"\n[{status}]")
        elif keycode in (82, 114):  # R, r
            self.reset_requested = True
            print("\n🔄 最初から再生します")
        elif keycode in (76, 108):  # L, l
            self.loop_mode = not self.loop_mode
            loop_str = "ON" if self.loop_mode else "OFF"
            print(f"\n🔁 ループ再生: {loop_str}")
        elif 49 <= keycode <= 57:  # メインキー 1〜9
            speed = float(keycode - 48)
            self.playback_speed = speed
            print(f"\n⏩ 再生速度: {speed:.1f}倍速")
        elif 321 <= keycode <= 329:  # テンキー 1〜9
            speed = float(keycode - 320)
            self.playback_speed = speed
            print(f"\n⏩ 再生速度: {speed:.1f}倍速")

    def update_joints(self, target_positions):
        for i, sid in enumerate(SERVO_IDS):
            if sid in target_positions:
                target_val = target_positions[sid]
                rad = raw_to_radian(sid, target_val)
                if i < self.model.nq:
                    self.data.qpos[i] = rad

        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def launch(self):
        print("==================================================")
        print(" 🚀 操作キー一覧:")
        print("   [Space] : 一時停止 / 再生")
        print("   [1]〜[9]: 再生速度変更 (1倍速〜9倍速)")
        print("   [R]     : 最初からリプレイ")
        print("==================================================")
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )
        return self.viewer

    def is_running(self):
        return self.viewer is not None and self.viewer.is_running()