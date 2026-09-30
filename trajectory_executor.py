"""
==============================================================================
ロボットアーム軌道実行＆実機同期エンジン
(core/trajectory_executor.py)
==============================================================================
"""

import time
import math
import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple

from core.kinematics import (
    radian_to_raw,
    raw_to_radian,
    JOINT_CONFIG,
    get_home_radians
)


class TrajectoryExecutor:
    def __init__(self, sim=None, servo_controller=None, default_speed_ms: int = 1200):
        """
        Args:
            sim: MujocoSimViewer インスタンス (シミュレーション可視化用)
            servo_controller: BusServoController インスタンス (実機接続時のみ)
            default_speed_ms: 姿勢間の基準移動時間 (ms)
        """
        self.sim = sim
        self.controller = servo_controller
        self.default_speed_ms = default_speed_ms
        self.is_real_connected = (self.controller is not None and getattr(self.controller, 'is_open', False))

    def move_to_rad(
        self,
        target_rad: Dict[int, float],
        duration_sec: float = 1.2,
        steps: int = 30,
        sync_viewer: bool = True,
        send_to_real: bool = True  # 👉 追加: 実機送信を制御するフラグ
    ) -> bool:
        """
        指定された関節角度 (rad) へ S 字加減速で補間移動。
        send_to_real=True かつ実機接続時のみ実機サーボへ指令を送信。
        """
        # 1. サーボ可動域リミットチェック
        for sid, rad in target_rad.items():
            if sid in JOINT_CONFIG and JOINT_CONFIG[sid]["type"] == "bounded":
                raw_val = radian_to_raw(sid, rad)
                cfg = JOINT_CONFIG[sid]
                if not (cfg["f_min"] <= raw_val <= cfg["f_max"]):
                    print(f"❌ 軌道実行エラー: サーボ ID{sid} の指令値 (Raw={raw_val}) が安全範囲外です。")
                    return False

        # 2. 開始姿勢の取得
        if self.sim is not None:
            start_qpos = np.copy(self.sim.data.qpos[:6])
        else:
            start_qpos = np.array([get_home_radians()[i] for i in range(1, 7)])

        goal_qpos = np.copy(start_qpos)
        for sid, rad in target_rad.items():
            if 1 <= sid <= 6:
                goal_qpos[sid - 1] = rad

        dt = duration_sec / steps

        # 3. 補間ストリーミング実行
        for s in range(1, steps + 1):
            ratio = s / float(steps)
            t = ratio * ratio * ratio * (ratio * (ratio * 6 - 15) + 10)
            current = (1.0 - t) * start_qpos + t * goal_qpos

            # シミュレーション更新
            if self.sim is not None:
                self.sim.data.qpos[:6] = current
                mujoco.mj_forward(self.sim.model, self.sim.data)
                if sync_viewer and self.sim.viewer is not None:
                    self.sim.viewer.sync()

            # 実機サーボ更新 (send_to_real が True のときだけ送信)
            if self.is_real_connected and send_to_real:
                step_raws = {sid: radian_to_raw(sid, float(current[sid - 1])) for sid in range(1, 7)}
                self.controller.move_servos(step_raws)

            time.sleep(dt)

        return True

    def execute_waypoints(
        self,
        waypoint_list: List[Tuple[Dict[int, float], float, str]],
        send_to_real: bool = True  # 👉 追加
    ) -> bool:
        """
        複数のキーポーズを順次実行。
        send_to_real=False の場合はシミュレーション上のみ再生される。
        """
        for i, (wp_rad, dur, desc) in enumerate(waypoint_list, 1):
            print(f"   ▶️ [{i}/{len(waypoint_list)}] {desc} (所要時間: {dur:.1f}s)...")
            success = self.move_to_rad(wp_rad, duration_sec=dur, send_to_real=send_to_real)
            if not success:
                print(f"⚠️ 中断: ステップ '{desc}' の実行に失敗しました。")
                return False
        return True

    def move_to_home_and_wait(self, home_rad: Dict[int, float], timeout: float = 4.0):
        """
        auto_calibrate_workspace_2.py の確証ロジックに基づく 2段階安全 Home 復帰
        1. 手首ピッチ (ID 4) を先行引き上げ
        2. 全軸を Home 姿勢へ移動
        3. サーボ物理到達の監視
        """
        print("🏠 Home 姿勢へ移動中...")

        # 現在の実機姿勢を読み取り (取得できない場合は sim または home_rad をフォールバック)
        current_rad = {}
        if self.is_real_connected and hasattr(self.controller, 'driver') and self.controller.driver:
            for sid in range(1, 7):
                pos = self.controller.driver.read_position(sid)
                if pos is not None:
                    current_rad[sid] = raw_to_radian(sid, pos)
                else:
                    current_rad[sid] = float(self.sim.data.qpos[sid - 1]) if self.sim else home_rad[sid]
        else:
            current_rad = {sid: float(self.sim.data.qpos[sid - 1]) if self.sim else home_rad[sid] for sid in range(1, 7)}

        # ----------------------------------------------------------------------
        # ステップ 1: まず ID 4 (手首ピッチ) を上向きに逃がす (所要 1.0秒)
        # (机面との擦れや無理なトルクを防ぎ、ID4 の戻り遅れ・脱調を解消)
        # ----------------------------------------------------------------------
        stage1_rad = dict(current_rad)
        stage1_rad[4] = home_rad[4]
        self.move_to_rad(stage1_rad, duration_sec=1.0, steps=20, send_to_real=True)

        # ----------------------------------------------------------------------
        # ステップ 2: 全軸を Home 姿勢へ補間移動 (余裕を持たせて 2.0秒)
        # ----------------------------------------------------------------------
        self.move_to_rad(home_rad, duration_sec=2.0, steps=35, send_to_real=True)

        # ----------------------------------------------------------------------
        # ステップ 3: サーボの物理到達を監視 (auto_calibrate_workspace_2.py 準拠)
        # ----------------------------------------------------------------------
        if self.is_real_connected and hasattr(self.controller, 'driver') and self.controller.driver:
            start_t = time.time()
            while time.time() - start_t < timeout:
                all_reached = True
                for sid in [1, 2, 3, 4]:
                    pos = self.controller.driver.read_position(sid)
                    if pos is not None:
                        cur_angle = raw_to_radian(sid, pos)
                        # 自重による負荷がかかる ID4 は許容誤差を 0.15 rad (約8.5度) に設定
                        threshold = 0.15 if sid == 4 else 0.08
                        if abs(cur_angle - home_rad[sid]) > threshold:
                            all_reached = False
                            break
                if all_reached:
                    break
                time.sleep(0.05)
            time.sleep(0.3)  # 静止安定化マージン

        print("✅ Home 姿勢への復帰が完了しました。")