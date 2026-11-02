"""
==============================================================================
TACHIKOMA 軌道・認識統合 3D 鳥瞰ビジュアライザー (services/arm_worker/trajectory_visualizer.py)
==============================================================================
【モジュールの役割】
机上の実写カメラ画像（ArUcoマーカー補正済み正射影画像）を作図領域の底面 (Z=-2.0mm) に敷き詰め、
その上に SO-ARM100 の正確な順運動学 (FK) に基づくアーム手先 (TCP) の 3次元搬送放物線、
および Pick / Place の接地ピンを立体的に重ね描きして静止画レンダリングします。
MuJoCo や AVX 命令に一切依存せず、ノートPCや HeroBox 上で 1秒未満で高速出力します。

【コマンドライン引数と指定方法】
本スクリプトはスタンドアロンで動作し、以下の引数で任意の写真やモーション CSV を指定可能です。

1. --image : 底面に敷く机上カメラ画像のファイルパス
   - 指定例: --image data/camera_captures/topdown_warped.jpg
   - 指定例: --image data/sample_tabletop_photo/WIN_20260920_11_09_03_Pro.jpg
   - 斜めから撮影された生のカメラ写真であっても、内部の VisionProjector が
     4隅の ArUco マーカー (#0〜#3) を自動検出し、その場で机面を正方形に正射影切り抜きします。
   - ※ 省略時はリポジトリ内の既存サンプル写真から自動で探索・選択されます。

2. --csv : 手先軌道を描画するモーション CSV ファイルパス
   - 指定例: --csv motions/task_step1_red_wooden_block_20261029_150759.csv
   - 各時刻の関節角度 (q1〜q6 または id_1〜id_6) から SO-ARM100 の URDF モデルに
     準拠した 4x4 同次変換行列で爪先端 (TCP) の 3次元座標を復元します。
   - ※ 省略時は motions/ 配下にある最新の CSV ファイルが自動選択されます。

3. --out : 出力先画像ファイルの保存先パス (省略可)
   - 指定例: --out motions/my_custom_preview.png
   - ※ 省略時は motions/preview_trajectory_latest.png に上書き保存されます。

【実行コマンド例 (PowerShell / Bash)】
  # 最新のモーション CSV と自動検出画像で即時テスト実行
  python services/arm_worker/trajectory_visualizer.py

  # 特定の写真と CSV を明示して実行
  python services/arm_worker/trajectory_visualizer.py `
    --image data/camera_captures/topdown_warped.jpg `
    --csv motions/task_step1_red_wooden_block_20261029_150759.csv `
    --out motions/preview_step1.png
==============================================================================
"""

import os
import sys
import math
import time
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

# プロジェクトルートのパス解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from config.joint_config import SIM_OFFSETS, SIM_DIRECTIONS
except ImportError:
    SIM_OFFSETS = {1: 2130, 2: 973, 3: 3061, 4: 2368, 5: 3050, 6: 1837}
    SIM_DIRECTIONS = {1: -1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}

MOTIONS_DIR = os.path.join(BASE_DIR, "motions")


# ==============================================================================
# 1. 3D 同次変換幾何学ライブラリ (URDF / MuJoCo 厳密準拠)
# ==============================================================================
def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ], dtype=np.float64)


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ], dtype=np.float64)


def make_transform(xyz: List[float], rpy: List[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz
    return T


def raw_to_radian(sid: int, target_raw: int) -> float:
    offset = SIM_OFFSETS.get(sid, 2048)
    direction = SIM_DIRECTIONS.get(sid, 1.0)
    diff = (target_raw - offset) * direction
    return diff * (2.0 * math.pi / 4096.0)


def forward_kinematics_tcp(q: Dict[int, float]) -> np.ndarray:
    """assets/so100.urdf の kinematic tree を完全にトレースした順運動学 (FK)"""
    q1 = q.get(1, 0.0)
    q2 = q.get(2, 0.0)
    q3 = q.get(3, 0.0)
    q4 = q.get(4, 0.0)
    q5 = q.get(5, 0.0)

    # 1. Joint shoulder_pan (base -> shoulder)
    T_base_shoulder = make_transform([0.0, -0.0452, 0.0165], [1.57079, 0.0, 0.0])
    R_j1 = np.eye(4)
    R_j1[:3, :3] = rot_axis(np.array([0.0, 1.0, 0.0]), q1)
    T1 = T_base_shoulder @ R_j1

    # 2. Joint shoulder_lift (shoulder -> upper_arm)
    T_shoulder_upper = make_transform([0.0, 0.1025, 0.0306], [-1.8, 0.0, 0.0])
    R_j2 = np.eye(4)
    R_j2[:3, :3] = rot_axis(np.array([1.0, 0.0, 0.0]), q2)
    T2 = T1 @ T_shoulder_upper @ R_j2

    # 3. Joint elbow_flex (upper_arm -> lower_arm)
    T_upper_lower = make_transform([0.0, 0.11257, 0.028], [1.57079, 0.0, 0.0])
    R_j3 = np.eye(4)
    R_j3[:3, :3] = rot_axis(np.array([1.0, 0.0, 0.0]), q3)
    T3 = T2 @ T_upper_lower @ R_j3

    # 4. Joint wrist_flex (lower_arm -> wrist)
    T_lower_wrist = make_transform([0.0, 0.0052, 0.1349], [-1.0, 0.0, 0.0])
    R_j4 = np.eye(4)
    R_j4[:3, :3] = rot_axis(np.array([1.0, 0.0, 0.0]), q4)
    T4 = T3 @ T_lower_wrist @ R_j4

    # 5. Joint wrist_roll (wrist -> gripper)
    T_wrist_gripper = make_transform([0.0, -0.0601, 0.0], [0.0, 1.57079, 0.0])
    R_j5 = np.eye(4)
    R_j5[:3, :3] = rot_axis(np.array([0.0, 1.0, 0.0]), q5)
    T5 = T4 @ T_wrist_gripper @ R_j5

    # 6. Gripper link から爪先端把持点 (TCP) への固定オフセット [m]
    p_tcp_local = np.array([-0.010, -0.045, 0.0, 1.0], dtype=np.float64)
    p_tcp_base = T5 @ p_tcp_local

    x_mm = -p_tcp_base[1] * 1000.0
    y_mm = p_tcp_base[0] * 1000.0
    z_mm = p_tcp_base[2] * 1000.0

    return np.array([x_mm, y_mm, z_mm], dtype=np.float64)


def parse_motion_csv(csv_path: str) -> List[np.ndarray]:
    points = []
    if not os.path.exists(csv_path):
        return points

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = {}
            for sid in range(1, 7):
                k = f"q{sid}"
                if k in row and row[k] != "":
                    try:
                        q[sid] = float(row[k])
                    except ValueError:
                        q[sid] = 0.0
                else:
                    k_id = f"id_{sid}"
                    if k_id in row and row[k_id] != "":
                        try:
                            raw_val = int(row[k_id])
                            q[sid] = raw_to_radian(sid, raw_val)
                        except ValueError:
                            q[sid] = 0.0
                    else:
                        q[sid] = 0.0

            tcp = forward_kinematics_tcp(q)
            points.append(tcp)

    return points


# ==============================================================================
# 2. VisionProjector による厳密正射影 ＆ 物理座標連動
# ==============================================================================
def crop_tabletop_with_aruco(image_path: str) -> Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]:
    """
    VisionProjector の出力画像 (500x500px) と物理ワークスペースの範囲を厳密に整合する。
    左右反転 (水平ミラー) を補正してアームから見た時計回りの実配置 (#0, #1, #3, #2) に一致させる。
    """
    default_bounds = (90.0, 590.0, -300.0, 200.0)
    if not image_path or not os.path.exists(image_path):
        return None, default_bounds

    frame = cv2.imread(image_path)
    if frame is None:
        return None, default_bounds

    try:
        from core.vision_projector import VisionProjector
        projector = VisionProjector()
        if projector.update_homography(frame): 
            warped = projector.warp_to_topdown(frame, out_w=500, out_h=500) 
            if warped is not None:
                pts_phys = projector.marker_phys_xy 
                
                # 手前側 (ID 2, 3) と 奥側 (ID 0, 1) の平均 X 座標
                x_front = float((pts_phys[2][0] + pts_phys[3][0]) / 2.0)
                x_back = float((pts_phys[0][0] + pts_phys[1][0]) / 2.0)
                # 左側 (ID 0, 2) と 右側 (ID 1, 3) の平均 Y 座標
                y_left = float((pts_phys[0][1] + pts_phys[2][1]) / 2.0)
                y_right = float((pts_phys[1][1] + pts_phys[3][1]) / 2.0)

                # マーカー間隔 (400px に相当)
                dx = abs(x_back - x_front)
                dy = abs(y_right - y_left)
                scale_x = dx / 400.0
                scale_y = dy / 400.0

                # 画像全体 (0〜500px) の物理境界 [mm]
                x_min = x_front - 50.0 * scale_x
                x_max = x_back + 50.0 * scale_x
                y_min = y_left - 50.0 * scale_y
                y_max = y_right + 50.0 * scale_y

                # OpenCV (BGR) -> RGB 変換
                rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                
                # 上下反転 (X奥手前整合) + 左右反転 (Y左右ミラー補正)
                rgb_aligned = np.fliplr(np.flipud(rgb_warped))

                return rgb_aligned, (x_min, x_max, y_min, y_max)
    except Exception as e:
        print(f"⚠️ [Visualizer] VisionProjector 正射影例外: {e}")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame, default_bounds


# ==============================================================================
# 3. 局所最小点 (谷) 判定による Pick / Place 検出
# ==============================================================================
def find_pick_and_place_indices(zs: np.ndarray) -> Tuple[int, int]:
    n = len(zs)
    if n < 10:
        return 0, max(0, n - 1)

    valleys = []
    for i in range(3, n - 3):
        if zs[i] <= zs[i - 1] and zs[i] <= zs[i - 2] and zs[i] <= zs[i + 1] and zs[i] <= zs[i + 2]:
            valleys.append(i)

    if len(valleys) >= 2:
        pick_idx = valleys[0]
        place_idx = valleys[-1]
    elif len(valleys) == 1:
        pick_idx = valleys[0]
        place_idx = int(np.argmin(zs[pick_idx + 1:])) + pick_idx + 1 if pick_idx + 1 < n else pick_idx
    else:
        half = n // 2
        pick_idx = int(np.argmin(zs[:half]))
        place_idx = half + int(np.argmin(zs[half:]))

    return pick_idx, place_idx


# ==============================================================================
# 4. メインレンダリングルーチン
# ==============================================================================
def render_preview_image(
    image_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    start_t = time.perf_counter()

    # 1. 軌道点列の取得
    pts = parse_motion_csv(csv_path) if csv_path else []

    # 2. 机面画像の切り抜きと正確な物理境界取得
    warped_img, (x_min, x_max, y_min, y_max) = crop_tabletop_with_aruco(image_path)

    # 3. 描画初期化
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#1a1a24")
    ax.set_facecolor("#1a1a24")

    # 4. 底面画像の敷き詰め (Z=-2.0mm)
    if warped_img is not None:
        h, w, _ = warped_img.shape
        img_arr = warped_img / 255.0

        gx = np.linspace(x_min, x_max, h)
        gy = np.linspace(y_min, y_max, w)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        Z = np.full_like(X, -2.0)

        ax.plot_surface(
            X, Y, Z,
            rstride=max(1, h // 100),
            cstride=max(1, w // 100),
            facecolors=img_arr,
            shade=False,
            alpha=0.92,
            zorder=1
        )
    else:
        gx = np.linspace(x_min, x_max, 10)
        gy = np.linspace(y_min, y_max, 10)
        X, Y = np.meshgrid(gx, gy)
        Z = np.zeros_like(X)
        ax.plot_wireframe(X, Y, Z, color="#34495e", linewidth=0.8, alpha=0.6, zorder=1)

    # 5. 3次元爪先軌道のプロット
    if len(pts) > 1:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])

        # メイン軌道
        ax.plot(xs, ys, zs, color="#00d2d3", linewidth=3.2, label="TCP Trajectory", zorder=7)
        # 机面への影投影 (Z=0 平面)
        ax.plot(xs, ys, np.zeros_like(zs), color="#00d2d3", linestyle=":", linewidth=1.2, alpha=0.35, zorder=2)

        # Home 開始点
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="#54a0ff", s=80, edgecolors="white", linewidth=1.5, label="Home", zorder=8)

        # 局所最小点による Pick / Place の厳密判定
        pick_idx, place_idx = find_pick_and_place_indices(zs)

        # PICK マーカー (机面すれすれの接地)
        ax.scatter([xs[pick_idx]], [ys[pick_idx]], [zs[pick_idx]], color="#10ac84", s=140, marker="o", edgecolors="white", linewidth=2.0, label="Pick", zorder=9)
        ax.text(xs[pick_idx], ys[pick_idx], zs[pick_idx] + 8.0, f"PICK (Z={zs[pick_idx]:.0f}mm)", color="#1dd1a1", fontsize=9, fontweight="bold", zorder=10)

        # PLACE マーカー
        ax.scatter([xs[place_idx]], [ys[place_idx]], [zs[place_idx]], color="#ee5253", s=140, marker="^", edgecolors="white", linewidth=2.0, label="Place", zorder=9)
        ax.text(xs[place_idx], ys[place_idx], zs[place_idx] + 8.0, f"PLACE (Z={zs[place_idx]:.0f}mm)", color="#ff6b6b", fontsize=9, fontweight="bold", zorder=10)

    # 6. アーム土台原点 (0, 0, 0)
    ax.scatter([0], [0], [0], color="#576574", s=100, marker="s", edgecolors="white", linewidth=1.2, zorder=3)
    ax.text(0, 0, 8, "Arm Base (0,0)", color="#c8d6e5", fontsize=8, zorder=4)

    # 7. 表示範囲・アングル調整 (400x400mm ワークスペース全体をカバー)
    ax.set_xlim(0, 600)
    ax.set_ylim(-320, 260)
    ax.set_zlim(0, 240)

    ax.set_xlabel("X (Forward) [mm]", color="#c8d6e5", labelpad=12)
    ax.set_ylabel("Y (Lateral) [mm]", color="#c8d6e5", labelpad=12)
    ax.set_zlabel("Z (Height) [mm]", color="#c8d6e5", labelpad=8)
    ax.tick_params(colors="#8395a7", labelsize=8)

    # 見晴らしの良い鳥瞰アングル (仰角 34度、方位角 -58度)
    ax.view_init(elev=34, azim=-58)

    elapsed_sec = time.perf_counter() - start_t
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"TACHIKOMA 3D Trajectory & Visual Inspection ({ts_str}) [Render: {elapsed_sec:.2f}s]", color="#ffffff", fontsize=11, fontweight="bold", pad=12)
    ax.legend(facecolor="#222f3e", edgecolor="#576574", labelcolor="#c8d6e5", loc="upper left", fontsize=8)

    plt.tight_layout()

    # 8. 保存
    if not output_path:
        os.makedirs(MOTIONS_DIR, exist_ok=True)
        output_path = os.path.join(MOTIONS_DIR, "preview_trajectory_latest.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"🖼️ [Visualizer] 3D 鳥瞰画像を保存しました: {output_path} (所要時間: {elapsed_sec:.2f}秒)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TACHIKOMA 3D 軌道ビジュアライザー")
    parser.add_argument("--image", type=str, default=None, help="生カメラ画像パス")
    parser.add_argument("--csv", type=str, default=None, help="モーション CSV パス")
    parser.add_argument("--out", type=str, default=None, help="出力先画像パス")
    args = parser.parse_args()

    img_path = args.image
    if not img_path or not os.path.exists(img_path):
        for candidate in [
            os.path.join(BASE_DIR, "data", "camera_captures", "topdown_warped.jpg"),
            os.path.join(BASE_DIR, "data", "camera_captures", "vlm_detected_result.jpg"),
            os.path.join(BASE_DIR, "data", "sample_tabletop_photo", "WIN_20260920_11_09_03_Pro.jpg"),
        ]:
            if os.path.exists(candidate):
                img_path = candidate
                break

    csv_file = args.csv
    if not csv_file or not os.path.exists(csv_file):
        if os.path.exists(MOTIONS_DIR):
            c_files = [os.path.join(MOTIONS_DIR, f) for f in os.listdir(MOTIONS_DIR) if f.endswith(".csv")]
            if c_files:
                c_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                csv_file = c_files[0]

    out_file = args.out or os.path.join(MOTIONS_DIR, "preview_trajectory_latest.png")
    render_preview_image(image_path=img_path, csv_path=csv_file, output_path=out_file)