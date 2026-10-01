"""
==============================================================================
拡張机上ワールドステート統合ビルダー
(core/world_state_builder.py)
==============================================================================
【機能】
TabletopDetector の幾何計測データと MultimodalTagger の VLM 詳細分析を統合し、
LLM が高精度に物理・外観推論を行うためのリッチな World State JSON を生成する。
==============================================================================
"""

import time
import math
from typing import List, Dict, Any


def evaluate_reachability(distance_mm: float) -> str:
    """アームベースからの距離に基づき到達性と把持難易度を評価"""
    if distance_mm < 160.0:
        return "近接域 (アーム根元に近いため干渉注意)"
    elif distance_mm <= 330.0:
        return "最適作業域 (高い把持成功率が見込める)"
    elif distance_mm <= 390.0:
        return "延伸域 (自重たわみ・ピッチ角低下に注意)"
    else:
        return "限界域 (IK解が得られない可能性あり)"


def get_relative_region_label(x_mm: float, y_mm: float) -> str:
    """ロボット視点での直感的な領域表現"""
    depth = "手前" if x_mm < 250.0 else "奥"
    if y_mm < -50.0:
        horiz = "左側"
    elif y_mm > 50.0:
        horiz = "右側"
    else:
        horiz = "中央"
    return f"{depth}{horiz}"


def build_rich_world_state(
    detected_objs: List[Dict[str, Any]],
    object_profiles: Dict[int, Dict[str, Any]],
    place_x_mm: float = 180.0,
    place_y_mm: float = 160.0
) -> Dict[str, Any]:
    """
    OpenCV 幾何計測と VLM 物体属性を統合した高解像度 World State を生成
    """
    objects_list = []

    for idx, obj in enumerate(detected_objs):
        x_mm, y_mm = obj["phys_xy"]
        major_mm, minor_mm = obj["size_mm"]
        angle_deg = obj["angle_deg"]

        # VLM からのリッチなプロファイル情報 (未解析時はデフォルト値)
        prof = object_profiles.get(idx, {})
        display_name = prof.get("display_name", f"物体 #{idx}")
        category = prof.get("category", "unknown")
        visual_desc = prof.get("description", "特徴未解析の物体")
        color = prof.get("color", "不明")

        # 幾何・アスペクト比計算
        aspect_ratio = round(major_mm / max(1.0, minor_mm), 2)
        dist_to_base = math.hypot(x_mm, y_mm)
        dist_to_place = math.hypot(x_mm - place_x_mm, y_mm - place_y_mm)
        is_already_placed = dist_to_place < 45.0

        # SO-ARM100 の爪幅（最大約 65mm）に基づく物理把持可否判定
        is_graspable = minor_mm <= 65.0

        objects_list.append({
            "id": idx,
            "name": display_name,
            "category": category,
            "visual_description": visual_desc,
            "color": color,
            "physical": {
                "position_xy_mm": [round(x_mm, 1), round(y_mm, 1)],
                "size_mm": [round(major_mm, 1), round(minor_mm, 1)],
                "aspect_ratio": aspect_ratio,
                "angle_deg": round(angle_deg, 1),
                "is_graspable": is_graspable
            },
            "spatial_relations": {
                "region_label": get_relative_region_label(x_mm, y_mm),
                "distance_to_base_mm": round(dist_to_base, 1),
                "reachability": evaluate_reachability(dist_to_base),
                "is_in_place_area": is_already_placed
            }
        })

    return {
        "timestamp": time.time(),
        "robot_base_origin": [0.0, 0.0, 0.0],
        "total_objects": len(objects_list),
        "objects": objects_list,
        "workspace_zones": {
            "place_area": {
                "name": "指定排出・集積ゾーン (トレイ)",
                "center_xy_mm": [place_x_mm, place_y_mm],
                "radius_mm": 45.0
            }
        }
    }