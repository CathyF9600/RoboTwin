import json
import os
from typing import List

# 定义任务列表
TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch"
]

# 目标 JSONL 文件路径
JSON_PATH = "/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin.jsonl"

def generate_paths() -> List[str]:
    """生成所有任务的路径列表"""
    paths = []
    for task in TASKS:
        for id in range(50):  # 0 到 49 共 50 个 ID
            paths.append(f"/home/fyc/RoboTwin/data/{task}/demo_randomized/data/episode{id}.hdf5")
    return paths

import json
import os
from typing import List, Dict, Any

def update_jsonl_file(new_paths: List[str]):
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    if "datalist" not in data or not isinstance(data["datalist"], list):
        raise ValueError('"datalist" key not found or is not a list')

    data["datalist"].extend(new_paths)

    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    new_paths = generate_paths()
    update_jsonl_file(new_paths)