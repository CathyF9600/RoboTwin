import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser
import argparse

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"
    print('opening config', config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = seed_list[-1] + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            print('traj_data', traj_data)
            input()
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)

def print_hdf5_structure(g, prefix=""):
    for key in g:
        item = g[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            print(f"{path} [Group]")
            print_hdf5_structure(item, path)
        else:
            print(f"{path} [Dataset] shape: {item.shape}")

import h5py
def get_key(hdf5_path):    
    
    with h5py.File(hdf5_path, "r") as f:
        print("Listing all keys:")
        print_hdf5_structure(f)
        
def save_camera_images_and_actions(hdf5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        print("Listing all keys:")
        print_hdf5_structure(f)
        
        # Detect cameras with RGB data
        cameras = []
        for cam in f["observation"]:
            if "rgb" in f[f"observation/{cam}"]:
                cameras.append(cam)

        if not cameras:
            print("No RGB camera data found in HDF5.")
            return

        print("Saving images from cameras:", cameras)

        num_frames = f[f"observation/{cameras[0]}/rgb"].shape[0]
        left_ee = f["endpose/left_endpose"][()]       # shape (T, 7)
        right_ee = f["endpose/right_endpose"][()]     # shape (T, 7)
        left_grip = f["endpose/left_gripper"][()]  # shape (T,)
        right_grip = f["endpose/right_gripper"][()]  # shape (T,)
        # Save images and print actions
        for frame_idx in range(2):
            # for cam in cameras:
            #     rgb_encoded = f[f"observation/{cam}/rgb"][frame_idx]
            #     img = cv2.imdecode(np.frombuffer(rgb_encoded, np.uint8), cv2.IMREAD_COLOR)
            #     img_path = os.path.join(output_dir, cam, f"frame_{frame_idx:04d}.jpg")
            #     cv2.imwrite(img_path, img)

            # EE pose
            lpos = left_ee[frame_idx]
            rpos = right_ee[frame_idx]
            lg = left_grip[frame_idx]
            rg = right_grip[frame_idx]

            print(f"Frame {frame_idx:04d}", lpos)
            print(f"  Left EE:  pos=({lpos[0]:.3f}, {lpos[1]:.3f}, {lpos[2]:.3f})  quat=({lpos[3]:.3f}, {lpos[4]:.3f}, {lpos[5]:.3f}, {lpos[6]:.3f})  gripper={lg:.2f}")
            # print(f"  Right EE: pos=({rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f})  quat=({rpos[3]:.3f}, {rpos[4]:.3f}, {rpos[5]:.3f}, {rpos[6]:.3f})  gripper={rg:.2f}")

        print(f"\nSaved {num_frames} frames per camera to '{output_dir}'")

def read_values(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        print("Listing all keys:")
        print_hdf5_structure(f)
        
        if "task_name" not in f:
            raise KeyError("No 'task_name' found in the HDF5 file.")
        task_name = f["task_name"][()].decode("utf-8")
        print("Task Name:", task_name)
        # Detect cameras with RGB data
        cameras = []
        for cam in f["observation"]:
            if "rgb" in f[f"observation/{cam}"]:
                cameras.append(cam)

        if not cameras:
            print("No RGB camera data found in HDF5.")
            return

        print("Saving images from cameras:", cameras)

        num_frames = f[f"observation/{cameras[0]}/rgb"].shape[0]
        left_ee = f["endpose/left_endpose"][()]       # shape (T, 7)
        right_ee = f["endpose/right_endpose"][()]     # shape (T, 7)
        left_grip = f["endpose/left_gripper"][()]  # shape (T,)
        right_grip = f["endpose/right_gripper"][()]  # shape (T,)
        try:
            left_new_key = f["endpose/left_endpose_xyzw"][()]
            right_new_key = f["endpose/right_endpose_xyzw"][()]
        except:
            left_new_key = None
            right_new_key = None
            print("No combined endpose found in HDF5.")
        # Save images and print actions
        for frame_idx in range(num_frames):
            # for cam in cameras:
            #     rgb_encoded = f[f"observation/{cam}/rgb"][frame_idx]
            #     img = cv2.imdecode(np.frombuffer(rgb_encoded, np.uint8), cv2.IMREAD_COLOR)
            #     img_path = os.path.join(output_dir, cam, f"frame_{frame_idx:04d}.jpg")
            #     cv2.imwrite(img_path, img)

            # EE pose
            lpos = left_ee[frame_idx]
            rpos = right_ee[frame_idx]
            lg = left_grip[frame_idx]
            rg = right_grip[frame_idx]
            lposn = left_new_key[frame_idx] if left_new_key is not None else lpos
            rposn = right_new_key[frame_idx] if right_new_key is not None else rpos
            

            print(f"Frame {frame_idx:04d}", lpos)
            print(f"  Left EE:  pos=({lpos[0]:.3f}, {lpos[1]:.3f}, {lpos[2]:.3f})  quat=({lpos[3]:.3f}, {lpos[4]:.3f}, {lpos[5]:.3f}, {lpos[6]:.3f})  gripper={lg:.2f}")
            print(f"  Right EE: pos=({rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f})  quat=({rpos[3]:.3f}, {rpos[4]:.3f}, {rpos[5]:.3f}, {rpos[6]:.3f})  gripper={rg:.2f}")
            try: 
                print(f"  Left EE:  pos=({lposn[0]:.3f}, {lposn[1]:.3f}, {lposn[2]:.3f})  quat=({lposn[3]:.3f}, {lposn[4]:.3f}, {lposn[5]:.3f}, {lposn[6]:.3f})  gripper={lg:.2f}")
                print(f"  Right EE: pos=({rposn[0]:.3f}, {rposn[1]:.3f}, {rposn[2]:.3f})  quat=({rposn[3]:.3f}, {rposn[4]:.3f}, {rposn[5]:.3f}, {rposn[6]:.3f})  gripper={rg:.2f}")
            except:
                print("No combined endpose found in HDF5.")
        # print(f"\nSaved {num_frames} frames per camera to '{output_dir}'")
        
def add_separate_swapped_endposes(hdf5_path):
    with h5py.File(hdf5_path, "a") as f:
        left_key = "endpose/left_endpose"
        right_key = "endpose/right_endpose"
        left_new_key = "endpose/left_endpose_xyzw"
        right_new_key = "endpose/right_endpose_xyzw"

        if left_key not in f or right_key not in f:
            raise KeyError("Missing left or right endpose in HDF5.")

        left = f[left_key][()]    # shape (T, 7)
        right = f[right_key][()]  # shape (T, 7)

        def swap_quat(data):
            pos = data[:, :3]
            quat = data[:, 3:]         # wxyz
            quat_swapped = quat[:, [1, 2, 3, 0]]  # xyzw
            return np.concatenate([pos, quat_swapped], axis=1)

        left_swapped = swap_quat(left)
        right_swapped = swap_quat(right)

        for key, arr in [(left_new_key, left_swapped), (right_new_key, right_swapped)]:
            if key in f:
                print(f"Overwriting existing dataset: {key}")
                del f[key]
            f.create_dataset(key, data=arr)
        
def add_lang(hdf5_path):
    with h5py.File(hdf5_path, "a") as f:  # Use append mode so we can add data
        print("Listing all keys:")
        print_hdf5_structure(f)

        # === Parse task name from path ===
        task_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(hdf5_path)))).replace("_", " ")   # demo_randomized -> open_microwave
        
        print(f"Parsed task name: {task_name}")

        if "task_name" in f:
            print("Overwriting existing task_name")
            del f["task_name"]
        f.create_dataset("task_name", data=task_name.encode("utf-8"))

        # === Load and add instructions ===
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(hdf5_path)),  # demo_randomized/
            "instructions",
            os.path.basename(hdf5_path).replace(".hdf5", ".json")
        )

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Instruction file not found: {json_path}")

        with open(json_path, "r") as jf:
            instr_data = json.load(jf)

        if "instructions" in f:
            print("Overwriting existing instructions")
            del f["instructions"]
        instr_group = f.create_group("instructions")

        for split in ["seen", "unseen"]:
            lines = instr_data.get(split, [])
            print("lines", lines)
            encoded_lines = [s.encode("utf-8") for s in lines]
            max_len = max(len(x) for x in encoded_lines) if encoded_lines else 1
            instr_group.create_dataset(
                split, data=encoded_lines, dtype=f"S{max_len}"
            )

        print("✅ Successfully added task_name and instructions to HDF5.")

if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    # parser = ArgumentParser()
    # parser.add_argument("task_name", type=str)
    # parser.add_argument("task_config", type=str)
    # parser = parser.parse_args()
    # task_name = parser.task_name
    # task_config = parser.task_config

    # main(task_name=task_name, task_config=task_config)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the HDF5 file")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to save output images")
    args = parser.parse_args()

    # read_values(args.hdf5_path)
    # input('add now')
    add_lang(args.hdf5_path)
    add_separate_swapped_endposes(args.hdf5_path)
    read_values(args.hdf5_path)