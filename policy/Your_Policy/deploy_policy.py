import sys

sys.path.append("./")

import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import importlib
import yaml
from argparse import ArgumentParser


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


def load_env(task_name, task_config):
    CONFIGS_PATH = "task_config"
    task = class_decorator(task_name)

    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
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
            raise ValueError("missing embodiment files")
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
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config

    task.setup_demo(**args)
    return task


def replay_actions_from_hdf5(env, hdf5_path, video_path):
    with h5py.File(hdf5_path, 'r') as f:
        left_ee = f["endpose/left_endpose_xyzw"][()]         # (T, 7)
        right_ee = f["endpose/right_endpose_xyzw"][()]       # (T, 7)
        left_grip = f["endpose/left_gripper"][()]       # (T,)
        right_grip = f["endpose/right_gripper"][()]     # (T,)

        # Reshape grippers to (T,1) so we can concat along last axis
        left_grip = left_grip[:, None]                  # (T,1)
        right_grip = right_grip[:, None]                # (T,1)

        # Final shape (T, 16)
        actions = np.concatenate([left_ee, left_grip, right_ee, right_grip], axis=-1)

        print(f"Loaded {len(actions)} actions from {hdf5_path}")

        # actions = f["joint_action/vector"][()]         # (T, 7)
        # right_ee = f["joint_action/right_endpose"][()]       # (T, 7)
        # left_grip = f["joint_action/left_arm"][()]       # (T,)
        # right_grip = f["joint_action/right_gripper"][()]     # (T,)        
    obs = env.reset()
    frames = []

    for action in tqdm(actions[:100], desc="Replaying"):
        print('action', action)
        env.take_action(action, action_type='ee')  # target_pose: np.array([x, y, z, qx, qy, qz, qw])
        # env.take_action(action, action_type='qpos') 
        obs = env.get_obs()
        print(obs.keys())  # look for image-related keys
        frame = obs["observation"]["head_camera"]["rgb"] 
        if frame is not None:
            frames.append(frame)

    save_video(frames, video_path)
    print(f"Saved video to: {video_path}")


def save_video(frames, output_path, fps=15):
    print(frames[0].shape)
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame in frames:
        if frame.shape[2] == 3:
            bgr_frame = frame[:, :, ::-1]  # Convert RGB to BGR for OpenCV
            out.write(bgr_frame)
    out.release()


def main():
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (envs.<task>)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (without .yml)")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to replay HDF5 file")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output video")
    args = parser.parse_args()

    env = load_env(args.task_name, args.task_config)
    replay_actions_from_hdf5(env, args.hdf5_path, args.output_path)


if __name__ == "__main__":
    main()

# python policy/Your_Policy/deploy_policy.py \
#     --task_name open_microwave \
#     --task_config demo_randomized \
#     --hdf5_path /home/dodo/fyc/RoboTwin/data/open_microwave/demo_randomized/data/episode0.hdf5 \
#     --output_path /home/dodo/fyc/RoboTwin/extracted_frames/output_video.mp4

# # import packages and module here


# def encode_obs(observation):  # Post-Process Observation
#     obs = observation
#     # ...
#     return obs


# def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
#     Your_Model = None
#     # ...
#     return Your_Model  # return your policy model


# def eval(TASK_ENV, model, observation):
#     """
#     All the function interfaces below are just examples
#     You can modify them according to your implementation
#     But we strongly recommend keeping the code logic unchanged
#     """
#     obs = encode_obs(observation)  # Post-Process Observation
#     instruction = TASK_ENV.get_instruction()

#     if len(
#             model.obs_cache
#     ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
#         model.update_obs(obs)

#     actions = model.get_action()  # Get Action according to observation chunk

#     for action in actions:  # Execute each step of the action
#         # see for https://robotwin-platform.github.io/doc/control-robot.md more details
#         TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
#         # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
#         observation = TASK_ENV.get_obs()
#         obs = encode_obs(observation)
#         model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


# def reset_model(model):  
#     # Clean the model cache at the beginning of every evaluation episode, such as the observation window
#     pass
