"""
Re-renders LIBERO dataset since there are subtle rendering differences between
provided offline data and environment rollouts.
"""

import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

import wandb
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

LIBERO_RAW_DATA_PATH = "/home/karl/code/LIBERO/datasets/libero_spatial"
LIBERO_TARGET_PATH = "/shared/karl/data/libero_raw/libero_spatial"

LIBERO_TASK_SUITE = "libero_spatial"
RESOLUTION = 128

WANBD_ENTITY = "stanford-voltron"
WANDB_PROJECT = "openvla"
WANDB_VIDEO_SUBSAMPLE_RATE = 5
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


# Generate task suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[LIBERO_TASK_SUITE]()
num_tasks_in_suite = task_suite.n_tasks

# Set up wandb logging so we can save a few demos
wandb.init(
    entity=WANBD_ENTITY,
    project=WANDB_PROJECT,
    name=f"REGEN-{LIBERO_TASK_SUITE}-{DATE_TIME}",
)

replay_success = []
for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    # Get task in suite.
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, "llava", None, resolution=RESOLUTION)
    env.seed(0)

    # Get dataset for task
    orig_data_path = os.path.join(LIBERO_RAW_DATA_PATH, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]

    # Create output file for re-rendered demos
    new_data_path = os.path.join(LIBERO_TARGET_PATH, f"{task.name}_demo.hdf5")
    new_data_file = h5py.File(new_data_path, "w")
    grp = new_data_file.create_group("data")

    # Replay demos in original dataset
    for i in range(len(orig_data.keys())):
        demo_data = orig_data[f"demo_{i}"]
        actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        # Set initial state -- wait for a few steps for environment to settle
        env.reset()
        env.set_init_state(orig_states[0])
        for _ in range(5):
            obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

        # Replay actions and record frames
        states = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []
        for action in tqdm.tqdm(actions):
            states.append(env.sim.get_state().flatten())

            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            ee_states.append(
                np.hstack(
                    (
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                    )
                )
            )
            robot_states.append(
                np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
            )
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

            obs, reward, done, info = env.step(action.tolist())

        # Count number of successful replays
        replay_success.append(done)
        print(f"Avg replay success: {np.mean(replay_success)}")

        # Save replayed trajectories to HDF5
        dones = np.zeros(len(actions)).astype(np.uint8)
        dones[-1] = 1
        rewards = np.zeros(len(actions)).astype(np.uint8)
        rewards[-1] = 1
        assert len(actions) == len(agentview_images)

        ep_data_grp = grp.create_group(f"demo_{i}")

        obs_grp = ep_data_grp.create_group("obs")
        obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
        obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
        obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
        obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
        obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

        obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
        obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=np.stack(states))
        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)

        # Log first video from each task
        if i < 5:
            wandb.log(
                {
                    f"{task_description}": wandb.Video(
                        np.stack(agentview_images, axis=0)[::WANDB_VIDEO_SUBSAMPLE_RATE][:, ::-1, ::-1].transpose(
                            0, 3, 1, 2
                        )
                    )
                }
            )

    # Finished demo file -- closing
    orig_data_file.close()
    new_data_file.close()
