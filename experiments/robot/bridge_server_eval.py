"""
Simple script that runs Bridge eval loop and pipes actions from VLA server to robot control server.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import click
import cv2
import draccus
import imageio
import json_numpy
import numpy as np
import requests
from widowx_env import WidowXGym
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

from experiments.robot.utils import get_image

json_numpy.patch()

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.25, -0.05, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
# WORKSPACE_BOUNDS = [[0.2, -0.13, 0.06, -1.57, 0], [0.33, 0.13, 0.25, 1.57, 0]] # sink
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


@dataclass
class VLABridgeConfig:
    # fmt: off
    # VLA server port
    vla_port: int = 8000                        # Port on which VLA server is running

    # Robot client configuration
    robot_host: str = "0.0.0.0"                 # Host that is running the WidowX controller
    robot_port: int = 5556                      # Port on which the WidowX controller is running

    # Environment parameters
    max_timesteps: int = 120                    # Maximum number of steps per episode
    blocking: bool = False                      # Whether to run with blocking control
    img_size: int = 224                         # Image resolution for VLA vision backbone
    sticky_gripper_steps: int = 1               # Number of timesteps for which gripper action is held constant

    # Misc
    save_dir: str = "/tmp"                      # Directory for saving output videos
    exp_name: str = ""                          # Prefix for experiment output folder
    show_image: bool = False                    # Whether to display robot observation during rollout
    verbose: bool = False                       # Whether to print out debug info

    # fmt: on


@draccus.wrap()
def run_vla_bridge(cfg: VLABridgeConfig):
    # Set up WidowX client
    if cfg.blocking:
        ENV_PARAMS["move_duration"] = 0.4
    else:
        assert ENV_PARAMS["move_duration"] == 0.2, STEP_DURATION_MESSAGE
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    widowx_client = WidowXClient(host=cfg.robot_host, port=cfg.robot_port)
    widowx_client.init(env_params, image_size=cfg.img_size)
    env = WidowXGym(widowx_client, cfg.img_size, cfg.blocking, cfg.sticky_gripper_steps)

    # Set up logging
    exp_dir = os.path.join(cfg.save_dir, f"{cfg.exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(exp_dir)

    instruction = ""
    while True:
        # Get instruction from user
        logging.info("Current instruction: ", instruction)
        if click.confirm("Take a new instruction?", default=True) or instruction == "":
            instruction = input("Instruction?")

        input("Press [Enter] to start.")

        # Reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        # Do rollout
        last_tstep = time.time()
        images = []
        t = 0

        try:
            while t < cfg.max_timesteps:
                if time.time() > last_tstep + STEP_DURATION:
                    last_tstep = time.time()

                    # Save images
                    images.append(obs["image_primary"][-1])

                    if cfg.show_image:
                        bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(20)

                    # Get action
                    forward_pass_time = time.time()
                    img = get_image(obs, cfg.img_size, key="image_primary")
                    action = requests.post(
                        f"http://0.0.0.0:{cfg.vla_port}/act",
                        json={"image": np.array(img), "instruction": instruction, "unnorm_key": "bridge_orig"},
                    ).json()
                    if cfg.verbose:
                        logging.info("Forward pass time: ", time.time() - forward_pass_time)

                    # Perform environment step
                    start_time = time.time()
                    obs, _, _, truncated, _ = env.step(action)
                    if cfg.verbose:
                        logging.info("Step time: ", time.time() - start_time)

                    t += 1

                    if truncated:
                        break
        except KeyboardInterrupt:
            obs, _ = env.reset()

        # Save video
        save_path = os.path.join(
            exp_dir,
            f"{instruction.replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4",
        )
        imageio.mimsave(save_path, np.stack(images), fps=1.0 / ENV_PARAMS["move_duration"] * 3)


if __name__ == "__main__":
    run_vla_bridge()
