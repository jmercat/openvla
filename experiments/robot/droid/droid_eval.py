"""
Simple script that runs DROID eval loop and pipes actions from VLA server to robot.
"""

import logging
import time
from dataclasses import dataclass

import draccus
import json_numpy
import numpy as np
import requests
from droid.user_interface.eval_gui import EvalGUI
from droid_utils import R6_to_euler
from PIL import Image

json_numpy.patch()


@dataclass
class VLADroidEvalConfig:
    # fmt: off
    # VLA server parameters
    vla_host: str = "0.0.0.0"                   # Host for VLA server, default: localhost
    vla_port: int = 8000                        # Port on which VLA server is running

    # VLA action parameters
    dataset_name: str = "droid"                 # Dataset name used for un-normalization key

    # Environment parameters
    img_size: int = 224                         # Image resolution for VLA vision backbone
    camera_key: str = "28451778_left"           # Key for retrieving camera observation
    control_hz: int = 3                         # Control frequency for DROID environment

    # Misc
    verbose: bool = False                       # Whether to print out debug info

    # fmt: on


class OpenVLAPolicy:
    def __init__(self, cfg: VLADroidEvalConfig):
        self.cfg = cfg
        self.instruction = None

    def forward(self, observation):
        assert self.cfg.camera_key in observation["image"], (
            f"The camera key you specified, {self.cfg.camera_key} is "
            f"not part of the observation, choose from: {observation['image'].keys()}."
        )
        assert self.instruction is not None, "Please specify an instruction before calling the policy."

        # Prepare image observation
        image = observation["image"][self.cfg.camera_key][:, :, :3]
        image = image[:, :, ::-1]  # Flip channels BGR --> RGB
        image = Image.fromarray(image)
        image = image.resize((self.cfg.img_size, self.cfg.img_size), Image.Resampling.LANCZOS)

        # Request action from VLA server
        forward_pass_time = time.time()
        action = requests.post(
            f"http://{self.cfg.vla_host}:{self.cfg.vla_port}/act",
            json={"image": np.array(image), "instruction": self.instruction, "unnorm_key": self.cfg.dataset_name},
        ).json()
        if self.cfg.verbose:
            logging.info(f"Action: {action}")
            logging.info(f"Forward pass time: {time.time() - forward_pass_time}")

        # Convert R6 rotation to euler & absolute gripper action to relative
        rot_act_euler = R6_to_euler(action[3:9])
        relative_gripper_act = action[-1:] - observation["robot_state"]["gripper_position"]
        action = np.concatenate(
            (
                action[:3],
                rot_act_euler,
                relative_gripper_act,
            )
        )
        if self.cfg.verbose:
            logging.info(f"Converted action: {action}")

        return np.clip(action, -1, 1)

    def load_goal_imgs(self, goal_img_dict):
        raise NotImplementedError("OpenVLA policies currently do not support goal conditioning.")

    def load_lang(self, text):
        self.instruction = text


@draccus.wrap()
def run_droid_eval(cfg: VLADroidEvalConfig):
    policy = OpenVLAPolicy(cfg)
    EvalGUI(policy=policy, control_hz=cfg.control_hz)


if __name__ == "__main__":
    run_droid_eval()
