import os
from collections import deque

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from torchvision import transforms as T

from experiments.robot.utils import TemporalEnsembleWrapper, get_dp_frame_stack


def get_libero_env(task, model_family, model):
    """Initializes and returns the LIBERO environment along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
    env = OffScreenRenderEnv(**env_args)
    # (For Octo only) Wrap the robot environment.
    if model_family == "octo":
        env = TemporalEnsembleWrapper(env, pred_horizon=4)
    return env, task_description


def get_libero_dummy_action(model_family: str):
    if model_family == "octo":
        # TODO: don't hardcode the action horizon for Octo
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1])[None], (4, 1))
    else:
        return [0, 0, 0, 0, 0, 0, -1]


def get_libero_img(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = Image.fromarray(img)
    img = img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    return np.array(img)


class DiffusionPolicyWrapper:
    """Simple wrapper that accumulates past observations."""

    def __init__(self, cfg, policy):
        self.num_frames = get_dp_frame_stack(cfg.pretrained_checkpoint)
        self.policy = policy
        self.obs_history = None
        self.img_transform = T.ToTensor()

    def reset(self):
        self.obs_history = None
        self.policy.start_episode()

    def _set_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        self.obs_history = {}
        for k in init_obs:
            self.obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)],
                maxlen=self.num_frames,
            )

    def add_obs(self, obs):
        if self.obs_history is None:
            self._set_initial_obs_history(obs)

        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])

    def get_obs_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        if self.num_frames == 1:
            return {k: np.concatenate(self.obs_history[k], axis=0)[0] for k in self.obs_history}
        else:
            return {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}

    def forward(self, observation):
        obs = {
            "robot_state/cartesian_position": observation["state"][..., :6],
            "robot_state/gripper_position": observation["state"][..., -1:],
            "static_image": self.img_transform(observation["full_image"]),
        }

        # set item of obs as np.array
        for k in obs:
            obs[k] = np.array(obs[k])

        self.add_obs(obs)
        obs_history = self.get_obs_history()
        action = self.policy(obs_history)

        return action
