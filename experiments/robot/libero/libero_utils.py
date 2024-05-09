import os

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from experiments.robot.utils import TemporalEnsembleWrapper


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
