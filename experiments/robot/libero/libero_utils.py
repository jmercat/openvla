import os
from collections import deque

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from experiments.robot.utils import get_vla_action

try:
    # Only used for Octo
    import gym
except ImportError:
    pass


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    Note: this is a nearly exact copy of the TemporalEnsembleWrapper from Octo, but removes
      the action_space attribute since Libero envs do not expose this.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):
        super().__init__(env)
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight

        self.act_history = deque(maxlen=self.pred_horizon)

    def step(self, actions):
        assert len(actions) >= self.pred_horizon

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.act_history)]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)


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


###############################################################################
## Libero Action functions: Since env cannot be wrapped they need to be adapted
###############################################################################


def get_octo_action(model, obs, task_label, policy_function):
    """Generates an action with the Octo policy."""
    task = model.create_tasks(texts=[task_label])
    obs = {
        "image_primary": obs["full_image"][None],
        "timestep_pad_mask": np.ones((1,)),
    }
    action = np.array(policy_function(obs, task), dtype=np.float64)

    metadata = model.dataset_statistics["action"]
    mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
    return np.where(
        mask,
        (action * metadata["std"]) + metadata["mean"],
        action,
    )


def get_action(cfg, model, obs, task_label, policy_function=None):
    """Queries the model to get an action."""
    if cfg.model_family == "llava":
        action = get_vla_action(model, obs, task_label, unnorm_key=cfg.unnorm_key, center_crop=cfg.center_crop)
    elif cfg.model_family == "diffusion_policy":
        raise NotImplementedError
        # action = get_dp_action(model, obs, instruction)
    elif cfg.model_family == "octo":
        action = get_octo_action(model, obs, task_label, policy_function)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action
