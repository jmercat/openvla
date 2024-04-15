"""Utils for evaluating policies in real-world robot environments."""

import os
import sys
import time
from functools import partial
from pathlib import Path

import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES

sys.path.append("./")  # hack so that the interpreter can find experiments.robot
from experiments.robot.widowx_env import WidowXGym

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_widowx_env_params(cfg):
    """Gets (mostly default) environment parameters for the WidowX environment."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True
    return env_params


def get_widowx_env(cfg, model=None):
    """Get WidowX control environment."""
    # Set up the WidowX environment parameters.
    env_params = get_widowx_env_params(cfg)
    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_quat])
    env_params["start_state"] = list(start_state)
    # Set up the WidowX client.
    widowx_client = WidowXClient(host=cfg.host_ip, port=cfg.port)
    widowx_client.init(env_params)
    env = WidowXGym(
        widowx_client,
        cfg=cfg,
        blocking=cfg.blocking,
    )
    # (For Octo only) Wrap the robot environment.
    if cfg.model_family == "octo":
        from octo.utils.gym_wrappers import (
            HistoryWrapper,
            TemporalEnsembleWrapper,
            UnnormalizeActionProprio,
        )

        env = UnnormalizeActionProprio(env, model.dataset_statistics["bridge_dataset"], normalization_type="normal")
        env = HistoryWrapper(env, horizon=1)
        env = TemporalEnsembleWrapper(env, pred_horizon=1)
    return env


def get_vla_image_resize_size(vision_backbone_id: str) -> int:
    """Gets VLA image resize size from vision backbone ID."""
    return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "llava":
        resize_size = get_vla_image_resize_size(cfg.model.vision_backbone_id)
    elif cfg.model_family == "octo":
        resize_size = 256
    elif cfg.model_family == "rt_1_x":
        resize_size = (640, 480)  # PIL expects (W, H)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Prepare for model loading.
    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    set_seed(cfg.seed)
    # Load VLA checkpoint.
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vla = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def get_model(cfg):
    """Load model for evaluation."""
    if cfg.model_family == "llava":
        model = get_vla(cfg)
    elif cfg.model_family == "octo":
        from octo.model.octo_model import OctoModel

        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    elif cfg.model_family == "rt_1_x":
        from experiments.baselines.rt_1_x.rt_1_x_policy import RT1XPolicy

        model = RT1XPolicy(saved_model_path=cfg.pretrained_checkpoint)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # do nothing, let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_gif(rollout_images, idx):
    """Saves a GIF of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    gif_path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.gif"
    imageio.mimsave(gif_path, rollout_images, loop=0)
    print(f"Saved rollout GIF at path {gif_path}")


def resize_image(img, resize_size):
    """Takes numpy array corresponding to a single image and returns resized image as numpy array."""
    assert isinstance(resize_size, tuple)
    img = Image.fromarray(img)
    BRIDGE_ORIG_IMG_SIZE = (256, 256)
    img = img.resize(BRIDGE_ORIG_IMG_SIZE, Image.Resampling.LANCZOS)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    img = np.array(img)
    return img


def get_preprocessed_image(obs, resize_size):
    """
    Extracts image from observations and preprocesses it.

    Preprocess the image the exact same way that the Berkeley Bridge folks did it
    to minimize distribution shift.
    NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
                    resized up to a different resolution by some models. This is just so that we're in-distribution
                    w.r.t. the original preprocessing at train time.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    if len(obs["full_image"].shape) == 4:  # history included
        num_images_in_history = obs["full_image"].shape[0]
        assert resize_size[0] >= resize_size[1]  # in PIL format: (W, H) where W >= H
        W, H = resize_size
        new_images = np.zeros((num_images_in_history, H, W, obs["full_image"].shape[-1]), dtype=np.uint8)
        for i in range(num_images_in_history):
            new_images[i] = resize_image(obs["full_image"][i], resize_size)
        obs["full_image"] = new_images
    else:  # no history
        obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def get_octo_policy_function(model):
    """Returns a JAX JIT-compiled Octo policy function."""
    import jax

    # create policy function
    @jax.jit
    def sample_actions(
        pretrained_model,
        observations,
        tasks,
        rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    def supply_rng(f, rng):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return f(*args, rng=key, **kwargs)

        return wrapped

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        ),
        rng=jax.random.PRNGKey(0),
    )

    return policy_fn


def get_vla_action(vla, obs, task_label):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")
    assert image.size[0] == image.size[1]
    action = vla.predict_action(image, task_label, unnorm_key="bridge_orig", do_sample=False)
    return action


def get_octo_action(model, obs, task_label, policy_function):
    """Generates an action with the Octo policy."""
    task = model.create_tasks(texts=[task_label])
    obs = {
        "image_primary": obs["full_image"],
        # "proprio": obs["proprio"], <-- Octo paper says proprio makes performance worse
        "pad_mask": obs["pad_mask"],
    }
    action = np.array(policy_function(obs, task), dtype=np.float64)
    return action


def get_rt_1_x_action(model, obs, task_label):
    """Generates an action with the RT-1-X policy."""
    action = model.predict_action(obs, task_label)
    return action


def get_action(cfg, model, obs, task_label, policy_function=None):
    """Queries the model to get an action."""
    if cfg.model_family == "llava":
        action = get_vla_action(model, obs, task_label)
        assert action.shape == (ACTION_DIM,)
    elif cfg.model_family == "octo":
        action = get_octo_action(model, obs, task_label, policy_function)
    elif cfg.model_family == "rt_1_x":
        action = get_rt_1_x_action(model, obs, task_label)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    history_included = len(obs["full_image"].shape) == 4
    if history_included:
        obs["full_image"][-1] = new_obs["full_image"]
        obs["image_primary"][-1] = new_obs["image_primary"]
        obs["proprio"][-1] = new_obs["proprio"]
    else:
        obs["full_image"] = new_obs["full_image"]
        obs["image_primary"] = new_obs["image_primary"]
        obs["proprio"] = new_obs["proprio"]
    return obs
