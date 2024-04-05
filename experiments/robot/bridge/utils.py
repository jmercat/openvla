"""Utils for evaluating policies in real-world robot environments."""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES

sys.path.append("../../")  # hack so that the interpreter can find widowx_real_env
from experiments.robot.bridge.widowx_stanford_env import JaxRLWidowXEnv

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_widowx_env():
    """Get WidowX control environment."""

    class AttrDict(defaultdict):
        __slots__ = ()
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    variant = AttrDict(lambda: False)
    env_params = {
        "fix_zangle": True,  # do not apply random rotations to start state
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [[0.1, -0.25, 0.095, -1.57, 0], [0.4, 0.25, 0.4, 1.57, 0]],
        "action_clipping": "xyz",
        "catch_environment_except": True,
        "add_states": variant.add_states,
        "from_states": variant.from_states,
        "reward_type": variant.reward_type,
        "start_transform": None,
        "randomize_initpos": "full_area",
    }
    env = JaxRLWidowXEnv(env_params)
    return env


def get_vla_image_resize_size(vision_backbone_id: str) -> int:
    """Gets VLA image resize size from vision backbone ID."""
    return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


def get_image_resize_size(cfg) -> int:
    """Gets image resize size (square)."""
    if cfg.model_family == "llava":
        resize_size = get_vla_image_resize_size(cfg.model.vision_backbone_id)
    elif cfg.model_family == "octo":
        resize_size = 256
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
        model = None  # TODO
    else:
        raise ValueError("Unexpected `model_family` found in config.")
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


def get_image(obs, resize_size, key="pixels"):
    """Extracts image from observations and preprocesses it."""
    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
    # to minimize distribution shift.
    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
    # resized up to a different resolution by some models. This is just so that we're in-distribution
    # w.r.t. the original preprocessing at train time.
    img = obs[key][0]
    img = Image.fromarray(img)
    BRIDGE_ORIG_IMG_SIZE = 256
    img = img.resize((BRIDGE_ORIG_IMG_SIZE, BRIDGE_ORIG_IMG_SIZE), Image.Resampling.LANCZOS)
    img = img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    return img


def get_vla_action(vla, image, task_label):
    """Generates an action with the VLA policy."""
    assert image.size[0] == image.size[1]
    action = vla.predict_action(image, task_label, unnorm_key="bridge_orig", do_sample=False)
    return action


def get_action(cfg, model, image, task_label):
    """Queries the model to get an action."""
    if cfg.model_family == "llava":
        action = get_vla_action(model, image, task_label)
    elif cfg.model_family == "octo":
        action = None  # TODO
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    assert action.shape == (ACTION_DIM,)
    return action
