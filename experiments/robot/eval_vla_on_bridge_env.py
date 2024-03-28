"""
eval_vla_on_bridge_env.py

Runs a VLA checkpoint in a real-world Bridge V2 environment.

Usage:
    python experiments/robot/eval_vla_on_bridge_env.py \
        --model.type <VLM_TYPE> \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --data_stats_path ./experiments/dataset_statistics/bridge_stats.json
"""

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

import draccus
import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

from prismatic.conf import ModelConfig, ModelRegistry
from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES
from prismatic.vla.action_tokenizer import ActionTokenizer

sys.path.append("./")  # hack so that the interpreter can find widowx_real_env
from experiments.robot.widowx_real_env import JaxRLWidowXEnv

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


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


def get_env():
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


def get_image_resize_size(vision_backbone_id: str) -> Tuple[int, int]:
    """Gets image resize size from vision backbone ID."""
    return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


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


def get_img(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
    # to minimize distribution shift.
    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
    # resized up to a different resolution by some models. This is just so that we're in-distribution
    # w.r.t. the original preprocessing at train time.
    img = obs["pixels"][0]
    img = Image.fromarray(img)
    BRIDGE_ORIG_IMG_SIZE = 256
    img = img.resize((BRIDGE_ORIG_IMG_SIZE, BRIDGE_ORIG_IMG_SIZE), Image.Resampling.LANCZOS)
    img = img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    return img


def get_vla_action(vla, image, task_label, tokenizer, action_tokenizer):
    """Generates an action with the VLA policy."""
    assert image.size[0] == image.size[1]
    prompt_builder = vla.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=f"What action should the robot take to {task_label.lower()}?")
    prompt_text = prompt_builder.get_prompt()
    generated_text = vla.generate_with_prompt(image, prompt_text, max_new_tokens=ACTION_DIM, do_sample=False)
    predicted_action_token_ids = torch.unsqueeze(
        torch.Tensor(tokenizer(generated_text)["input_ids"][-ACTION_DIM:]).long(), dim=0
    ).to(DEVICE)
    normalized_action = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())[0]
    return normalized_action


def get_action_norm_metadata(data_stats_path):
    """Retrieves dataset statistics used for action unnormalization."""
    with open(data_stats_path, "r") as f:
        metadata = json.load(f)
    return metadata


def unnormalize_action(action, metadata):
    """Un-normalizes action to be in the original dataset scale.
    Loads in a file containing action stats (min, max, std, etc.) and uses those to
    scale the actions.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    To un-normalize, we solve for x:
        x = 0.5 * (y + 1) * (orig_high - orig_low) + orig_low

    action: numpy array, shape=(7,), dtype=np.float32
    """
    action_low = np.array(metadata["action"]["q01"])
    action_high = np.array(metadata["action"]["q99"])
    out = 0.5 * (action + 1) * (action_high - action_low) + action_low
    return out


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "llava"

    # Pre-trained VLA model checkpoint to load
    pretrained_checkpoint: Union[str, Path] = Path(
        "/scr/moojink/checkpoints/tri/reproduction-llava-v15+mx-bridge+n1+b32+x7/checkpoints/step-077500-epoch-00-loss=0.0488.pt"
    )

    # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(
            ModelRegistry.REPRODUCTION_7B.model_id
        )
    )

    # Dataset statistics path (for action unnormalization)
    data_stats_path: str = "./experiments/dataset_statistics/bridge_stats.json"

    # Environment-specific variables
    max_episodes = 50                                           # Maximum number of rollouts
    max_steps = 50                                              # Maximum number of steps per rollout

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # Get action unnormalization stats and image resize size.
    metadata = get_action_norm_metadata(cfg.data_stats_path)
    resize_size = get_image_resize_size(cfg.model.vision_backbone_id)
    # Get VLA policy, tokenizer, and action tokenizer.
    vla = get_vla(cfg)
    tokenizer = vla.llm_backbone.get_tokenizer()
    action_tokenizer = ActionTokenizer(tokenizer)
    # Initialize the WidowX environment.
    env = get_env()
    # Start evaluation.
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        task_label = get_next_task_label(task_label)
        rollout_images = []
        env.reset()
        env.start()
        t = 0
        zero_action_count = 0
        step_duration = 0.2  # divide 1 by this to get control frequency
        input(f"Press Enter to start episode {episode_idx+1}...")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                # Get environment observations.
                obs = env._get_obs()
                rollout_images.append(obs["pixels"][0])
                if time.time() > last_tstamp + step_duration:
                    print(f"t: {t}")
                    last_tstamp = time.time()
                    # Get preprocessed image.
                    img = get_img(obs, resize_size)
                    # Generate action with VLA model.
                    normalized_action = get_vla_action(vla, img, task_label, tokenizer, action_tokenizer)
                    action = unnormalize_action(normalized_action, metadata)
                    # End episode early if the robot doesn't move at all for a few consecutive steps.
                    if np.isclose(np.linalg.norm(action), 1, atol=0.01) and np.linalg.norm(action[:6]) < 0.01:
                        zero_action_count += 1
                        if zero_action_count == 5:
                            print("Ending episode early due to robot inaction.")
                            break
                    else:
                        zero_action_count = 0
                    # Execute action in environment.
                    tstamp_return_obs = last_tstamp + step_duration
                    print("action:", action)
                    _, _, _, _ = env.step({"action": action, "tstamp_return_obs": tstamp_return_obs})
                    t += 1
            except Exception as e:
                print(f"Caught exception: {e}")
                break
        save_rollout_gif(rollout_images, episode_idx)
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    main()
