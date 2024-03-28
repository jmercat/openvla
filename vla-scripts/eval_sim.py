"""
eval_vla_on_real2sim_env.py

Runs a VLA checkpoint in a real-to-sim environment.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

import draccus
import numpy as np
import real2sim
import torch
import tqdm
import wandb
from accelerate.utils import set_seed
from PIL import Image
from real2sim.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle

from prismatic.conf import ModelConfig, ModelRegistry
from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES
from prismatic.models.vlms import OpenVLA

assert "MS2_ASSET_DIR" in os.environ, (
    "Environment variable MS2_ASSET_DIR not set. "
    "Usage: `MS2_ASSET_DIR=./ManiSkill2_real2sim/data python test_real2sim.py ...`"
)

NUM_EPISODES = 5
MAX_STEPS = 100  # TODO: retrieve this from env
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_image_resize_size(vision_backbone_id: str) -> Tuple[int, int]:
    """Gets image resize size from vision backbone ID."""
    return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


def convert_maniskill(action):
    """
    Applies transforms to raw VLA action that Maniskill Real2sim env expects.
    Converts rotation to axis_angle.
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1] and binarizes.
    """
    assert action.shape[0] == 7

    # Change rotation to axis-angle
    roll, pitch, yaw = action[3], action[4], action[5]
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action[3:6] = action_rotation_ax * action_rotation_angle

    # Binarize final gripper dimension & map to [-1...1]
    action[-1] = 2.0 * (action[-1] > 0.5) - 1.0
    return action


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "llava"

    # Pre-trained VLA model checkpoint to load
    pretrained_checkpoint: Union[str, Path] = Path(
        "/shared/karl/models/open_vla/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/step-080000-epoch-09-loss=0.0987.pt"
    )

    # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(
            ModelRegistry.REPRODUCTION_7B.model_id
        )
    )

    # Environment
    env_name: str = 'widowx_spoon_on_towel'

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # (Optional)

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def eval_policy(cfg: GenerateConfig) -> None:
    resize_size = get_image_resize_size(cfg.model.vision_backbone_id)
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # initialize logging
    wandb.init(project="openvla", name=f"EVAL_{cfg.model.model_id}_{cfg.env_name}_{TIME}", entity="clvr")

    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(cfg.seed)

    # Load Base VLM from checkpoint path
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vla: OpenVLA = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLA parameter not in full precision: {param}"

    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(device)

    # Start evaluation.
    num_episodes = 0
    num_successes = 0
    # Initialize the real2sim environment.
    env = real2sim.make(cfg.env_name)
    for j in range(NUM_EPISODES):
        # Reset environment.
        obs, reset_info = env.reset()
        # Setup.
        t = 0
        instruction = env.get_language_instruction()
        print(f"Task: {instruction}")
        rollout_images = []
        print(f"Starting episode {j+1}...")
        done, truncated = False, False
        with tqdm.tqdm(total=MAX_STEPS) as pbar:
            while not (done or truncated):
                try:
                    t += 1
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    image = Image.fromarray(image)

                    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
                    # to minimize distribution shift.
                    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
                    # resized up to a different resolution by some models. This is just so that we're in-distribution
                    # w.r.t. the original preprocessing at train time.
                    IMAGE_BASE_PREPROCESS_SIZE = 256
                    image = image.resize(
                        (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), Image.Resampling.LANCZOS
                    )

                    image = image.resize(
                        (resize_size, resize_size), Image.Resampling.LANCZOS
                    )  # also resize to size seen at train time
                    image = image.convert("RGB")
                    rollout_images.append(np.array(image))
                    action = vla.predict_action(image, instruction)
                    action = convert_maniskill(action)
                    obs, reward, done, truncated, info = env.step(action)
                    if done:
                        num_successes += 1
                        break
                    pbar.update()
                except Exception as e:
                    print(f"Caught exception: {e}")
                    break
            wandb.log({"rollout_video": wandb.Video(np.array(rollout_images).transpose(0, 3, 1, 2))})
            num_episodes += 1
            print(f"# episodes completed: {num_episodes}, # successes: {num_successes}")


if __name__ == "__main__":
    eval_policy()
