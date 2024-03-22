"""
eval_vla_on_real2sim_env.py

Runs a VLA checkpoint in a real-to-sim environment.

Usage examples:
    VK_ICD_FILENAMES=/iris/u/moojink/prismatic-dev/nvidia_icd.json MS2_ASSET_DIR=/iris/u/moojink/Real2Sim/ManiSkill2_real2sim/data python vla-scripts/eval_vla_on_real2sim_env.py --model.type siglip-224px+7b --pretrained_checkpoint /scr/moojink/checkpoints/tri/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/checkpoints/step-080000-epoch-09-loss=0.0987.pt
    VK_ICD_FILENAMES=/iris/u/moojink/prismatic-dev/nvidia_icd.json MS2_ASSET_DIR=/iris/u/moojink/Real2Sim/ManiSkill2_real2sim/data python vla-scripts/eval_vla_on_real2sim_env.py --model.type siglip-224px+7b --pretrained_checkpoint /sphinx/u/moojink/checkpoints/tri/prismatic-dev/runs/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/checkpoints/step-080000-epoch-09-loss=0.0987.pt
"""
import draccus
import glob
import imageio
import json
import numpy as np
import os
import real2sim
import time
import torch
from accelerate.utils import set_seed
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from prismatic.conf import ModelConfig, ModelRegistry
from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES
from prismatic.vla.action_tokenizer import ActionTokenizer
from real2sim.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from typing import Optional, Tuple, Type, Union


assert 'MS2_ASSET_DIR' in os.environ, "Environment variable MS2_ASSET_DIR not set. Usage: `MS2_ASSET_DIR=./ManiSkill2_real2sim/data python test_real2sim.py ...`"

ACTION_DIM = 7
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_image_resize_size(vision_backbone_id: str) -> Tuple[int, int]:
    """Gets image resize size from vision backbone ID."""
    return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


def save_rollout_gif(rollout_images, idx, success, task_description):
    os.makedirs('./rollouts', exist_ok=True)
    processed_task_description = (
        task_description
        .lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:40]
    )
    gif_path = f'./rollouts/rollout-{TIME}-{idx+1}-success={success}-task={processed_task_description}.gif'
    imageio.mimsave(gif_path, rollout_images, loop=0, fps=30)
    print(f'Saved rollout GIF at path {gif_path}')


def get_vla_action(vlm, image, task_description, tokenizer, action_tokenizer, device):
    assert image.size[0] == image.size[1]
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=f"What action should the robot take to {task_description.lower()}?")
    prompt_text = prompt_builder.get_prompt()
    generated_text = vlm.generate_with_prompt(image, prompt_text, max_new_tokens=ACTION_DIM, do_sample=False)
    predicted_action_token_ids = torch.unsqueeze(torch.Tensor(tokenizer(generated_text)['input_ids'][-ACTION_DIM:]).long(), dim=0).to(device)
    normalized_action = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())[0]
    return normalized_action


def get_action_norm_metadata(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        metadata = json.load(f)
    return metadata


def unnormalize_action(action, metadata, skip_gripper_action=True):
    """Un-normalizes action to be in the original dataset scale.
    Loads in a file containing action stats (min, max, std, etc.) and uses those to
    scale the actions.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    To un-normalize, we solve for x:
        x = 0.5 * (y + 1) * (orig_high - orig_low) + orig_low

    action: numpy array, shape=(7,), dtype=np.float32
    """
    action_low = np.array(metadata['action']['q01'])
    action_high = np.array(metadata['action']['q99'])
    if skip_gripper_action:
        out = action
        if len(action.shape) == 1:
            out[:-1] = 0.5 * (action[:-1] + 1) * (action_high[:-1] - action_low[:-1]) + action_low[:-1]
        else:
            out[:,:-1] = 0.5 * (action[:,:-1] + 1) * (action_high[:-1] - action_low[:-1]) + action_low[:-1]
    else:
        out = 0.5 * (action + 1) * (action_high - action_low) + action_low
    return out


def normalize_gripper_action(action):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    This is necessary because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1]
    by default by the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # To implement, just normalize the last action to [-1,+1].
    orig_low, orig_high = 0., 1.
    action[-1] = 2 * (action[-1] - orig_low) / (orig_high - orig_low) - 1
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
def eval(cfg: GenerateConfig) -> None:
    resize_size = get_image_resize_size(cfg.model.vision_backbone_id)
    debug = False
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Get action unnormalization stats.
    dataset_statistics_path = "/shared/karl/models/open_vla/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/" \
                              "dataset_statistics_6660f1fd7de2514abf18365431cdbe7edf09f55ed235830a649d7a45f6ffc3a4.json"
    metadata = get_action_norm_metadata(dataset_statistics_path)

    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(cfg.seed)

    # Load Base VLM from checkpoint path
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Cast to half precision.
    vlm.vision_backbone.to(dtype=vlm.vision_backbone.half_precision_dtype)
    vlm.llm_backbone.to(dtype=vlm.llm_backbone.half_precision_dtype)
    vlm.to(dtype=vlm.llm_backbone.half_precision_dtype)
    vlm.to(device)

    # Create action tokenizer.
    tokenizer = vlm.llm_backbone.get_tokenizer()
    action_tokenizer = ActionTokenizer(tokenizer)

    # Start evaluation.
    num_episodes = 0
    num_successes = 0
    for i in range(10):
        # Initialize the real2sim environment.
        env = real2sim.make(cfg.env_name)
        for j in range(5):
            # Reset environment.
            obs, reset_info = env.reset()
            # Setup.
            t = 0
            os.makedirs('./temp', exist_ok=True)
            instruction = env.get_language_instruction()
            print(f'Task: {instruction}')
            rollout_images = []
            print(f"Starting episode {i+1}...")
            done, truncated = False, False
            while not (done or truncated):
                try:
                    t += 1
                    print(f't: {t}')
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    rollout_images.append(image)
                    image = Image.fromarray(image)

                    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
                    # to minimize distribution shift.
                    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
                    # resized up to a different resolution by some models. This is just so that we're in-distribution
                    # w.r.t. the original preprocessing at train time.
                    IMAGE_BASE_PREPROCESS_SIZE = 256
                    image = image.resize((IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), Image.Resampling.LANCZOS)

                    image = image.resize((resize_size, resize_size), Image.Resampling.LANCZOS) # also resize to size seen at train time
                    image = image.convert("RGB")
                    image.save(f'temp/{t}.png')
                    normalized_action = get_vla_action(vlm, image, instruction, tokenizer, action_tokenizer, device)
                    print('normalized_action:', normalized_action)
                    action = unnormalize_action(normalized_action, metadata)
                    action = normalize_gripper_action(action) # gripper action: [0,1] -> [-1,+1] (because the env expects the latter) # TODO
                    print('action:', action)
                    obs, reward, done, truncated, info = env.step(action)
                    if done == True:
                        num_successes += 1
                        break
                except Exception as e:
                    print(f"Caught exception: {e}")
                    break
            save_rollout_gif(rollout_images, num_episodes, success=done, task_description=instruction)
            num_episodes += 1
            print(f"# episodes completed: {num_episodes}, # successes: {num_successes}")


if __name__ == "__main__":
    eval()