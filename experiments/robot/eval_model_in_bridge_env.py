"""
eval_model_in_bridge_env.py

Runs a model checkpoint in a real-world Bridge V2 environment.

Usage:
    # VLA:
    python experiments/robot/eval_model_in_bridge_env.py \
        --model.type <VLM_TYPE> \
        --pretrained_checkpoint <CHECKPOINT_PATH>

    # Octo:
    python experiments/robot/eval_model_in_bridge_env.py \
        --model_family octo
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus
import numpy as np

from prismatic.conf import ModelConfig, ModelRegistry

sys.path.append("./")  # hack so that the interpreter can find experiments.robot
from experiments.robot.utils import (
    get_action,
    get_image,
    get_image_resize_size,
    get_model,
    get_next_task_label,
    get_widowx_env,
    save_rollout_gif,
)


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

    # Environment-specific variables
    max_episodes = 50                                           # Maximum number of rollouts
    max_steps = 50                                              # Maximum number of steps per rollout
    control_frequency = 5                                       # Robot control frequency in Hz

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
    # Get image resize size.
    resize_size = get_image_resize_size(cfg)
    # Load model.
    model = get_model(cfg)
    # Initialize the WidowX environment.
    env = get_widowx_env()
    # Start evaluation.
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user.
        task_label = get_next_task_label(task_label)
        rollout_images = []
        # Reset environment.
        env.reset()
        env.start()
        # Setup.
        t = 0
        zero_action_count = 0
        step_duration = 1.0 / cfg.control_frequency
        # Start episode.
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
                    img = get_image(obs, resize_size)
                    # Query model to get action.
                    action = get_action(cfg, model, img, task_label)
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
        # Save a replay GIF of the episode.
        save_rollout_gif(rollout_images, episode_idx)
        # Redo episode or continue.
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    main()
