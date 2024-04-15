"""
eval_model_in_bridge_env.py

Runs a model checkpoint in a real-world Bridge V2 environment.

Usage:
    # VLA:
    python experiments/robot/eval_model_in_bridge_env.py \
        --model.type <VLM_TYPE> \
        --pretrained_checkpoint <CHECKPOINT_PATH>

    # Octo:
    python experiments/robot/eval_model_in_bridge_env.py --model_family octo \
         --blocking True --control_frequency 2.5

    # RT-1-X:
    python experiments/robot/eval_model_in_bridge_env.py --model_family rt_1_x \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import draccus
import numpy as np

from prismatic.conf import ModelConfig, ModelRegistry

# TODO (@moojink) Hack so that the interpreter can find experiments.robot
sys.path.append("./")
from experiments.robot.utils import (
    get_action,
    get_image_resize_size,
    get_model,
    get_next_task_label,
    get_octo_policy_function,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_gif,
)


@dataclass
class GenerateConfig:
    # fmt: off

    # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.REPRODUCTION_7B.model_id)
    )
    model_family: str = "llava"                                 # Base VLM model family (for prompt builder)

    # Model Parameters
    pretrained_checkpoint: Union[str, Path] = Path(             # Pretrained VLA checkpoint to load
        "/scr/moojink/checkpoints/tri/reproduction-llava-v15+mx-bridge+n1+b32+x7/checkpoints/"
        "step-077500-epoch-00-loss=0.0488.pt"
    )

    # Environment-Specific Parameters
    host_ip: str = "localhost"
    port: int = 5556

    # Note (@moojink) =>> Setting initial orientation with a 30 degree offset -- more natural!
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False
    max_episodes: int = 50
    max_steps: int = 60
    control_frequency: float = 5

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Load Model --> Get Expected Image Dimensions
    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # [Octo] Create JAX JIT-compiled policy function.
    policy_fn = None
    if cfg.model_family == "octo":
        policy_fn = get_octo_policy_function(model)

    # Initialize the Widow-X Environment
    env = get_widowx_env(cfg, model)

    # === Start Evaluation ===
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get Task Description from User
        task_label = get_next_task_label(task_label)
        rollout_images = []

        # Reset Environment
        obs, _ = env.reset()

        # Setup
        t = 0
        zero_action_count = 0
        step_duration = 1.0 / cfg.control_frequency

        # Start Episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the Camera Image and Proprioceptive State
                    obs = refresh_obs(obs, env)

                    # Save Image for Rollout GIF =>> Switch on History / No History
                    if len(obs["full_image"].shape) == 4:
                        rollout_images.append(obs["full_image"][-1])
                    else:
                        rollout_images.append(obs["full_image"])

                    # Get Preprocessed Image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query Model --> Get Action
                    action = get_action(cfg, model, obs, task_label, policy_fn)

                    # [OpenVLA] End episode early if the robot doesn't move at all for a few consecutive steps!
                    #   - Reason: Inference is pretty slow with a single local GPU...
                    if (
                        cfg.model_family == "llava"
                        and np.isclose(np.linalg.norm(action), 1, atol=0.01)
                        and np.linalg.norm(action[:6]) < 0.01
                    ):
                        zero_action_count += 1
                        if zero_action_count == 5:
                            print("Ending episode early due to robot inaction.")
                            break
                    else:
                        zero_action_count = 0

                    # Execute Action
                    print("action:", action)
                    obs, _, _, _, _ = env.step(action)
                    t += 1

            except Exception as e:
                print(f"Caught exception: {e}")
                break

        # Save a Replay GIF of the Episode
        save_rollout_gif(rollout_images, episode_idx)

        # Redo Episode or Continue...
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
