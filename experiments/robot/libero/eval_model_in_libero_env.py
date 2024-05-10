"""
eval_model_in_libero_env.py

Runs a model checkpoint in a simulated libero environment.

Usage:
    python experiments/robot/libero/eval_libero.py \
        --model.type <VLM_TYPE> \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
from prismatic.conf import ModelConfig, ModelRegistry

# TODO (moojink) Hack so that the interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_img,
)
from experiments.robot.utils import (
    get_action,
    get_image_resize_size,
    get_model,
    get_octo_policy_function,
    normalize_gripper_action,
)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclass
class GenerateConfig:
    # fmt: off

    # Pre-trained model class
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.REPRODUCTION_7B.model_id)
    )
    model_family: str = "llava"

    # Model Parameters
    pretrained_checkpoint: Union[str, Path] = Path(
        "/shared/karl/models/open_vla/siglip-224px+mx-oxe-magic-soup+n8+b32+x7/"
        "checkpoints/step-152500-epoch-27-loss=0.1637.pt"
    )

    unnorm_key: str = "libero_spatial"                          # Dataset name for action unnormalization
    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    # Task suite (options: libero_spatial, libero_object, libero_goal, libero_90, libero_10, libero_100)
    task_suite_name: str = "libero_spatial"

    # LIBERO-related args
    max_steps = 300                                             # Max number of steps per episode
    num_steps_wait = 10                                         # Number of steps to wait for objs to stabilize in sim
    num_trials_per_task = 5                                     # Number of rollouts per task
    save_videos_per_task = 5                                    # Number of videos to be logged per task
    video_temp_subsample = 5                                    # Temporal subsampling to make videos shorter

    # Weights & Biases
    wandb_project: str = "openvla"                              # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                      # Name of entity to log under

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Load Model
    model = get_model(cfg)

    # [Octo] Create JAX JIT-compiled policy function.
    policy_fn = None
    if cfg.model_family == "octo":
        policy_fn = get_octo_policy_function(model)

    # Initialize W&B
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}",
    )

    # Get Expected Image Dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize task suite (e.g., LIBERO-Object).
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    # Start evaluation.
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite.
        task = task_suite.get_task(task_id)

        # Initialize the LIBERO environment.
        env, task_description = get_libero_env(task, cfg.model_family, model)
        init_states = task_suite.get_task_init_states(task_id)

        # Start episodes.
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")

            # Reset environment.
            obs = env.reset()
            init_state_id = episode_idx
            env.set_init_state(init_states[init_state_id])

            # Setup.
            t = 0
            rollout_images = []
            print(f"Starting episode {task_episodes+1}...")
            while t < cfg.max_steps + cfg.num_steps_wait:
                try:
                    # NOTE: Do nothing for the first few timesteps, because the environment just drops objects
                    # and we need to wait for them to fall...
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image.
                    img = get_libero_img(obs, resize_size)
                    rollout_images.append(img)

                    # Generate action with model.
                    action = get_action(
                        cfg, model, {"full_image": img}, task_description, policy_function=policy_fn, octo_nowrap=True
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the env expects the latter.
                    action = normalize_gripper_action(action)

                    # Execute action in environment.
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"Caught exception: {e}")
                    break

            task_episodes += 1
            if task_successes < cfg.save_videos_per_task or task_episodes - task_successes < cfg.save_videos_per_task:
                # Save rollout GIF.
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {
                        f"{task_description}/{group}/{idx}": wandb.Video(
                            np.array(rollout_images[:: cfg.video_temp_subsample]).transpose(0, 3, 1, 2)
                        )
                    }
                )

        # Log and update total metrics
        wandb.log(
            {
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            }
        )
        total_episodes += task_episodes
        total_successes += task_successes
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Log total metrics
    wandb.log(
        {
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        }
    )


if __name__ == "__main__":
    eval_libero()
