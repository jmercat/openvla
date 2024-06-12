"""
eval_model_in_libero_env.py

Runs a model checkpoint in a simulated libero environment.

Usage:
    OpenVLA:
        Notes:
            - Set `center_crop==False` if not using random crop image aug'
            - Set initial_states_path == "DEFAULT" to use the LIBERO task suite's default initial positions.

        python experiments/robot/libero/eval_model_in_libero_env.py \
            --model_family llava \
            --model.type <VLM_TYPE> \
            --action_space cartesian_velocity \
            --task_suite_name [ libero_object | libero_spatial ] \
            --pretrained_checkpoint <CHECKPOINT_PATH> \
            --initial_states_path <INITIAL_STATES_JSON_PATH> \
            --center_crop [ True | False ]

    Diffusion Policy:
        TODO

    Octo:
        TODO
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import tqdm
import wandb
from libero.libero import benchmark

from prismatic.conf import ModelConfig, ModelRegistry

# TODO (moojink) Hack so that the interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    DiffusionPolicyWrapper,
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
    update_dp_task_label,
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
                                                                #   (will be overridden by `task_suite_name`)
    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    # Diffusion Policy args
    dp_action_horizon: int = None                              # Action chunk size (None means use value in config)
    action_space: str = "cartesian_velocity"

    # Task suite (options: libero_spatial, libero_object, libero_goal, libero_90, libero_10, libero_100)
    task_suite_name: str = "libero_spatial"

    # Initial states file
    initial_states_path: str = "./experiments/robot/libero/libero_spatial_initial_states.json"

    # LIBERO-related args
    max_steps = 300                                             # Max number of steps per episode
    num_steps_wait = 10                                         # Number of steps to wait for objs to stabilize in sim
    num_trials_per_task = 50                                    # Number of rollouts per task
    save_videos_per_task = 50                                   # Number of videos to be logged per task
    video_temp_subsample = 5                                    # Temporal subsampling to make videos shorter

    # Weights & Biases
    wandb_project: str = "openvla"                              # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                      # Name of entity to log under

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 7                                               # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    if cfg.model_family == "diffusion_policy":
        assert not cfg.center_crop, "Expecting `center_crop==False` because DP wrapper already handles center cropping!"

    # Override action unnormalization key with task suite name.
    cfg.unnorm_key = cfg.task_suite_name

    # Load Model
    model = get_model(cfg)
    if cfg.model_family == "diffusion_policy":
        model = DiffusionPolicyWrapper(cfg, model)

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

    # Initialize task suite (e.g., LIBERO-Spatial).
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    # Load all initial states (these should match the initial states seen in the training demos).
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as F:
            all_initial_states = json.load(F)
        print(f"Using initial states from {cfg.initial_states_path}")
    else:
        print("Using default initial states")

    # Start evaluation.
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite.
        task = task_suite.get_task(task_id)

        # Initialize the LIBERO environment.
        env, task_description = get_libero_env(task, cfg.model_family, model, resolution=256)

        # If applicable, get default LIBERO initial states.
        if cfg.initial_states_path == "DEFAULT":
            initial_states = task_suite.get_task_init_states(task_id)

        # Start episodes.
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")

            # Reset environment.
            env.reset()

            # Either use the default initial states or fetch them from saved JSON.
            if cfg.initial_states_path == "DEFAULT":
                obs = env.set_init_state(initial_states[episode_idx])
            else:
                # Get keys for fetching initial episode state.
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                # Skip episodes that do not have demos (e.g. because the demos did not succeed).
                if episode_key not in all_initial_states[initial_states_task_key].keys():
                    print(f"Skipping task {task_id} episode {episode_idx} due to lack of corresponding demo!")
                    continue

                # Get initial state and use it to initialize env.
                init_state = np.array(all_initial_states[initial_states_task_key][episode_key])
                obs = env.set_init_state(init_state)

            # [Diffusion Policy] Reset observation history, action chunk queue, and language input.
            if cfg.model_family == "diffusion_policy":
                model.reset()
                update_dp_task_label(task_description)

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
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"])
                        ),
                    }
                    action = get_action(
                        cfg, model, observation, task_description, policy_function=policy_fn, octo_nowrap=True
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the env expects the latter.
                    action = normalize_gripper_action(action)

                    # Execute action in environment.
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
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

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log and update total metrics
        wandb.log(
            {
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            }
        )
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
