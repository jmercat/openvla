"""
eval_model_in_tdroid_env.py

Runs a model checkpoint in a real-world T-DROID (tabletop DROID) Panda robot environment.

Usage:
    OpenVLA:
        Note: Set `center_crop==False` if not using random crop image aug'

        python experiments/robot/tdroid/eval_model_in_tdroid_env.py \
            --model_family llava \
            --model.type <VLM_TYPE> \
            --action_space cartesian_velocity \
            --center_crop True \
            --unnorm_key <DATASET_NAME> \
            --pretrained_checkpoint <CHECKPOINT_PATH>

        Example:
            python experiments/robot/tdroid/eval_model_in_tdroid_env.py \
                --model_family llava \
                --model.type siglip-224px+7b \
                --action_space cartesian_velocity \
                --center_crop True \
                --unnorm_key mjk_panda_4 \
                --pretrained_checkpoint /scr/moojink/checkpoints/moojink/prismatic-dev/runs/
                  siglip-224px+mx-mjk_panda_4+n1+b32+x7--from_siglip224_oxe_magic_soup_152K_checkpoint
                  --image_aug/checkpoints/step-015000-epoch-681-loss=0.0004.pt

    Diffusion Policy:
        Note: Always set `center_crop==False` because DP wrapper already center crops

        python experiments/robot/tdroid/eval_model_in_tdroid_env.py \
            --model_family diffusion_policy \
            --action_space cartesian_position \
            --center_crop False \
            --pretrained_checkpoint <CHECKPOINT_PATH>

        Example:
            python experiments/robot/tdroid/eval_model_in_tdroid_env.py \
                --model_family diffusion_policy \
                --action_space cartesian_position \
                --center_crop False \
                --pretrained_checkpoint /iris/u/moojink/prismatic-dev/droid_dp_runs/droid/im/
                diffusion_policy/04-29-None/bz_128_noise_samples_8_sample_weights_1_dataset_names
                _mjk_panda_4_abs_cams_static_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/
                20240429213208/models/model_epoch_2900.pth

    Octo:

        python experiments/robot/tdroid/eval_model_in_tdroid_env.py \
            --model_family octo \
            --action_space cartesian_velocity \
            --pretrained_checkpoint /scr/moojink/checkpoints/kpertsch/octo_tdroid_task1_20240501_202241
"""

import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus

from prismatic.conf import ModelConfig, ModelRegistry

# TODO (@moojink) Hack so that the interpreter can find local packages
sys.path.append(".")
from experiments.robot.tdroid.tdroid_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_tdroid_env,
    save_rollout_video,
)
from experiments.robot.utils import (
    get_action,
    get_image_resize_size,
    get_model,
    get_octo_policy_function,
    normalize_gripper_action,
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

    # Note (@moojink) =>> Setting initial orientation with a 30 degree offset -- more natural!
    randomize_init_robot_state: bool = False
    camera_serial_num: str = "140122076178"

    max_episodes: int = 50
    max_steps: int = 150
    control_frequency: float = 5

    action_space: str = "cartesian_velocity"
    unnorm_key: str = ""

    center_crop: bool = False                                  # Center crop? (if trained w/ random crop image aug)

    # Diffusion Policy args
    dp_action_horizon: int = None                              # Action chunk size (None means use value found in config)

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
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    if cfg.model_family == "diffusion_policy":
        assert not cfg.center_crop, "Expecting `center_crop==False` because DP wrapper already handles center cropping!"

    # Load model.
    model = get_model(cfg, wrap_diffusion_policy_for_droid=True)

    # Get expected image dimensions.
    resize_size = get_image_resize_size(cfg)

    # [Octo] Create JAX JIT-compiled policy function.
    policy_fn = None
    if cfg.model_family == "octo":
        policy_fn = get_octo_policy_function(model)

    # Initialize the Panda environment.
    env = get_tdroid_env(cfg)

    # Start evaluation.
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user.
        task_label = get_next_task_label(task_label)
        rollout_images = []

        # Reset environment.
        env.reset(randomize=cfg.randomize_init_robot_state)

        # [Diffusion Policy] Reset the observation history.
        if cfg.model_family == "diffusion_policy":
            model.fs_wrapper.reset()

        # Setup.
        t = 0
        step_duration = 1.0 / cfg.control_frequency

        # Start episode.
        input(f"Press Enter to start episode {episode_idx+1}...")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Get observations.
                    obs_dict = env.get_observation()

                    # IMPORTANT: Make a copy of the observations. For some reason, the RealSense camera dies
                    #            if you use the original observations and store those in a list (e.g. for
                    #            a rollout video).
                    obs = deepcopy(obs_dict)

                    # Save image for rollout video.
                    image = obs["image"][cfg.camera_serial_num][:]  # shape: (H, W, C)
                    rollout_images.append(image)

                    obs["full_image"] = image

                    # Get preprocessed image.
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Override original camera obs with preprocessed one (necessary for Diffusion Policy).
                    obs["image"][cfg.camera_serial_num] = obs["full_image"]

                    # Query model to get action.
                    action = get_action(cfg, model, obs, task_label, policy_fn, octo_nowrap=True)

                    # Normalize gripper action [0,1] -> [-1,+1] because the env expects the latter.
                    action = normalize_gripper_action(action)

                    # Execute action in environment.
                    print("action:", action)
                    _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode.
        save_rollout_video(rollout_images, episode_idx)

        # Redo episode or continue.
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    main()
