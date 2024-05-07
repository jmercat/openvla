import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES

try:
    # Only needed for Diffusion Policy
    import robomimic.utils.file_utils as FileUtils
    from droid.evaluation.policy_wrapper import PolicyWrapperRobomimic
except ImportError:
    pass

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "llava":
        resize_size = VISION_BACKBONES[cfg.model.vision_backbone_id]["kwargs"]["default_image_size"]
    elif cfg.model_family == "diffusion_policy":
        RESIZE_SIZE_OPTIONS = [128, 224]
        resize_size = int(input(f"Enter the resize size for diffusion policy (options: {RESIZE_SIZE_OPTIONS}): "))
        assert resize_size in RESIZE_SIZE_OPTIONS, "Invalid resize_size!"
    elif cfg.model_family == "octo":
        resize_size = 256
    elif cfg.model_family == "rt_1_x":
        resize_size = (640, 480)  # PIL expects (W, H)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def apply_center_crop(im, t_h, t_w):
    """
    Source: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/utils/obs_utils.py#L268

    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]


def normalize_gripper_action(action):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    This is necessary because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1]
    by default by the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # To implement, just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[-1] = 2 * (action[-1] - orig_low) / (orig_high - orig_low) - 1
    return action


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


###################################################################################################
# Model loading code
###################################################################################################


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


def get_diffusion_policy(cfg):
    """
    Loads a returns a Diffusion Policy model from checkpoint.
    Also returns camera kwargs for initializing the RobotEnv.
    """
    # Set random seed.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load checkpoint dictionary and config.
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=cfg.pretrained_checkpoint)
    config = json.loads(ckpt_dict["config"])

    # If action chunk size is specified on the command line, override it in the config.
    if cfg.dp_action_horizon:
        config["algo"]["horizon"]["action_horizon"] = cfg.dp_action_horizon
    print(f'Diffusion Policy action chunk size: {config["algo"]["horizon"]["action_horizon"]}')

    # Load policy.
    ckpt_dict["config"] = json.dumps(config)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=DEVICE, verbose=True)
    policy.goal_mode = config["train"]["goal_mode"]
    policy.eval_mode = True

    # Determine the action space from the saved train config.
    action_keys = config["train"]["action_keys"]
    if "action/rel_pos" in action_keys:
        action_space = "cartesian_velocity"
        for k in action_keys:
            assert not k.startswith("action/abs_")
    elif "action/abs_pos" in action_keys:
        action_space = "cartesian_position"
        for k in action_keys:
            assert not k.startswith("action/rel_")
    else:
        raise ValueError
    assert action_space == cfg.action_space, "Command-line action space arg does not match recorded action space!"

    # Determine the action space for the gripper.
    if "action/gripper_velocity" in action_keys:
        gripper_action_space = "velocity"
    elif "action/gripper_position" in action_keys:
        gripper_action_space = "position"
    else:
        raise ValueError

    # Prepare the policy wrapper.
    data_processing_kwargs = dict(
        timestep_filtering_kwargs=dict(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
        ),
        image_transform_kwargs=dict(
            remove_alpha=True,
            bgr_to_rgb=True,
            to_tensor=True,
            augment=False,
        ),
    )
    timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
    image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})
    policy_data_processing_kwargs = {}
    policy_timestep_filtering_kwargs = policy_data_processing_kwargs.get("timestep_filtering_kwargs", {})
    policy_image_transform_kwargs = policy_data_processing_kwargs.get("image_transform_kwargs", {})
    policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
    policy_image_transform_kwargs.update(image_transform_kwargs)
    fs = config["train"]["frame_stack"]
    wrapped_policy = PolicyWrapperRobomimic(
        policy=policy,
        timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
        image_transform_kwargs=policy_image_transform_kwargs,
        frame_stack=fs,
        eval_mode=True,
    )

    return wrapped_policy


def get_model(cfg):
    """Load model for evaluation."""
    if cfg.model_family == "llava":
        model = get_vla(cfg)
    elif cfg.model_family == "diffusion_policy":
        model = get_diffusion_policy(cfg)
    elif cfg.model_family == "octo":
        from octo.model.octo_model import OctoModel

        model = OctoModel.load_pretrained(cfg.pretrained_checkpoint)
    elif cfg.model_family == "rt_1_x":
        from experiments.baselines.rt_1_x.rt_1_x_policy import RT1XPolicy

        model = RT1XPolicy(saved_model_path=cfg.pretrained_checkpoint)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


###################################################################################################
# Action querying code
###################################################################################################


def get_vla_action(vla, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    if center_crop:
        temp_image = np.array(image)  # (H, W, C)
        crop_scale = 0.9
        temp_image_cropped = apply_center_crop(
            temp_image, t_h=int(crop_scale * temp_image.shape[0]), t_w=int(crop_scale * temp_image.shape[1])
        )
        temp_image = Image.fromarray(temp_image_cropped)
        temp_image = temp_image.resize(image.size, Image.Resampling.LANCZOS)
        image = temp_image

    action = vla.predict_action(image, task_label, unnorm_key=unnorm_key, do_sample=False)
    return action


def get_dp_action(model, obs, task_label):
    """Generates an action with the Diffusion Policy."""
    # Write task label to file, if file doesn't already exist.
    # (The Diffusion Policy class expects to see this file during inference.)
    if not os.path.exists("eval_params"):
        os.makedirs("eval_params")
    if not os.path.exists("eval_params/lang_command.txt"):
        with open("eval_params/lang_command.txt", "w") as file:
            file.write(task_label)

    # Get action.
    action = model.forward(obs)
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
        action = get_vla_action(model, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop)
        assert action.shape == (ACTION_DIM,)
    elif cfg.model_family == "diffusion_policy":
        action = get_dp_action(model, obs, task_label)
    elif cfg.model_family == "octo":
        action = get_octo_action(model, obs, task_label, policy_function)
    elif cfg.model_family == "rt_1_x":
        action = get_rt_1_x_action(model, obs, task_label)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action
