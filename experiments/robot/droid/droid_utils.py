import math

import cv2
import numpy as np
import requests
from droid.robot_env import RobotEnv
from PIL import Image
from scipy.spatial.transform import Rotation as R

from experiments.robot.utils import TemporalEnsembleWrapper, apply_center_crop


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    return vec / norm


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def R6_to_rmat(r6_mat):
    a1, a2 = r6_mat[:3], r6_mat[3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.vstack((b1, b2, b3))
    return out


def R6_to_euler(r6_mat):
    return rmat_to_euler(R6_to_rmat(r6_mat))


def get_droid_env(cfg):
    """Get DROID control environment."""
    env = RobotEnv(action_space=cfg.action_space, gripper_action_space=cfg.gripper_action_space)
    # (For Octo only) Wrap the robot environment.
    if cfg.model_family == "octo":
        env = TemporalEnsembleWrapper(env, pred_horizon=4)
    return env


def get_preprocessed_image(obs, resize_size):
    """
    Extracts image from observations and preprocesses it.

    Preprocess the image the exact same way that the dataset builder did it
    to minimize distribution shift.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Convert the image to RGB (as done in the dataset builder).
    img = Image.fromarray(obs["full_image"])
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image.
    img = Image.fromarray(img)
    DROID_ORIG_IMG_SIZE = (320, 180)  # (W, H)
    img = img.resize(DROID_ORIG_IMG_SIZE, Image.Resampling.LANCZOS)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = np.array(img)

    return img


def get_vla_server_action(cfg, obs, language_instruction, center_crop=False):
    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    image = Image.fromarray(obs["full_image"])
    if center_crop:
        temp_image = np.array(image)  # (H, W, C)
        crop_scale = 0.9
        sqrt_crop_scale = math.sqrt(crop_scale)
        temp_image_cropped = apply_center_crop(
            temp_image, t_h=int(sqrt_crop_scale * temp_image.shape[0]), t_w=int(sqrt_crop_scale * temp_image.shape[1])
        )
        temp_image = Image.fromarray(temp_image_cropped)
        temp_image = temp_image.resize(image.size, Image.Resampling.BILINEAR)  # IMPORTANT: dlimp uses BILINEAR resize
        image = temp_image

    action = requests.post(
        f"http://{cfg.vla_host}:{cfg.vla_port}/act",
        json={"image": np.array(image), "instruction": language_instruction, "unnorm_key": cfg.unnorm_key},
    ).json()
    return np.array(action)
