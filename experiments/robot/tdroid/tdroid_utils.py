"""Utils for evaluating policies in real-world robot environments."""

import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

# TODO (@moojink) Hack so that the interpreter can find local packages
sys.path.append(".")

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


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


def save_rollout_video(rollout_images, idx):
    """Saves an MP4 of an episode."""
    # Convert to RGB since the images were captured with cv2 (which uses BGR).
    rollout_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in rollout_images]

    os.makedirs("./rollouts", exist_ok=True)
    mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def resize_image(img, resize_size):
    """Takes numpy array corresponding to a single image and returns resized image as numpy array."""
    assert isinstance(resize_size, tuple)
    return img


def get_preprocessed_image(obs, resize_size):
    """
    Extracts image from observations and preprocesses it.

    Preprocess the image the exact same way that the dataset builder did it
    to minimize distribution shift.
    NOTE (Moo Jin): Yes, we resize down to 270x360 first even though the image may end up being
                    resized up to a different resolution by some models. This is just so that we're in-distribution
                    w.r.t. the original preprocessing at train time.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Convert the image to RGB (as done in the dataset builder).
    img = cv2.cvtColor(obs["full_image"], cv2.COLOR_BGR2RGB)

    # Resize the image.
    img = Image.fromarray(img)
    MJK_PANDA_ORIG_IMG_SIZE = (360, 270)  # (W, H)
    img = img.resize(MJK_PANDA_ORIG_IMG_SIZE, Image.Resampling.LANCZOS)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    img = np.array(img)

    return img
