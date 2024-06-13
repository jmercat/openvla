import numpy as np
import tensorflow_datasets as tfds

import wandb

ds = tfds.load("droid_wipe", data_dir="/raid/datasets", split="train")

imgs = []
for episode in ds:
    for step in episode["steps"]:
        imgs.append(np.array(step["observation"]["exterior_image_2_left"]))

imgs = np.array(imgs[:: (15 * 5)])

wandb.init(entity="kpertsch", project="render_rlds")
wandb.log({"video": wandb.Video(imgs.transpose(0, 3, 1, 2), format="mp4")})
