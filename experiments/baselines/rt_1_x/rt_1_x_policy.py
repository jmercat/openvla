"""Code adapted from RT-1-X inference example notebook:
https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts


def normalize_task_name(task_name):
    """Preprocesses the input task string."""
    replaced = (
        task_name.replace("_", " ")
        .replace("1f", " ")
        .replace("4f", " ")
        .replace("-", " ")
        .replace("50", " ")
        .replace("55", " ")
        .replace("56", " ")
    )
    return replaced.lstrip(" ").rstrip(" ")


def unnormalize_action_with_bound(
    actions: tf.Tensor,
    orig_low: float,
    orig_high: float,
    safety_margin: float = 0,
    rescaled_max: float = 1.0,
    rescaled_min: float = -1.0,
) -> tf.Tensor:
    """Un-normalizes actions back to the original dataset scale."""
    actions = (actions - rescaled_min) / (rescaled_max - rescaled_min) * (orig_high - orig_low) + orig_low
    return tf.clip_by_value(
        actions,
        rescaled_min + safety_margin,
        rescaled_max - safety_margin,
    )


def rescale_action(action):
    """Rescales action."""
    action["world_vector"] = unnormalize_action_with_bound(
        action["world_vector"],
        orig_low=-0.05,
        orig_high=0.05,
        safety_margin=0.01,
        rescaled_max=1.75,
        rescaled_min=-1.75,
    )
    action["rotation_delta"] = unnormalize_action_with_bound(
        action["rotation_delta"],
        orig_low=-0.25,
        orig_high=0.25,
        safety_margin=0.01,
        rescaled_max=1.4,
        rescaled_min=-1.4,
    )
    return action


def to_bridge_action(from_step):
    """Convert model output to Bridge dataset action."""
    model_action = {}
    # Get delta XYS and rotation actions.
    model_action["world_vector"] = from_step["world_vector"]
    model_action["rotation_delta"] = from_step["rotation_delta"]
    # Get binary gripper open/close action.
    # For open_gripper in Bridge, 0 == fully close and 1 == fully open.
    # However, the model was trained such that -1 == fully open and +1 == fully close.
    # Therefore, we must map -1 -> 1 and +1 -> 0.
    gripper_action = from_step["gripper_closedness_action"].item()
    assert gripper_action in [-1.0, 1.0]
    model_action["gripper_closedness_action"] = tf.cond(
        pred=(gripper_action == 1.0),
        true_fn=lambda: tf.constant([0.0], dtype=tf.float32),
        false_fn=lambda: tf.constant([1.0], dtype=tf.float32),
    )
    # Un-normalize action.
    model_action = rescale_action(model_action)
    # Combine action into one numpy array.
    model_action = np.concatenate(
        (model_action["world_vector"], model_action["rotation_delta"], model_action["gripper_closedness_action"])
    )
    return model_action


class RT1XPolicy:
    """RT-1-X policy wrapper for Bridge environments."""

    def __init__(self, saved_model_path):
        physical_devices = tf.config.list_physical_devices("GPU")
        # Re-enable TensorFlow's GPU usage (in case it was disabled by Prismatic packages).
        tf.config.set_visible_devices(physical_devices, "GPU")
        # Prevent TensorFlow from immediately using all GPU VRAM; instead, grow usage over time.
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Load TF model checkpoint.
        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path, load_specs_from_pbtxt=True, use_tf_function=True
        )
        # Obtain a dummy observation, where the features are all 0.
        self.base_observation = tf_agents.specs.zero_spec_nest(
            tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation)
        )
        # Initialize the policy's state.
        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)
        # Load language model for embedding the task string.
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self.natural_language_embedding = None
        self.task_label = None

    def update_task_label(self, task_label):
        """Set the task label and embedding."""
        self.task_label = task_label
        with tf.device("/CPU:0"):
            self.natural_language_embedding = self.embed([task_label])[0]
        print(f"Updated task. Current task label: {self.task_label}")

    def predict_action(self, observations, task_label):
        """Predict robot action given observations and task label."""
        # Check if we need to update the task name embedding.
        if task_label != self.task_label:
            self.update_task_label(task_label)
        assert self.task_label is not None, "self.task_label not set in RT1XPolicy!"
        # Format the inputs: image and task name embedding.
        # As shown in the RT-1-X inference example notebook, we only use the latest frame.
        image = observations["full_image"]
        print("original image shape: ", np.array(image).shape)
        image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
        image = tf.cast(image, np.uint8)
        assert image.shape == (256, 320, 3), f"image shape is {image.shape}"
        self.base_observation["image"] = image.numpy().squeeze()
        self.base_observation["natural_language_embedding"] = self.natural_language_embedding
        # Feed the inputs through RT-1-X to get action.
        tfa_time_step = ts.transition(self.base_observation, reward=np.zeros((), dtype=np.float32))
        policy_step = self.tfa_policy.action(tfa_time_step, self.policy_state)
        action = policy_step.action
        action = to_bridge_action(action)
        # Set policy state.
        self.policy_state = policy_step.state
        return action
