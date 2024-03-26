"""Episode transforms for DROID dataset."""
from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


def rmat_to_euler(rot_mat):
    return tfg.euler.from_rotation_matrix(rot_mat)


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def invert_rmat(rot_mat):
    return tfg.rotation_matrix_3d.inverse(rot_mat)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def change_velocity_act_frame(velocity, frame):
    R_frame = euler_to_rmat(frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = mat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


def rand_swap_exterior_images(img1, img2):
    return tf.cond(
        tf.random.uniform(shape=[]) > 0.5,
        lambda: (img1, img2),
        lambda: (img2, img1)
    )


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_velocity"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    wrist_act = change_velocity_act_frame(
        trajectory["action_dict"]["cartesian_velocity"],
        trajectory["observation"]["cartesian_position"]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory