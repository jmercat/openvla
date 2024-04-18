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


def rotmat_to_rot6d(mat):
    """
    Converts rotation matrix to R6 rotation representation (first two rows in rotation matrix).
    Args:
        mat: rotation matrix

    Returns: 6d vector (first two rows of rotation matrix)

    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    """
    Translates velocity actions (translation + rotation) from base frame of the robot to wrist frame.
    Args:
        velocity: 6d velocity action (3 x translation, 3 x rotation)
        wrist_in_robot_frame: 6d pose of the end-effector in robot base frame

    Returns: 9d velocity action in robot wrist frame (3 x translation, 6 x rotation as R6)

    """
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    # dR = rotmat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_velocity"][:, 3:6]))
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
    """
    DROID dataset transformation for actions expressed in *wrist* frame of the robot.
    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory["action_dict"]["cartesian_velocity"], trajectory["observation"]["cartesian_position"]
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
