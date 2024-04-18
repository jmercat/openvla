import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    return vec / norm
    out = (vec.T / norm).T
    return out


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
