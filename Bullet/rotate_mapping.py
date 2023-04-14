import pybullet as p
import numpy as np


def xoy_point_rotate_y_axis(position: list or np.ndarray, theta: float):
    # theta:rad
    quaternion = p.getQuaternionFromEuler([0, -theta, 0])
    rotate_matrix = p.getMatrixFromQuaternion(quaternion)
    rotate_matrix = np.array([rotate_matrix[0:3], rotate_matrix[3:6], rotate_matrix[6:9]])
    new_position = rotate_matrix @ np.array(position)
    return new_position

