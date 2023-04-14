import pybullet as p
import numpy as np

from .rotate_mapping import xoy_point_rotate_y_axis
def add_box(size, position, quaternion=None):
    if quaternion is None:
        quaternion = [0, 0, 0, 1]
    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0, 0, 1, 1])
    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id,
                      basePosition=position, baseOrientation=quaternion)


def add_walls():
    add_box([0.4, 0.02, 0.05], [0.4, 0.7, 0.05])
    add_box([0.4, 0.02, 0.05], [0.4, -0.7, 0.05])
    add_box([0.02, 0.7, 0.05], [0.8, 0.0, 0.05])


def add_walls_incline(theta: float):
    """
    :param theta: incline rad
    :return:
    """
    quaternion = p.getQuaternionFromEuler([0, -theta, 0])
    add_box([0.4, 0.02, 0.05], xoy_point_rotate_y_axis([0.4, 0.7, 0.05], theta=theta), quaternion=quaternion)
    add_box([0.4, 0.02, 0.05], xoy_point_rotate_y_axis([0.4, -0.7, 0.05], theta=theta), quaternion=quaternion)
    add_box([0.02, 0.7, 0.05], xoy_point_rotate_y_axis([0.8, 0.0, 0.05], theta=theta), quaternion=quaternion)
    add_box([0.02, 0.7, 0.05], xoy_point_rotate_y_axis([0.2, 0.0, 0.05], theta=theta), quaternion=quaternion)
    add_box([0.4, 0.7, 0.05], xoy_point_rotate_y_axis([0.4, 0.0, -0.05], theta=theta), quaternion=quaternion)

