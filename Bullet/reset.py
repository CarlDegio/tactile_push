import pybullet as p
import pybulletX as px


def reset_ur10(robot: px.Robot, desired_joint_positions=None):
    if desired_joint_positions is None:
        desired_joint_positions = [0.0, -1.7, 2.6, 0.5, 1.4, 0]
    for i in range(len(robot.free_joint_indices)):
        p.resetJointState(robot.id, robot.free_joint_indices[i], desired_joint_positions[i])
