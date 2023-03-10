import pybullet as p
import pybulletX as px


def reset_ur10_joints(robot: px.Robot, desired_joint_positions=None):
    if desired_joint_positions is None:
        desired_joint_positions = [0.0, -1.7, 2.6, 0.5, 1.4, 0]
    for i in range(len(robot.free_joint_indices)):
        p.resetJointState(robot.id, robot.free_joint_indices[i], desired_joint_positions[i])


def reset_ur10_cartesian(robot: px.Robot, desired_position=None, desired_orientation=None):
    """
    move digit to desired pose
    """
    if desired_position is None:
        desired_position = [0.3, 0.0, 0.01]
    if desired_orientation is None:
        desired_orientation = [0, 0, 0, 1]
    for i in range(5):  # far distance lead to imprecise of IK
        desired_joint_positions = p.calculateInverseKinematics(
            robot.id, robot.get_joint_index_by_name("digit_joint"), desired_position, desired_orientation,
        )
        reset_ur10_joints(robot, desired_joint_positions)
