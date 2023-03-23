import pybullet as p
import pybulletX as px


def reset_ur10_joints(robot: px.Robot, desired_joint_positions=None):
    if desired_joint_positions is None:
        desired_joint_positions = [-0.84, -1.49, 2.7, 0.36, 1.57, -0.84]
    for i in range(len(robot.free_joint_indices)):
        p.resetJointState(robot.id, robot.free_joint_indices[i], desired_joint_positions[i])


def reset_ur10_cartesian(robot: px.Robot, desired_position=None, desired_orientation=None):
    """
    move digit to desired pose
    """
    if desired_position is None:
        desired_position = [0.2, 0.0, 0.01]
    if desired_orientation is None:
        desired_orientation = [0, 0, 0, 1]
    reset_ur10_joints(robot)  # avoid IK fall in another solution
    for i in range(5):  # far distance lead to imprecise of IK
        desired_joint_positions = p.calculateInverseKinematics(
            robot.id, robot.get_joint_index_by_name("digit_joint"), desired_position, desired_orientation,
        )
        reset_ur10_joints(robot, desired_joint_positions)


def reset_ball_pos(sphere: px.Body, desired_position=None):
    """
    move ball to desired pose
    """
    if desired_position is None:
        desired_position = [0.7, 0.0, 0.5]
    for i in range(3):
        p.resetBasePositionAndOrientation(sphere.id, desired_position, [0, 0, 0, 1])
