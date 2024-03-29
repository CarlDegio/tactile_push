import pybullet as p
import pybulletX as px
import numpy as np
from .rotate_mapping import xoy_point_rotate_y_axis


def get_ee_pose(robot: px.Robot):
    state = p.getLinkState(robot.id, robot.get_joint_index_by_name("digit_joint"))
    real_ee_position, real_ee_quaternion = state[0:2]
    return real_ee_position, real_ee_quaternion


def get_ee_vel(robot: px.Robot):
    state = p.getLinkState(robot.id, robot.get_joint_index_by_name("digit_joint"), computeLinkVelocity=1)
    real_ee_linear_velocity, real_ee_angular_velocity = state[6:8]
    return real_ee_linear_velocity, real_ee_angular_velocity


def get_ball_pos(ball: px.Body):
    ball_pos = ball.get_base_pose()[0]
    return ball_pos


def get_ball_vel(ball: px.Body):
    ball_vel = ball.get_base_velocity()[0]
    return ball_vel


def get_ball_quaternion(ball: px.Body):
    ball_angle = ball.get_base_pose()[1]
    return ball_angle


def get_ball_angle_vel(ball: px.Body):
    ball_angle_vel = ball.get_base_velocity()[1]
    return ball_angle_vel


def check_ball_in_region(ball: px.Body, region_x, region_y):
    ball_pos = get_ball_pos(ball)
    if region_x[0] < ball_pos[0] < region_x[1] and region_y[0] < ball_pos[1] < region_y[1]:
        return True
    else:
        return False


def check_ball_in_region_3d(ball: px.Body, theta, region_x, region_y):
    ball_pos = get_ball_pos(ball)
    ball_plane_pos = xoy_point_rotate_y_axis(ball_pos, -theta)
    if region_x[0] < ball_plane_pos[0] < region_x[1] and region_y[0] < ball_plane_pos[1] < region_y[1]:
        return True
    else:
        return False


def calc_ball_to_goal(ball: px.Body, goal_pos):
    ball_pos = get_ball_pos(ball)
    dis = np.sqrt((ball_pos[0] - goal_pos[0]) ** 2 + (ball_pos[1] - goal_pos[1]) ** 2)
    return dis


def calc_ball_to_goal_3d(ball: px.Body, goal_pos):
    ball_pos = get_ball_pos(ball)
    dis = np.sqrt((ball_pos[0] - goal_pos[0]) ** 2 +
                  (ball_pos[1] - goal_pos[1]) ** 2 +
                  (ball_pos[2] - goal_pos[2]) ** 2)
    return dis
