import os

import gym
import tacto
from gym import spaces
import numpy as np
import pybullet as p
import pybulletX as px

from Bullet import draw_debug, add_wall, reset, get_state, rotate_mapping
from DigitUtil import depth_process


class PushBallEnv1(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode=None, seed=None, dense_reward=False):
        self.step_repeat = 24
        self.max_step = 80
        self.np_random = None
        self.step_num = 0
        self.seed(seed)
        self.dense_reward = dense_reward
        self.incline_rad = np.pi / 180 * 10
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode is None:
            self.render_mode = "none"
        else:
            self.render_mode = render_mode

        if self.render_mode == "human":
            px.init(mode=p.GUI)
        else:
            px.init(mode=p.DIRECT)
        self.robot = px.Robot(os.path.join(project_path, "Meshes/ur10_tactile.urdf"), use_fixed_base=True, flags=1)
        self.sphere = px.Body(os.path.join(project_path, "Meshes/sphere_small/sphere_small.urdf"),
                              base_position=[0.32, 0, 0.03],
                              use_fixed_base=False,
                              flags=1)
        self.digits = tacto.Sensor()
        self.digits.add_camera(self.robot.id, self.robot.get_joint_index_by_name("digit_joint"))
        self.digits.add_body(self.sphere)
        self.depth_kit = depth_process.DepthKit()

        self.desire_plane_pos = np.array([0.25, 0.0, 0.01])
        self.desire_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.desire_plane_pos, theta=self.incline_rad)
        self.desire_plane_quaternion = np.array([0, 0, 0, 1])
        self.desire_real_quaternion = p.getQuaternionFromEuler([0, -self.incline_rad, 0])
        self.scene_quaternion = p.getQuaternionFromEuler([0, -self.incline_rad, 0])
        self.ball_plane_pos = np.array([0.32, 0.0, 0.03])
        self.ball_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.ball_plane_pos, theta=self.incline_rad)

        draw_debug.draw_frame(self.robot.get_joint_index_by_name("digit_joint"))
        draw_debug.draw_area(size=[0.05, 0.1, 0.05],
                             position=rotate_mapping.xoy_point_rotate_y_axis([0.6, 0, 0], theta=self.incline_rad),
                             quaternion=self.scene_quaternion)
        add_wall.add_walls_incline(self.incline_rad)

        # 2d end effort pos and vel, 2d ball pos and vel, 1d ball angular and vel
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(0.25, 0.68, shape=(1,), dtype=float),
                "y": spaces.Box(-0.65, 0.65, shape=(1,), dtype=float),
                "angular": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "vx": spaces.Box(0, 0.12, shape=(1,), dtype=float),
                "vy": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "vangular": spaces.Box(-0.5, 0.5, shape=(1,), dtype=float),
                "ball_x": spaces.Box(0.2, 0.8, shape=(1,), dtype=float),
                "ball_y": spaces.Box(-0.7, 0.7, shape=(1,), dtype=float),
                "ball_vx": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "ball_vy": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "tactile_mid": spaces.Box(0, 120, shape=(1,), dtype=float),
                "tactile_sum": spaces.Box(0, 120 * 160 / 40, shape=(1,), dtype=float),
            }
        )

        # end effort move
        self.action_space = spaces.Dict(
            {
                "forward": spaces.Box(0, 0.0005, shape=(1,), dtype=float),
                "horizontal": spaces.Box(-0.0005, 0.0005, shape=(1,), dtype=float),
                "rotate": spaces.Box(-0.002, 0.002, shape=(1,), dtype=float),
            }
        )

    def _get_obs(self):
        color, depth = self.digits.render()
        if self.render_mode == "human":
            self.digits.updateGUI(color, depth)
        self.depth_kit.update_depth(depth[0])

        # 将末端和球的空间坐标还原到平面坐标
        real_pos, real_quaternion = get_state.get_ee_pose(self.robot)
        plane_project_pos = rotate_mapping.xoy_point_rotate_y_axis(real_pos, theta=-self.incline_rad)

        real_linear_velocity, real_angular_velocity = get_state.get_ee_vel(self.robot)
        plane_project_vel = rotate_mapping.xoy_point_rotate_y_axis(real_linear_velocity, theta=-self.incline_rad)

        self.ball_real_pos = get_state.get_ball_pos(self.sphere)
        self.ball_plane_pos = rotate_mapping.xoy_point_rotate_y_axis(self.ball_real_pos, theta=-self.incline_rad)
        real_ball_vel = get_state.get_ball_vel(self.sphere)
        ball_plane_project_vel = rotate_mapping.xoy_point_rotate_y_axis(real_ball_vel, theta=-self.incline_rad)

        return {
            "x": plane_project_pos[0],
            "y": plane_project_pos[1],
            "angular": p.getEulerFromQuaternion(real_quaternion)[2],  # 恰好pybullet是按XYZ顺序转
            "vx": plane_project_vel[0],
            "vy": plane_project_vel[1],
            "vangular": real_angular_velocity[2] / np.cos(self.incline_rad),
            "ball_x": self.ball_plane_pos[0],
            "ball_y": self.ball_plane_pos[1],
            "ball_vx": ball_plane_project_vel[0],
            "ball_vy": ball_plane_project_vel[1],
            "tactile_mid": self.depth_kit.calc_center()[1],
            "tactile_sum": self.depth_kit.calc_total(),
        }

    def _get_info(self):
        return {
            "msg": "normal"
        }

    def _set_desire_pose(self, forward, horizontal, rotate):
        self.desire_plane_pos[0] += forward
        self.desire_plane_pos[0] = np.clip(self.desire_plane_pos[0], self.observation_space["x"].low,
                                           self.observation_space["x"].high)
        self.desire_plane_pos[1] += horizontal
        self.desire_plane_pos[1] = np.clip(self.desire_plane_pos[1], self.observation_space["y"].low,
                                           self.observation_space["y"].high)
        self.desire_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.desire_plane_pos, theta=self.incline_rad)

        desire_rotate_angle = p.getEulerFromQuaternion(self.desire_plane_quaternion)[2] + rotate
        desire_rotate_angle = np.clip(desire_rotate_angle, self.observation_space["angular"].low,
                                      self.observation_space["angular"].high)
        self.desire_plane_quaternion = p.getQuaternionFromEuler([0, 0, desire_rotate_angle])
        self.desire_real_quaternion = p.multiplyTransforms([0, 0, 0], self.scene_quaternion,
                                                           [0, 0, 0], self.desire_plane_quaternion)[1]

    def reset(self, seed=None, options=None):
        start_y = self.np_random.uniform(low=-0.6, high=0.6)
        self.desire_plane_pos = np.array([0.25, start_y, 0.01])
        self.desire_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.desire_plane_pos, theta=self.incline_rad)
        self.desire_plane_quaternion = np.array([0, 0, 0, 1])
        reset.reset_ur10_cartesian(self.robot, self.desire_real_pos, self.scene_quaternion)

        self.ball_plane_pos = np.array([0.32, start_y, 0.03])
        self.ball_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.ball_plane_pos, theta=self.incline_rad)
        reset.reset_ball_pos(self.sphere, self.ball_real_pos)

        observation = self._get_obs()
        self.step_num = 0

        return observation

    def step(self, action):
        action["forward"] = np.clip(action["forward"], self.action_space["forward"].low,
                                    self.action_space["forward"].high)
        action["horizontal"] = np.clip(action["horizontal"], self.action_space["horizontal"].low,
                                       self.action_space["horizontal"].high)
        action["rotate"] = np.clip(action["rotate"], self.action_space["rotate"].low, self.action_space["rotate"].high)

        for i in range(self.step_repeat):
            self._set_desire_pose(action["forward"], action["horizontal"], action["rotate"])
            desire_joint_position = p.calculateInverseKinematics(
                self.robot.id, self.robot.get_joint_index_by_name("digit_joint"), self.desire_real_pos,
                self.desire_real_quaternion
            )
            p.setJointMotorControlArray(
                bodyIndex=self.robot.id,
                jointIndices=self.robot.free_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=desire_joint_position
            )
            p.stepSimulation()
        self.step_num += 1

        observation = self._get_obs()
        info = self._get_info()
        if get_state.check_ball_in_region_3d(self.sphere,self.incline_rad, region_x=[0.55, 0.65], region_y=[-0.1, 0.1]):
            reward = 1
        else:
            reward = 0
        if self.dense_reward:
            reward -= get_state.calc_ball_to_goal_3d(self.sphere,
                                                     rotate_mapping.xoy_point_rotate_y_axis(
                                                              [0.6, 0, 0], theta=self.incline_rad)
                                                     )

        if self.step_num >= self.max_step:
            done = True
        else:
            done = False

        return observation, reward, done, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
