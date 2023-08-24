import os

import cv2
import gym
import tacto
from gym import spaces
import numpy as np
import pybullet as p
import pybulletX as px

from Bullet import draw_debug, add_wall, reset, get_state, rotate_mapping
from DigitUtil import depth_process
import warnings

class PushBallEnv1(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode=None, seed=None, dense_reward=False, tactile=True):
        self.step_repeat = 24
        self.max_step = 80
        self.np_random = None
        self.step_num = 0
        self.seed(seed)
        self.dense_reward = dense_reward
        self.tactile = tactile
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
        self.robot = px.Robot(os.path.join(project_path, "Meshes/ur10_tactile.urdf"),
                              use_fixed_base=True, flags=1, base_position=[-0.1, 0, 0])
        self.sphere = px.Body(os.path.join(project_path, "Meshes/sphere_small/sphere_small.urdf"),
                              base_position=[0.32, 0, 0.03],
                              use_fixed_base=False,
                              flags=1)
        self.digits = tacto.Sensor()
        self.digits.add_camera(self.robot.id, self.robot.get_joint_index_by_name("digit_joint"))
        self.digits.add_body(self.sphere)
        self.depth_kit = depth_process.DepthKit()

        self.ViewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0,0,0],
                                                              distance=1.2,
                                                              pitch=-33,
                                                              yaw=90,
                                                              roll=0,
                                                              upAxisIndex=2)
        self.ProjectionMatrix = p.computeProjectionMatrixFOV(fov=90,
                                                             aspect=1.0,
                                                             nearVal=0.01,
                                                             farVal=10.0)

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
        self.observation_space_raw = spaces.Dict(
            {
                "x": spaces.Box(0.24, 0.68, shape=(1,), dtype=float),
                "y": spaces.Box(-0.65, 0.65, shape=(1,), dtype=float),
                "angular": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "vx": spaces.Box(0, 0.12, shape=(1,), dtype=float),
                "vy": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "vangular": spaces.Box(-0.5, 0.5, shape=(1,), dtype=float),
                "ball_x": spaces.Box(0.2, 0.8, shape=(1,), dtype=float),
                "ball_y": spaces.Box(-0.7, 0.7, shape=(1,), dtype=float),
                "ball_vx": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "ball_vy": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "dep": spaces.Box(0, 1, shape=(64, 64, 3), dtype=float),
                "scene_image": spaces.Box(0, 255, shape=(64, 64, 3), dtype=float)
            }
        )
        if self.has_tactile():
            # self.observation_space["tactile_mid"] = spaces.Box(0, 120, shape=(1,), dtype=float)
            # self.observation_space["tactile_sum"] = spaces.Box(0, 120 * 160 / 40, shape=(1,), dtype=float)
            # self.observation_space["rgb"] = spaces.Box(0, 255, shape=(64, 64, 3), dtype=float)
            # self.observation_space["dep"] = spaces.Box(0, 1, shape=(64, 64, 3), dtype=float)
            pass

        # end effort move
        self.action_space = spaces.Dict(
            {
                "forward": spaces.Box(-0.0001, 0.0005, shape=(1,), dtype=float),
                "horizontal": spaces.Box(-0.0005, 0.0005, shape=(1,), dtype=float),
                "rotate": spaces.Box(-0.002, 0.002, shape=(1,), dtype=float),
            }
        )

    def _get_obs(self):

        # 将末端和球的空间坐标还原到平面坐标
        real_pos, real_quaternion = get_state.get_ee_pose(self.robot)
        plane_project_pos = rotate_mapping.xoy_point_rotate_y_axis(real_pos, theta=-self.incline_rad)

        real_linear_velocity, real_angular_velocity = get_state.get_ee_vel(self.robot)
        plane_project_vel = rotate_mapping.xoy_point_rotate_y_axis(real_linear_velocity, theta=-self.incline_rad)

        self.ball_real_pos = get_state.get_ball_pos(self.sphere)
        self.ball_plane_pos = rotate_mapping.xoy_point_rotate_y_axis(self.ball_real_pos, theta=-self.incline_rad)
        real_ball_vel = get_state.get_ball_vel(self.sphere)
        ball_plane_project_vel = rotate_mapping.xoy_point_rotate_y_axis(real_ball_vel, theta=-self.incline_rad)

        obs = {
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
        }
        if self.has_tactile():
            color, depth = self.digits.render()
            if self.render_mode == "human":
                self.digits.updateGUI(color, depth)
            self.depth_kit.update_depth(depth[0])
            color_processed,depth_processed=self.process_image(color[0],self.depth_kit.depth)
            obs.update({
                # "tactile_mid": self.depth_kit.calc_center()[1],
                # "tactile_sum": self.depth_kit.calc_total(),
                # "rgb": color_processed,
                "dep": depth_processed
            })
        scene_image=self.get_scene_image()
        obs.update({"scene_image":scene_image})

        return obs

    def process_image(self,color,depth):
        color=color[:120,:,:]
        color=cv2.resize(color,(64,64),cv2.INTER_NEAREST)
        depth=depth[:120,:]
        depth=cv2.resize(depth,(64,64),cv2.INTER_NEAREST)
        depth=np.expand_dims(depth,axis=-1)
        depth=np.broadcast_to(depth,(64,64,3))
        # cv2.imshow("resize",color)
        return color,depth

    def get_scene_image(self):
        width, height, scene_image_rgb, scene_image_dep, mask = p.getCameraImage(width=128,
                                                                                 height=128,
                                                                                 viewMatrix=self.ViewMatrix,
                                                                                 projectionMatrix=self.ProjectionMatrix,
                                                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                                                 flags=p.ER_NO_SEGMENTATION_MASK)
        scene_image_rgb = scene_image_rgb[:, :, :3][:, :, ::-1]
        scene_image_rgb= cv2.resize(scene_image_rgb, (64, 64), cv2.INTER_NEAREST)

        # cv2.imshow("scene_image",scene_image_rgb)
        # cv2.waitKey(1)
        return scene_image_rgb

    def _get_info(self):
        return {
            "msg": "normal"
        }

    def _set_desire_pose(self, forward, horizontal, rotate):
        self.desire_plane_pos[0] += forward
        self.desire_plane_pos[0] = np.clip(self.desire_plane_pos[0], self.observation_space_raw["x"].low,
                                           self.observation_space_raw["x"].high)
        self.desire_plane_pos[1] += horizontal
        self.desire_plane_pos[1] = np.clip(self.desire_plane_pos[1], self.observation_space_raw["y"].low,
                                           self.observation_space_raw["y"].high)
        self.desire_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.desire_plane_pos, theta=self.incline_rad)

        desire_rotate_angle = p.getEulerFromQuaternion(self.desire_plane_quaternion)[2] + rotate
        desire_rotate_angle = np.clip(desire_rotate_angle, self.observation_space_raw["angular"].low,
                                      self.observation_space_raw["angular"].high)
        self.desire_plane_quaternion = p.getQuaternionFromEuler([0, 0, desire_rotate_angle])
        self.desire_real_quaternion = p.multiplyTransforms([0, 0, 0], self.scene_quaternion,
                                                           [0, 0, 0], self.desire_plane_quaternion)[1]

    def reset(self, seed=None, options=None):
        start_y = self.np_random.uniform(low=-0.4, high=0.4)
        self.desire_plane_pos = np.array([0.25, start_y, 0.01])
        self.desire_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.desire_plane_pos, theta=self.incline_rad)
        self.desire_plane_quaternion = np.array([0, 0, 0, 1])
        reset.reset_ur10_cartesian(self.robot, self.desire_real_pos, self.scene_quaternion)

        self.ball_plane_pos = np.array([0.32, start_y, 0.03])
        self.ball_real_pos = rotate_mapping.xoy_point_rotate_y_axis(self.ball_plane_pos, theta=self.incline_rad)
        reset.reset_ball_pos(self.sphere, self.ball_real_pos)

        observation = self._get_obs()
        if abs(observation['y']-observation['ball_y']) > 0.01:
            warnings.warn("reset error: y != ball_y")
            print("error observation:", observation)
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
        if get_state.check_ball_in_region_3d(self.sphere, self.incline_rad, region_x=[0.55, 0.65],
                                             region_y=[-0.1, 0.1]):
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

    def has_tactile(self):
        if self.tactile:
            return True
        else:
            return False

    @property
    def observation_space(self):
        spaces = {
            "vec": gym.spaces.Box(shape=(10,), low=0, high=1),
            "image": gym.spaces.Box(0, 255.0, shape=(64, 64, 3), dtype=float),
            "scene_image": gym.spaces.Box(0, 255.0, shape=(64, 64, 3), dtype=float)
        }
        return gym.spaces.Dict(spaces)