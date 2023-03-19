import gym
import tacto
from gym import spaces
import numpy as np
import pybullet as p
import pybulletX as px

from Bullet import draw_debug, add_wall, reset, get_state
from DigitUtil import depth_process


class PushBallEnv0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, ):
        self.step_repeat = 24
        self.max_step = 80
        self.np_random = None
        self.step_num = 0

        px.init(mode=p.GUI)
        self.robot = px.Robot("Meshes/ur10_tactile.urdf", use_fixed_base=True, flags=1)
        self.sphere = px.Body("Meshes/sphere_small/sphere_small.urdf", base_position=[0.27, 0, 0.03],
                              use_fixed_base=False,
                              flags=1)
        self.digits = tacto.Sensor()
        self.digits.add_camera(self.robot.id, self.robot.get_joint_index_by_name("digit_joint"))
        self.digits.add_body(self.sphere)
        self.depth_kit = depth_process.DepthKit()

        self.desire_pos = np.array([0.2, 0.0, 0.01])
        self.desire_quaternion = np.array([0, 0, 0, 1])
        draw_debug.draw_frame(self.robot.get_joint_index_by_name("digit_joint"))
        draw_debug.draw_area(size=[0.05, 0.1, 0.05], position=[0.6, 0.0, 0])
        add_wall.add_walls()

        # 2d end effort pos and vel, 2d ball pos and vel, 1d ball angular and vel
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(0.2, 0.8, shape=(1,), dtype=float),
                "y": spaces.Box(-0.7, 0.7, shape=(1,), dtype=float),
                "angular": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "vx": spaces.Box(0, 0.12, shape=(1,), dtype=float),
                "vy": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "vangular": spaces.Box(-0.5, 0.5, shape=(1,), dtype=float),
                "ball_x": spaces.Box(0.2, 0.8, shape=(1,), dtype=float),
                "ball_y": spaces.Box(-0.7, 0.7, shape=(1,), dtype=float),
                "ball_vx": spaces.Box(0, 0.12, shape=(1,), dtype=float),
                "ball_vy": spaces.Box(-0.12, 0.12, shape=(1,), dtype=float),
                "tactile_mid": spaces.Box(0, 120, shape=(1,), dtype=float),
                "tactile_sum": spaces.Box(0, 120*160, shape=(1,), dtype=float),
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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        # TODO: numpy warning
        color, depth = self.digits.render()
        self.digits.updateGUI(color, depth)
        self.depth_kit.update_depth(depth[0])

        real_pos, real_quaternion = get_state.get_ee_pose(self.robot)
        real_linear_velocity, real_angular_velocity = get_state.get_ee_vel(self.robot)
        real_ball_pos = get_state.get_ball_pos(self.sphere)
        real_ball_vel = get_state.get_ball_vel(self.sphere)
        return {
            "x": real_pos[0],
            "y": real_pos[1],
            "angular": p.getEulerFromQuaternion(real_quaternion)[2],
            "vx": real_linear_velocity[0],
            "vy": real_linear_velocity[1],
            "vangular": real_linear_velocity[2],
            "ball_x": real_ball_pos[0],
            "ball_y": real_ball_pos[1],
            "ball_vx": real_ball_vel[0],
            "ball_vy": real_ball_vel[0],
            "tactile_mid": self.depth_kit.calc_center()[1],
            "tactile_sum": self.depth_kit.calc_total(),
        }

    def _get_info(self):
        return {
            "msg": "normal"
        }

    def reset(self, seed=None, options=None):
        self.seed(seed)

        start_y = self.np_random.uniform(low=-0.6, high=0.6)
        reset.reset_ur10_cartesian(self.robot, [0.2, start_y, 0.01], [0, 0, 0, 1])
        reset.reset_ball_pos(self.sphere, [0.27, start_y, 0.03])
        self.desire_pos = np.array([0.2, start_y, 0.01])
        self.desire_quaternion = np.array([0, 0, 0, 1])

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        action["forward"] = np.clip(action["forward"], self.action_space["forward"].low, self.action_space["forward"].high)
        action["horizontal"] = np.clip(action["horizontal"], self.action_space["horizontal"].low, self.action_space["horizontal"].high)
        action["rotate"] = np.clip(action["rotate"],self.action_space["rotate"].low, self.action_space["rotate"].high)
        # An episode is done iff the agent has reached the target

        for i in range(self.step_repeat):
            self.desire_pos[0] += action["forward"]
            self.desire_pos[1] += action["horizontal"]
            desire_rotate = p.getEulerFromQuaternion(self.desire_quaternion)[2] + action["rotate"]
            self.desire_quaternion = p.getQuaternionFromEuler([0, 0, desire_rotate])
            desire_joint_position = p.calculateInverseKinematics(
                self.robot.id, self.robot.get_joint_index_by_name("digit_joint"), self.desire_pos,
                self.desire_quaternion
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
        if get_state.check_ball_in_region(self.sphere, region_x=[0.55, 0.65], region_y=[-0.1, 0.1]):
            reward = 1
        else:
            reward = 0
        if self.step_num >= self.max_step:
            done = True
        else:
            done = False


        return observation, reward, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
