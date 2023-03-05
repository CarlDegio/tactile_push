import gym
import tacto
from gym import spaces
import numpy as np
import pybullet as p
import pybulletX as px


class PushBallEnv0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        px.init(mode=p.DIRECT)
        robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True, flags=1)
        sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.1, 0.5, 0.1],
                         use_fixed_base=False,
                         flags=1)
        digits = tacto.Sensor()
        digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
        digits.add_body(sphere)

        # 2d end effort pos and vel, 2d ball pos and vel, 1d ball angular and vel
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(0, 1, shape=(1,), dtype=int),
                "y": spaces.Box(0, 1, shape=(1,), dtype=int),
                "angular": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "vx": spaces.Box(0, 1, shape=(1,), dtype=float),
                "vy": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "vangular": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "ball_x": spaces.Box(0, 1, shape=(1,), dtype=float),
                "ball_y": spaces.Box(0, 1, shape=(1,), dtype=float),
                "ball_vx": spaces.Box(0, 1, shape=(1,), dtype=float),
                "ball_vy": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "tactile_mid": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "tactile_sum": spaces.Box(0, 1, shape=(1,), dtype=float),
            }
        )

        # end effort move
        self.action_space = spaces.Dict(
            {
                "forward": spaces.Box(0, 0.0005, shape=(1,), dtype=float),
                "horizontal": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "rotate": spaces.Box(-1, 1, shape=(1,), dtype=float),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        pass
