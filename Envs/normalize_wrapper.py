import gym
import numpy as np
from gym import spaces


class NormalizeWrapper(gym.Wrapper):
    """
    for sb3, normalize action [-1,1], obs [0,1]
    """

    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)
        self.env = env
        self.real_observation_space = self.env.observation_space
        self.real_action_space = self.env.action_space
        self.action_space = spaces.Box(shape=(3,), low=-1, high=1)
        if hasattr(self.env, 'has_tactile'):
            self.has_tactile = self.env.has_tactile()
        else:
            self.has_tactile = False  # TODO push on plane maybe error

        shape = 10
        if self.has_tactile:
            shape += 2
        if self.env.shape == "box":
            shape += 2

        self.observation_space = spaces.Box(shape=(shape,), low=0, high=1)

    def _normalize_obs(self, real_obs) -> np.ndarray:
        for key in real_obs.keys():
            real_obs[key] = (real_obs[key] - self.real_observation_space[key].low) / (
                    self.real_observation_space[key].high - self.real_observation_space[key].low)
        wrapper_obs = np.array([real_obs["x"], real_obs["y"], real_obs["angular"], real_obs["vx"], real_obs["vy"],
                                real_obs["vangular"], real_obs["ball_x"], real_obs["ball_y"], real_obs["ball_vx"],
                                real_obs["ball_vy"]])
        wrapper_obs = np.squeeze(wrapper_obs, axis=1)
        if self.has_tactile:
            wrapper_obs = np.append(wrapper_obs,[real_obs["tactile_mid"],real_obs["tactile_sum"]])
        if self.env.shape == "box":
            wrapper_obs = np.append(wrapper_obs, [real_obs["ball_angular"], real_obs["ball_vangular"]])
        return wrapper_obs

    def _denormalize_action(self, wrapper_action) -> dict:
        wrapper_action = (wrapper_action + 1) / 2
        real_action = {"forward": wrapper_action[0], "horizontal": wrapper_action[1], "rotate": wrapper_action[2]}
        for key in real_action.keys():
            real_action[key] = real_action[key] * (
                    self.real_action_space[key].high - self.real_action_space[key].low) + self.real_action_space[
                                   key].low
        return real_action

    def step(self, action) -> (np.ndarray, float, bool, dict):
        real_action = self._denormalize_action(action)
        observation, reward, done, info = self.env.step(real_action)
        observation = self._normalize_obs(observation)
        return observation, reward, done, info

    def reset(self, options=None):
        real_obs = self.env.reset(options=options)
        wrapper_obs = self._normalize_obs(real_obs)
        return wrapper_obs
