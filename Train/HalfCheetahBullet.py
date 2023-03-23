import os

import pybullet_envs

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = SAC('MlpPolicy', env, verbose=1,tensorboard_log="./halfcheetah_tensorboard/")
model.learn(total_timesteps=4000)

# log_dir = "/tmp/"
# model.save(log_dir + "sac_halfcheetah")
# stats_path = os.path.join(log_dir, "vec_normalize.pkl")
# env.save(stats_path)

