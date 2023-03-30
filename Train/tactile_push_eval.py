import gym
from stable_baselines3 import SAC

from Envs.normalize_wrapper import NormalizeWrapper


def push_ball_eval():
    env = gym.make("tactile_push/PushBall-v0", seed=1, dense_reward=True, render_mode="human")
    wrapperd_env = NormalizeWrapper(env)

    model = SAC.load("./Train/load_save/tactile_push_ball_sac_sde_6e5", env=wrapperd_env)
    vec_env = model.get_env()
    for i in range(5):
        observation = vec_env.reset()
        done = False
        while not done:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, done, info = vec_env.step(action)

# run in project root dir
