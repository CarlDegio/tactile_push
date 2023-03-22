import gym
from stable_baselines3 import SAC

from Envs.normalize_wrapper import NormalizeWrapper


def push_ball_train():
    env = gym.make("tactile_push/PushBall-v0")
    wrapperd_env = NormalizeWrapper(env)

    model = SAC("MlpPolicy", wrapperd_env, verbose=1, tensorboard_log="./Train/tactile_push_tensorboard/")
    model.learn(total_timesteps=300000)
    model.save("tactile_push_ball_sac")
    model.save_replay_buffer("tactile_push_ball_sac_buffer")


if __name__ == "__main__":
    push_ball_train()
