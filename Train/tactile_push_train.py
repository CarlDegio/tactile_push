import gym
from stable_baselines3 import SAC

from Envs.normalize_wrapper import NormalizeWrapper

def main():
    env=gym.make("tactile_push/PushBall-v0")
    wrapperd_env=NormalizeWrapper(env)

    model = SAC("MlpPolicy", wrapperd_env, verbose=1, tensorboard_log="./tactile_push_tensorboard/")
    model.learn(total_timesteps=50000)
    model.save("tactile_push_sac")

if __name__ == "__main__":
    main()
