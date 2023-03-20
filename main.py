import gym
import numpy as np

from Envs.normalize_wrapper import NormalizeWrapper
def main():
    env=gym.make("tactile_push/PushBall-v0")
    wrapperd_env=NormalizeWrapper(env)
    observation=wrapperd_env.reset(seed=1)
    print(observation)
    done=False
    while done==False:
        observation, reward, done, info=wrapperd_env.step(np.array([0.1,0.0,0.0]))
        print(observation[0])
    print("finish")


if __name__ == "__main__":
    main()
