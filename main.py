import gym
import numpy as np

from Envs.normalize_wrapper import NormalizeWrapper
def main():
    env=gym.make("tactile_push/PushBall-v1",render_mode="human",seed=1,shape="box")
    wrapperd_env=NormalizeWrapper(env)
    observation=wrapperd_env.reset()
    print(observation)
    done=False
    while done==False:
        observation, reward, done, info=wrapperd_env.step(np.array([1,0.0,0.0]))
        print(observation[-3])
    observation=wrapperd_env.reset()
    done=False
    while done==False:
        observation, reward, done, info=wrapperd_env.step(np.array([1,0.0,0.0]))
        print(observation[-1])
    print("finish")


if __name__ == "__main__":
    main()
