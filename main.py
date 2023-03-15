import gym
import Envs
def main():
    env=gym.make("tactile_push/PushBall-v0")
    observation=env.reset(seed=1)
    print(observation)
    observation, reward, done, info=env.step({"forward": 0.0005, "horizontal": 0.0005, "rotate": 0.001})
    print(observation)


if __name__ == "__main__":
    main()
