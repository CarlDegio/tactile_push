import gym
import Envs
def main():
    env=gym.make("tactile_push/PushBall-v0")
    env.reset(seed=1)


if __name__ == "__main__":
    main()
