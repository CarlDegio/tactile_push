import os
import gym
from stable_baselines3 import SAC
import numpy as np
from Envs.normalize_wrapper import NormalizeWrapper
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"].item()
        done = self.locals["dones"].item()
        self.total_reward += reward
        if done:
            self.logger.record("ep_reward", self.total_reward)
            self.total_reward = 0
        return True


def push_ball_train():
    env = gym.make("tactile_push/PushBall-v1", seed=1, dense_reward=True, shape="sphere")
    wrapperd_env = NormalizeWrapper(env)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = SAC("MlpPolicy",
                wrapperd_env,
                verbose=1,
                tensorboard_log=os.path.join(project_root, "Train/tactile_push_tensorboard/"),
                use_sde=True)
    for i in range(3):
        model.learn(total_timesteps=50000, reset_num_timesteps=False, tb_log_name="SAC_dense_ball",
                    callback=TensorboardCallback(), log_interval=1)
        model.save(os.path.join(project_root, "Train/load_save/tactile_push_ball_sac"))
        model.save_replay_buffer(os.path.join(project_root, "Train/load_save/tactile_push_ball_sac_buffer"))


if __name__ == '__main__':
    push_ball_train()
