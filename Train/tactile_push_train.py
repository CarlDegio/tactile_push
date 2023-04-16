import os
import gym
from stable_baselines3 import SAC

from Envs.normalize_wrapper import NormalizeWrapper


def push_ball_train():
    env = gym.make("tactile_push/PushBall-v1", seed=1, dense_reward=True)
    wrapperd_env = NormalizeWrapper(env)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = SAC("MlpPolicy",
                wrapperd_env,
                verbose=1,
                tensorboard_log=os.path.join(project_root, "Train/tactile_push_tensorboard/"),
                use_sde=True)
    for i in range(10):
        model.learn(total_timesteps=70000, reset_num_timesteps=False,tb_log_name="SAC_0415night")
        model.save(os.path.join(project_root, "Train/load_save/tactile_push_ball_sac"))
        model.save_replay_buffer(os.path.join(project_root, "Train/load_save/tactile_push_ball_sac_buffer"))

if __name__ == '__main__':
    push_ball_train()