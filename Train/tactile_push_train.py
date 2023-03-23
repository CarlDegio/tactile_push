import gym
from stable_baselines3 import SAC

from Envs.normalize_wrapper import NormalizeWrapper


def push_ball_train():
    env = gym.make("tactile_push/PushBall-v0",render_mode="human",seed=1)
    wrapperd_env = NormalizeWrapper(env)

    # model = SAC("MlpPolicy",
    #             wrapperd_env,
    #             verbose=1,
    #             tensorboard_log="./Train/tactile_push_tensorboard/",
    #             use_sde=True)
    # for i in range(5):
    #     model.learn(total_timesteps=20000, reset_num_timesteps=False,tb_log_name="SAC")
    #     model.save("./Train/tactile_push_ball_sac")
    #     model.save_replay_buffer("./Train/tactile_push_ball_sac_buffer")

    # model = SAC.load("./Train/tactile_push_ball_sac", env=wrapperd_env)
    # vec_env=model.get_env()
    # observation = vec_env.reset()
    # done=False
    # while done == False:
    #     action, _states = model.predict(observation,deterministic=True)
    #     observation, reward, done, info = vec_env.step(action)

# run in project root dir
