
import gym
from stable_baselines3 import SAC,PPO,DDPG

env = gym.make("Pendulum-v1")
model = DDPG("MlpPolicy", env, verbose=1,tensorboard_log="./pendulum_tensorboard/")
model.learn(total_timesteps=40000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

env.close()