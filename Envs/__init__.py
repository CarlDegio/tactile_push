from gym.envs.registration import register
from Envs.push_ball_env import PushBallEnv0
register(
    id="tactile_push/PushBall-v0",
    entry_point="Envs:PushBallEnv0",
)