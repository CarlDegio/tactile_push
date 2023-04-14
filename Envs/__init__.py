from gym.envs.registration import register
from Envs.push_ball_env import PushBallEnv0
from Envs.push_ball_incline_env import PushBallEnv1
register(
    id="tactile_push/PushBall-v0",
    entry_point="Envs:PushBallEnv0",
)
register(
    id="tactile_push/PushBall-v1",
    entry_point="Envs:PushBallEnv1",
)