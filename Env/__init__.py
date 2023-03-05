from gym.envs.registration import register

register(
    id="tactile_push/push_ball-v0",
    entry_point="Env.push_ball_env:PushBallEnv0",
)