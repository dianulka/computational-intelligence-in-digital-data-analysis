from gymnasium.envs.registration import register

register(
    id="SimpleMaze-v0",
    entry_point="gymnasium_env.envs:SimpleMazeEnv",
)
