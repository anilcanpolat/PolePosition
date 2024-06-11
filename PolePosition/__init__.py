import gymnasium as gym

print("Registering PolePosition environments...")  # Debug statement

gym.register(
    id='PolePosition-pixels-v0',
    entry_point='PolePosition.envs.pole_position_env:PolePositionEnv',
    kwargs={'obs_type': 'pixels', 'render_mode': 'rgb_array'},
    max_episode_steps=1000,
)

gym.register(
    id='PolePosition-features-v0',
    entry_point='PolePosition.envs.pole_position_env:PolePositionEnv',
    kwargs={'obs_type': 'features', 'render_mode': 'human'},
    max_episode_steps=1000,
)

print("PolePosition environments registered.")  # Debug statement