import gymnasium as gym
from stable_baselines3 import DQN
from PolePosition.envs.pole_position_env import PolePositionEnv

env = gym.make("PolePosition-features-v0", render_mode="human")

model = DQN.load("DQN_PolePosition", env=env)

# Test the agent
obs = env.reset()
obs = obs[0]  # Extract the observation if a tuple is returned

while True:
    action, _states = model.predict(obs, deterministic=True)
    # Step the environment and handle the five return values
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    env.clock.tick(env.metadata['render_fps'])
    if terminated or truncated:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the observation if a tuple is returned
