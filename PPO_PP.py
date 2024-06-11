import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from PolePosition.envs.pole_position_env import PolePositionEnv

# Parallel environments
vec_env = make_vec_env("PolePosition-features-v0", n_envs=1 )

# Instantiate the agent
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=500000)

# Save the agent 
# model.save("PPO_PolePosition")