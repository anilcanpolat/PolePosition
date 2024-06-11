import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
from PolePosition.envs.pole_position_env import PolePositionEnv

# Create the environment and wrap it with Monitor
env = gym.make("PolePosition-features-v0")
env = Monitor(env, filename=None, allow_early_resets=True)

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.10, exploration_final_eps=0.05)

# Train the agent
model.learn(total_timesteps=10000, log_interval=4)

# Save the agent    
model.save("DQN_PolePosition_Re_De")

# Extract the Monitor log
monitor_log = env.get_episode_rewards()
episode_lengths = env.get_episode_lengths()

# Create a DataFrame
data = pd.DataFrame({
    'episode_rewards': monitor_log,
    'episode_lengths': episode_lengths
})

# Calculate rolling mean for smoother curve
data['rewards_rolling_mean'] = data['episode_rewards'].rolling(window=100).mean()

# Plot the learning curve
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['rewards_rolling_mean'])
plt.xlabel('Episode')
plt.ylabel('Episode Reward (Rolling Mean)')
plt.title('Learning Curve')
plt.show()
