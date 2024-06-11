import gymnasium as gym
import numpy as np
from PolePosition.envs.pole_position_env import PolePositionEnv

def random_baseline(env_name, num_episodes):
    env = gym.make(env_name, render_mode='human')
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated, truncated = False, False

        while not terminated and not truncated:
            action = env.action_space.sample()  # Take a random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            env.render()
            # env.clock.tick(env.metadata['render_fps'])

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}")

    env.close()
    
    mean_reward = np.mean(total_rewards)
    print(f"Mean Reward over {num_episodes} episodes: {mean_reward}")

    return mean_reward

# Run the random baseline
env_name = "PolePosition-features-v0"
num_episodes = 10
mean_reward = random_baseline(env_name, num_episodes)
