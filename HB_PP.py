import pygame
import gymnasium as gym
from PolePosition.envs.pole_position_env import *

# Create the environment using gym.make
env = gym.make("PolePosition-features-v0", render_mode="human")

def custom_play(env, num_episodes=2):
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the observation if a tuple is returned
        done = False
        episode_reward = 0

        while not done:
            env.render()
            keys = pygame.key.get_pressed()
            action = 4  # Assume 'do nothing' action

            if keys[pygame.K_UP]:
                action = 2  # Accelerate
            elif keys[pygame.K_DOWN]:
                action = 3  # Brake
            elif keys[pygame.K_RIGHT]:
                action = 0  # Turn right
            elif keys[pygame.K_LEFT]:
                action = 1  # Turn left

            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward = reward

            if terminated or truncated:
                done = True

            env.clock.tick(env.metadata['render_fps'])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.close()
                    return

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}")

    mean_reward = sum(total_rewards) / num_episodes
    print(f"Mean reward over {num_episodes} episodes: {mean_reward}")

    return mean_reward

# Run the game to create a human baseline
mean_reward = custom_play(env)
env.close()
