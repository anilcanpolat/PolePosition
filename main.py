import pygame
import gymnasium as gym
from gymnasium.utils.play import play
from PolePosition.envs.pole_position_env import *

# Ensure the custom environment is registered
from PolePosition import *  # Make sure to import the registration script

# Create the environment using gym.make
env = gym.make("PolePosition-features-v0", render_mode="rgb_array")
# Start the game
# play(env, keys_to_action=keys_to_action)

# Without the .gym.play.play for higher resolution and render_mode="human"
def custom_play(env):
    
    env.reset()
    running = True
    final_reward = 0
    while running:
        
        env.render()
        keys = pygame.key.get_pressed()
        action = 4  # Assume 'do nothing' action

        if keys[pygame.K_UP]:
            action = 2  # Accelerate
        elif keys[pygame.K_DOWN]:
            action = 3  # Brake
        elif keys[pygame.K_RIGHT]:
            action = 0  # Turn left
        elif keys[pygame.K_LEFT]:
            action = 1  # Turn right

        _, reward, terminated, truncated, _ = env.step(action)
        final_reward += reward
        if terminated or truncated:
            print("Game Over: Resetting environment.")
            print("Final Reward: ", final_reward)
            final_reward = 0
            env.reset()

        env.clock.tick(env.metadata['render_fps'])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()

env = gym.make("PolePosition-pixels-v0", render_mode="human")
custom_play(env)