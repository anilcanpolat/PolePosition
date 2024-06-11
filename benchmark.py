from PolePosition.envs.pole_position_env import PolePositionEnv
import time
import gymnasium as gym
import numpy as np

def benchmark_fps(env_name, num_steps=1000, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    obs = env.reset()
    
    start_time = time.time()
    
    for _ in range(num_steps):
        action = env.action_space.sample()  # Take random actions
        obs, reward, done, truncated, info = env.step(action)
        if render_mode:
            env.render()
        if done or truncated:
            env.reset()
    
    end_time = time.time()
    env.close()
    
    elapsed_time = end_time - start_time
    fps = num_steps / elapsed_time
    print(f"Environment: {env_name}, Render mode: {render_mode}, Steps: {num_steps}, FPS: {fps:.2f}")

if __name__ == "__main__":
    benchmark_fps('PolePosition-features-v0', num_steps=1000, render_mode='human')  # Human render mode
    benchmark_fps('PolePosition-features-v0', num_steps=1000, render_mode='rgb_array')  # RGB array render mode
    benchmark_fps('PolePosition-features-v0', num_steps=1000, render_mode=None)  # No rendering
    benchmark_fps('PolePosition-pixels-v0', num_steps=1000, render_mode='human')  # Human render mode
    benchmark_fps('PolePosition-pixels-v0', num_steps=1000, render_mode='rgb_array')  # RGB array render mode
    benchmark_fps('PolePosition-pixels-v0', num_steps=1000, render_mode=None)  # No rendering