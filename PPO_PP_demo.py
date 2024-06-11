import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from PolePosition.envs.pole_position_env import PolePositionEnv

# Define a custom environment with larger window size for better visibility
def custom_env():
    return PolePositionEnv(render_mode='human', obs_type='features')

# Parallel environments for evaluation
vec_env = make_vec_env(custom_env, n_envs=1)

model = PPO.load("PPO_PolePosition")

obs = vec_env.reset()

# Main loop
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)

    # Render each environment
    for i in range(vec_env.num_envs):
        vec_env.envs[i].render()
        vec_env.envs[i].clock.tick(vec_env.envs[i].metadata['render_fps'])
