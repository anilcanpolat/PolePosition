import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from PolePosition.envs.pole_position_env import PolePositionEnv

# Parallel environments
vec_env = make_vec_env("PolePosition-features-v0", n_envs=1)

dqn_params = {
    'learning_rate': 1e-3,
    # 'buffer_size': 200000,
    # 'learning_starts': 1000,
    # 'batch_size': 32,
    # 'tau': 1.0,
    # 'gamma': 0.99,
    # 'train_freq': 4,
    # 'gradient_steps': 1,
    # 'exploration_initial_eps': 1.0,  # Increased initial exploration rate
    # 'exploration_final_eps': 0.1,  # Less aggressive final exploration rate
    # 'exploration_fraction': 0.5,  # Spread the exploration over a longer fraction of training
    # 'target_update_interval': 1000,
    # 'policy_kwargs': dict(net_arch=[256, 256]),
    # 'verbose': 1,
}

# Instantiate the agent
model = DQN('MlpPolicy', vec_env, **dqn_params)

# Train the agent
model.learn(total_timesteps=500000)

# Save the agent
model.save("DQN_PolePosition_new")