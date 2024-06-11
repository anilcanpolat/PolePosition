import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Step 1: Train an RL Agent on CartPole-v1
def train_ppo_agent():
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_cartpole")
    env.close()

    return model

# Step 2: Create and evaluate a random policy
def random_policy(env):
    return env.action_space.sample()

def evaluate_random_policy(env, num_episodes=100):
    total_rewards = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = random_policy(env)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    mean_sum_rewards = np.mean(total_rewards)
    return mean_sum_rewards

# Main execution
if __name__ == "__main__":
    # Train the PPO agent
    ppo_model = train_ppo_agent()
    
    # Load the CartPole-v1 environment and wrap it with Monitor
    env = Monitor(gym.make('CartPole-v1'))
    
    # Evaluate the trained PPO agent
    mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=100)
    print(f"Mean reward for PPO agent: {mean_reward} +/- {std_reward}")
    
    # Evaluate the random policy
    mean_random_reward = evaluate_random_policy(env, num_episodes=100)
    print(f"Mean sum of rewards over 100 episodes with random policy: {mean_random_reward}")

    # Load the saved PPO model
    ppo_model = PPO.load("ppo_cartpole")

    # Render the game using the trained PPO agent
    obs, _ = env.reset()
    while True:
        action, _states = ppo_model.predict(obs)
        obs, rewards, dones, _, _ = env.step(action)
        env.render()
        if dones:
            break

    env.close()
