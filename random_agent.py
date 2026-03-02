import gymnasium as gym
import numpy as np


def run_random_agent(episodes=10):
    env = gym.make("CarRacing-v3", render_mode=None)

    rewards = []

    for episode in range(episodes):
        observation, info = env.reset(seed=42 + episode)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    env.close()

    rewards = np.array(rewards)

    print("\n===== RANDOM AGENT PERFORMANCE =====")
    print(f"Average Reward: {rewards.mean():.2f}")
    print(f"Standard Deviation: {rewards.std():.2f}")
    print(f"Maximum Reward: {rewards.max():.2f}")
    print(f"Minimum Reward: {rewards.min():.2f}")


if __name__ == "__main__":
    run_random_agent()