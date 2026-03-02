import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(random_rewards, ppo_rewards=None):
    random_rewards = np.array(random_rewards)

    plt.figure(figsize=(10, 6))

    plt.plot(random_rewards, label="Random Agent", marker="o")

    if ppo_rewards is not None:
        ppo_rewards = np.array(ppo_rewards)
        plt.plot(ppo_rewards, label="PPO Agent", marker="o")

    plt.title("Agent Performance Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    plt.savefig("results_plot.png")
    plt.show()

    print("Plot saved as results_plot.png")


if __name__ == "__main__":
    # Example random rewards (replace with your actual values)
    random_rewards = [-29.33, -33.99, -42.69, -36.31, -34.64,
                      -31.74, -27.01, -38.84, -31.27, -32.66]

    plot_rewards(random_rewards)