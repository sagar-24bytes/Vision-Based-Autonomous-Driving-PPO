import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
np.core.numeric = np.core

MODEL_PATH = "models/ppo_carracing_500k_v2"

env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

model = PPO.load(MODEL_PATH, device="cpu", print_system_info=True)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset()