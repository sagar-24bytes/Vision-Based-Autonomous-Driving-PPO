import os
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


# ============================
# Configuration
# ============================
ENV_ID = "CarRacing-v2"
TOTAL_TIMESTEPS = 500_000
SEED = 42

MODELS_DIR = "models"
LOGS_DIR = "logs"


# ============================
# Directory Setup
# ============================
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ============================
# Environment Factory
# ============================
def make_env(seed: int = 0):
    def _init():
        env = gym.make(ENV_ID, continuous=True)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ============================
# Main Training Logic
# ============================
def main():
    set_random_seed(SEED)

    # Select device explicitly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Vectorized environment
    env = DummyVecEnv([make_env(SEED + i) for i in range(4)])
    # PPO model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        device=device
    )

    # Checkpoint callback (every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODELS_DIR,
        name_prefix="ppo_carracing"
    )

    try:
        # Train
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=1
        )
    finally:
        env.close()

    # Final model save
    model.save(os.path.join(MODELS_DIR, "ppo_carracing_500k_v2"))


if __name__ == "__main__":
    main()