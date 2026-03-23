import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import TwoArmCoopEnv, EpisodeLogger

LOG_DIR = "./logs_ppo_two_arm"
MODEL_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = TwoArmCoopEnv(render=True)
check_env(env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

total_timesteps = 300000
check_freq = 10000

success_rates = []
timesteps = []
mean_rewards = []
mean_episode_lengths = []

total_success = 0
total_episode = 0

for step in range(0, total_timesteps, check_freq):
    logger = EpisodeLogger()
    model.learn(
        total_timesteps=check_freq,
        reset_num_timesteps=False,
        callback=logger,
        tb_log_name="ppo_two_arm"
    )

    total_success += logger.success_count
    total_episode += logger.episode_count

    success_rate = (total_success / total_episode) * 100 if total_episode > 0 else 0.0
    mean_reward = np.mean(logger.episode_rewards) if logger.episode_rewards else 0.0
    mean_ep_len = np.mean(logger.episode_lengths) if logger.episode_lengths else 0.0

    success_rates.append(success_rate)
    timesteps.append(step + check_freq)
    mean_rewards.append(mean_reward)
    mean_episode_lengths.append(mean_ep_len)

    print(
        f"[{step + check_freq}/{total_timesteps}] "
        f"success_rate={success_rate:.2f}% | "
        f"mean_reward={mean_reward:.2f} | "
        f"mean_ep_len={mean_ep_len:.2f}"
    )

model.save(os.path.join(MODEL_DIR, "ppo_two_arm_final"))

pd.DataFrame({
    "Timesteps": timesteps,
    "Success Rate": success_rates
}).to_csv("success_rates_two_arm_ppo.csv", index=False)

pd.DataFrame({
    "Timesteps": timesteps,
    "Reward": mean_rewards
}).to_csv("reward_two_arm_ppo.csv", index=False)

pd.DataFrame({
    "Timesteps": timesteps,
    "Episode Length": mean_episode_lengths
}).to_csv("episode_len_two_arm_ppo.csv", index=False)

env.close()