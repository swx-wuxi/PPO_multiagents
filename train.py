import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import TwoArmCoopEnv, EpisodeLogger
from datetime import datetime

LOG_DIR = "./logs"
MODEL_DIR = "./models"
CSV_DIR = "./csv_logs"
FIG_DIR = "./plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

env = TwoArmCoopEnv(render=True)
check_env(env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    seed = 42,
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

total_timesteps = 500000
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
# Save the model and record the logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(os.path.join(MODEL_DIR, f"ppo_two_arm_final_{timestamp}"))

pd.DataFrame({
    "Timesteps": timesteps,
    "Success Rate": success_rates
}).to_csv(os.path.join(CSV_DIR,f"success_rate_curve_{timestamp}.csv"), index=False)

pd.DataFrame({
    "Timesteps": timesteps,
    "Reward": mean_rewards
}).to_csv(os.path.join(CSV_DIR,f"reward_{timestamp}.csv"), index=False)

pd.DataFrame({
    "Timesteps": timesteps,
    "Episode Length": mean_episode_lengths
}).to_csv(os.path.join(CSV_DIR,f"episode_len_{timestamp}.csv"),index=False)

# 画成功率曲线
plt.figure(figsize=(10, 6))
plt.plot(timesteps, success_rates, marker="o")
plt.xlabel("Timesteps")
plt.ylabel("Success Rate (%)")
plt.title("Training Success Rate Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"success_rate_curve_{timestamp}.png"), dpi=300)
plt.show()

# 画平均奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards, marker="o")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Training Reward Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"reward_curve_{timestamp}.png"), dpi=300)
plt.show()

env.close()