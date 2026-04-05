import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from env import TwoArmCoopEnv

from stable_baselines3 import PPO
from env import TwoArmCoopEnv
import numpy as np

model = PPO.load("models/ppo_two_arm_best_20260405_021052.zip")     # ent_cof = 0
# model = PPO.load("models/ppo_two_arm_best_20260405_215143.zip")   # ent_cof = 0.01

env = TwoArmCoopEnv(render=False)
final_d1 = []
final_d2 = []

n_eval_episodes = 100
success_count = 0
collision_count = 0
ep_lengths = []

best_episode_data = None
best_score = None
for ep in range(n_eval_episodes):
    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    ep_lengths.append(step_count)
    
    success = info.get("is_successful", False)
    collision = info.get("collision", False)

    if success:
        success_count += 1
    if collision:
        collision_count += 1
    
    d1 = info.get("d1", np.nan)
    d2 = info.get("d2", np.nan)
    final_d1.append(info.get("d1", np.nan))
    final_d2.append(info.get("d2", np.nan))
    
    traj_data = env.get_current_trajectory()

    # total_d = d1 + d2
    worst_d = max(d1, d2)
    best_d = min(d1, d2)

    success_flag = 1 if success else 0
    score = (-worst_d, -best_d, -step_count)

    if success and (best_score is None or (score > best_score)):
        best_score = score
        best_episode_data = traj_data

print("Success rate:", success_count / n_eval_episodes)
print("Collision rate:", collision_count / n_eval_episodes)
print("Average episode length:", np.mean(ep_lengths))

if best_episode_data is not None:
    env.visualize_trajectory(best_episode_data)