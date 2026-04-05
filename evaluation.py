import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from env import TwoArmCoopEnv

from stable_baselines3 import PPO
from env import TwoArmCoopEnv
import numpy as np

model = PPO.load("models/ppo_two_arm_best_20260405_021052.zip")
env = TwoArmCoopEnv(render=False)
final_d1 = []
final_d2 = []

n_eval_episodes = 100
success_count = 0
collision_count = 0
ep_lengths = []

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

    if info.get("is_successful", False):
        success_count += 1
    if info.get("collision", False):
        collision_count += 1
    
    final_d1.append(info.get("d1", np.nan))
    final_d2.append(info.get("d2", np.nan))

print(success_count)
print("Success rate:", success_count / n_eval_episodes)
print("Collision rate:", collision_count / n_eval_episodes)
print("Average episode length:", np.mean(ep_lengths))
# print(final_d1)
# print(final_d2)