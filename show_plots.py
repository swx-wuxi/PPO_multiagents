import pandas as pd
import matplotlib.pyplot as plt

# 1 读取奖励函数的比较曲线
# 读取无 RRT 数据
df_no_rrt = pd.read_csv("reward_without_rrt_1q1.csv")

# 读取有 RRT 数据
df_rrt = pd.read_csv("reward_with_rrt_1q1.csv")

# 读取DDPG 无RRT 的数据
ddpg_norrt = pd.read_csv("DDPG_reward_without_rrt_1q1.csv")

# 读取DDPG 有RRT 的数据
ddpg_rrt = pd.read_csv("DDPG_reward_with_rrt_1q1.csv")

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(df_rrt["Timesteps"], df_rrt["Reward"], marker='s', linestyle='--', color='r', label="SAC With RRT ")
plt.plot(df_no_rrt["Timesteps"], df_no_rrt["Reward"], marker='o', linestyle='--', color='b', label="SAC Without RRT")
plt.plot(ddpg_rrt["Timesteps"], ddpg_rrt["Reward"], marker='*', linestyle='--', color='g', label="DDPG With RRT ")
plt.plot(ddpg_norrt["Timesteps"], ddpg_norrt["Reward"], marker='.', linestyle='--', color='y', label="DDPG Without RRT ")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Comparison of Reward With and Without RRT")
plt.legend()
plt.grid(True)
plt.show()

# 2 读取成功率的比较曲线，完结！
# 读取无 RRT 数据
df_no_rrt = pd.read_csv("success_rates_without_rrt_1q1.csv")

# 读取有 RRT 数据
df_rrt = pd.read_csv("success_rates_with_rrt_1q1.csv")

# 读取DDPG 无RRT 的数据
ddpg_norrt = pd.read_csv("DDPG_success_rates_without_rrt_1q1.csv")

# 读取DDPG 有RRT 的数据
ddpg_rrt = pd.read_csv("DDPG_success_rates_with_rrt_1q1.csv")

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(df_rrt["Timesteps"], df_rrt["Success Rate"], marker='s', linestyle='--', color='r', label="SAC With RRT")
plt.plot(df_no_rrt["Timesteps"], df_no_rrt["Success Rate"], marker='o', linestyle='--', color='b', label="SAC Without RRT")
plt.plot(ddpg_rrt["Timesteps"], ddpg_rrt["Success Rate"], marker='*', linestyle='--', color='g', label="DDPG With RRT")
plt.plot(ddpg_norrt["Timesteps"], ddpg_norrt["Success Rate"], marker='.', linestyle='--', color='y', label="DDPG Without RRT")
plt.xlabel("Timesteps")
plt.ylabel("Success Rate (%)")
plt.title("Comparison of Success Rate With and Without RRT (1st Octant)")
plt.legend()
plt.grid(True)
plt.show()
