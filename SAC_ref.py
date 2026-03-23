import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import pybullet_data
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

# 修改为使用PyBullet内置的KUKA机械臂模型
URDF_PATH = "kuka_iiwa/model.urdf"

# 机械臂参数调整
JOINT_LIMITS = {
    'low': np.array([-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05]),
    'high': np.array([2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05])
}

# 初始和目标关节角度（适配7自由度机械臂）
q_start = np.zeros(7)
target_pos = np.array([-0.6, 0.5, 0.8])

# 动态障碍物参数
num_obstacles = 2  # 增加一个障碍物
obstacle_radius = 0.12
max_time = 50

# 奖励函数参数
beta = 0.01
sigma1 = 1
sigma2 = 1
    
# 障碍物运动参数
obstacle_start_pos_1 = np.array([-2, -5, 1.2])
obstacle_end_pos_1 = np.array([4, 8, 1.2])
obstacle_start_pos_2 = np.array([2 , 7 , 1.2])  # 第二个障碍物起始位置
obstacle_end_pos_2 = np.array([-0.5, -5, 1.2])  # 第二个障碍物结束位置
obstacle_speed = 0.02

# 放宽边界范围的小值
EPSILON = 1e-3

class ImprovedSafeRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.obs_dim = 7 + 1 + num_obstacles * 3
        self.action_dim = 7

        # 调整观测空间的边界
        obstacle_low = np.minimum(obstacle_start_pos_1, obstacle_end_pos_1) - EPSILON
        obstacle_high = np.maximum(obstacle_start_pos_1, obstacle_end_pos_1) + EPSILON
        self.observation_space = Box(
            low=np.concatenate([JOINT_LIMITS['low'], [0], obstacle_low, obstacle_low]),
            high=np.concatenate([JOINT_LIMITS['high'], [1], obstacle_high, obstacle_high]),
            dtype=np.float32
        )

        self.action_space = Box(
            low=-1.0 * np.ones(self.action_dim),
            high=1.0 * np.ones(self.action_dim),
            dtype=np.float32
        )

        # 物理引擎初始化
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.robot = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

        # 创建动态障碍物
        self.obstacles = []
        self._create_obstacles()

        # 初始化全局路径规划器（RRT*）
        self.global_path = self._generate_global_path()

        self.reset()

    def _create_obstacles(self):
        # 创建第一个障碍物
        obstacle_1 = {
            'id': p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius),
            'pos': obstacle_start_pos_1.copy(),
        }
        obstacle_1['body'] = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=obstacle_1['id'],
            basePosition=obstacle_start_pos_1
        )
        self.obstacles.append(obstacle_1)

        # 创建第二个反向运动的障碍物
        obstacle_2 = {
            'id': p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius),
            'pos': obstacle_start_pos_2.copy(),
        }
        obstacle_2['body'] = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=obstacle_2['id'],
            basePosition=obstacle_start_pos_2
        )
        self.obstacles.append(obstacle_2)

    def reset(self, seed=None):
        for j in range(p.getNumJoints(self.robot)):
            p.resetJointState(self.robot, j, q_start[j])

        for obstacle in self.obstacles:
            new_pos = obstacle['pos'].copy()
            p.resetBasePositionAndOrientation(
                obstacle['body'],
                new_pos,
                [0, 0, 0, 1]
            )

        self.current_time = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot, j)[0] for j in range(p.getNumJoints(self.robot))]
        obstacle_pos = [p.getBasePositionAndOrientation(o['body'])[0] for o in self.obstacles]
        return np.concatenate([
            joint_states,
            [self.current_time / max_time],
            np.array(obstacle_pos).flatten()
        ], dtype=np.float32)

    def step(self, action):
        dt = 0.1
        self.current_time += dt

        # 更新障碍物位置（两个障碍物）
        t = (self.current_time / max_time) * 2
        if t > 1.0:
            t = 2.0 - t
        new_pos_1 = obstacle_start_pos_1 + t * (obstacle_end_pos_1 - obstacle_start_pos_1)
        new_pos_2 = obstacle_start_pos_2 + t * (obstacle_end_pos_2 - obstacle_start_pos_2)

        for obstacle, new_pos in zip(self.obstacles, [new_pos_1, new_pos_2]):
            obstacle['pos'] = new_pos
            p.resetBasePositionAndOrientation(
                obstacle['body'],
                obstacle['pos'],
                [0, 0, 0, 1]
            )

        # 应用动作（限制在安全范围内）
        action = np.clip(action, -1, 1) * 0.2
        action_clapped = np.clip(action, -1, 1) * 0.2
        action_clapped += np.random.normal(0, 0.1, size=action_clapped.shape)
        new_joint_pos = np.array([p.getJointState(self.robot, j)[0] for j in range(self.action_dim)]) + action_clapped

        for j in range(self.action_dim):
            p.setJointMotorControl2(
                self.robot, j,
                p.POSITION_CONTROL,
                targetPosition=new_joint_pos[j],
                force=500
            )
        p.stepSimulation()

        # 计算奖励
        reward = 0
        terminated = False
        truncated  = False
        info = {'is_successful': False}  # 新增字段
        ee_pos = p.getLinkState(self.robot, 6)[0]
        D_PT = np.linalg.norm(ee_pos - target_pos)
        if self._check_collision():
            reward = -200 - 10 * np.exp(-D_PT)  # 降低碰撞惩罚
            print("Collision failure!")
            terminated = True
            truncated = True
        else:
            # 奖励函数
            R_position = np.exp(-D_PT / sigma1)
            R_direction = np.clip(np.dot((target_pos - ee_pos), p.getLinkState(self.robot, 6)[2][:3]), -2, 2)  # 增强目标方向奖励
            R_smoothness = -np.linalg.norm(action_clapped) / 5.0  # 增强平滑性奖励
            R_posture = -np.linalg.norm(new_joint_pos) / 100.0

            # 增加距离奖励
            obstacle_pos_1 = self.obstacles[0]['pos']  # 获取第一个障碍物的位置
            obstacle_pos_2 = self.obstacles[1]['pos']  # 获取第二个障碍物的位置
            # R_distance_1 = -np.linalg.norm(ee_pos - obstacle_pos_1) / 10.0  # 鼓励远离第一个障碍物
            # R_distance_2 = -np.linalg.norm(ee_pos - obstacle_pos_2) / 10.0  # 鼓励远离第二个障碍物
            R_distance_1 = -2.0 * np.exp(-np.linalg.norm(ee_pos - obstacle_pos_1))  # 平滑避障
            R_distance_2 = -2.0 * np.exp(-np.linalg.norm(ee_pos - obstacle_pos_2))
            # 增加探索激励
            R_explore = np.random.normal(0, 0.1)

            # 接近目标奖励更加平滑
            R_goal = 200 * np.exp(-10 * D_PT)
            if D_PT < 0.1:
                R_goal += 500
                terminated = True
                info['is_successful'] = True
                print("Reach the target successfully!")
                return self._get_obs(), reward, terminated, True, info
            reward = R_position + R_direction + R_smoothness + R_posture + R_distance_1 + R_distance_2 + R_explore+R_goal

            # 超时判断
            if self.current_time >= max_time:
                terminated = True
                truncated = True

        return self._get_obs(), reward, terminated, truncated, info
            
           
    def _check_collision(self):
        for obstacle in self.obstacles:
            closest = p.getClosestPoints(
                self.robot, obstacle['body'],
                distance=obstacle_radius + 0.01
            )
            if len(closest) > 0:
                return True
        return False

    def _generate_global_path(self):
        # 使用 RRT* 生成全局路径（简化版）
        # 这里可以调用外部的 RRT* 实现，返回路径点
        return np.array([
            [0, 0, 1.2],
            [-0.2, 0.1, 1.1],
            [-0.4, 0.35, 0.9],
            [-0.6, 0.5, 0.8]
        ])

    def close(self):
        p.disconnect()

# 环境验证
env = ImprovedSafeRobotEnv()
check_env(env)

# 添加训练回调类以记录数据
class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeLogger, self).__init__(verbose)
        self.rewards = []  # 记录每一步的奖励
        self.success_count = 0  # 成功到达目标点的次数，由于每次都会清零，会出现一些问题
        self.actions = []  # 记录每一步的动作
        self.ee_positions = []  # 记录末端执行器的位置
        self.episode_rewards = []  # 记录每个episode的总奖励
        self.episode_lengths = []  # 记录每个 episode 的长度
        self.current_episode_length = 0  # 当前 episode 的步数, 要归零
        self.episode_count = 0    #记录总的集的个数
    def _on_step(self) -> bool:
        # 从 info 中获取奖励值和成功标志
        infos = self.locals['infos']
        reward = self.locals['rewards']
        self.current_episode_length += 1 
        
        #1 统计步长
        done = self.locals['dones']  # 终止标志
        if done:
            self.episode_lengths.append(self.current_episode_length)  # 记录步数
            self.current_episode_length = 0  # 重置步数计数器
        
        # 2 统计成功率
        for info, r in zip(infos, reward):
            self.rewards.append(r)
            if info.get('is_successful', False):
                self.success_count += 1
                print("每个间隔成功的次数：",self.success_count)
        # # 记录动作和末端执行器位置
        # actions = self.locals.get('actions')
        # if actions is not None:
        #     # 假设 actions 是 numpy 数组
        #     if isinstance(actions, np.ndarray):
        #         self.actions.extend(actions)
        #     else:  # 如果 actions 是 PyTorch 张量
        #         self.actions.extend(actions.cpu().numpy())
        
        # 3 统计总的奖励函数
        ee_pos = np.array([p.getLinkState(env.robot, 6)[0]])
        self.ee_positions.append(ee_pos)
    
        if done: 
            self.episode_count += 1
            self.episode_rewards.append(sum(reward))  # 记录当前episode的总奖励
        return True

# 训练配置
model = SAC(
    'MlpPolicy', env,
    learning_rate=5e-5,
    buffer_size=int(1e7),
    batch_size=512,
    gamma=0.99,
    tau=0.05,
    ent_coef=0.1,
    verbose=1
)

# 增加训练时间

# model.learn(total_timesteps=100000, callback=logger)  # 增加训练时间

# 1 绘制探索集长度的变化
def plot_exploration_length(logger):
    plt.figure(figsize=(10, 6))
    plt.plot(logger.episode_lengths, label="Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Exploration Length Over Episodes")
    plt.legend()
    plt.show()

# 2 可视化训练奖励曲线
def plot_training_reward(logger):
    plt.figure(figsize=(10, 6))
    plt.plot(logger.rewards, label="Episode Reward")
    plt.xlabel("Episode Steps")
    plt.ylabel("Reward")
    plt.title("Training Reward Progress")
    plt.legend()
    plt.show()

# 3 绘制到达目标点的成功率
def plot_success_rate(logger):
    # 计算总episode数
    total_episodes = logger.episode_count
    # 成功率
    success_rate = (logger.success_count / total_episodes) * 100 if total_episodes > 0 else 0
    print("成功率：", success_rate)
    failed_rate = 100 - success_rate

    labels = ['Success', 'Failure']
    sizes = [success_rate, failed_rate]
    colors = ['lightgreen', 'lightcoral']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title("Success Rate of Reaching Target")
    plt.show()

# plot_success_rate(logger)

# 4 可视化机械臂轨迹规划效果并检测碰撞
def visualize_trajectory(env):
    obs, _ = env.reset()
    path = []
    obstacle_pos1 = []
    obstacle_pos2 = []

    for step in range(int(max_time / 0.1)):  #碰撞检测每一步step都会去做的，不需要单独写
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, info = env.step(action)  #self._get_obs(), reward, terminated, False, info
        ee_pos = p.getLinkState(env.robot, 6)[0]
        print(f"step {step}: d = {np.linalg.norm(np.array(ee_pos) - np.array(target_pos))} ") 
        #打印每一步EE距离目标点的距离

        time = step * 0.1  # 计算当前时间
        path.append((ee_pos, time))  # 同时记录位置和时间
        obstacle_pos1.append(env.obstacles[0]['pos'])
        obstacle_pos2.append(env.obstacles[1]['pos'])

        if np.linalg.norm(np.array(ee_pos) - np.array(target_pos)) <= 0.1  and  info['is_successful']:
            print("Successful trajectory!")
            break
        
        if terminated:
            print("Trjactory planning failed!")
            break

    obstacle_pos1 = np.array(obstacle_pos1)
    obstacle_pos2 = np.array(obstacle_pos2)

    # 绘制轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取位置和时间信息
    positions = [item[0] for item in path]
    times = [item[1] for item in path]
    positions = np.array(positions)
    
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory")
    ax.plot(obstacle_pos1[:, 0], obstacle_pos1[:, 1], obstacle_pos1[:, 2], 'r--', label="Obstacle1 Path")
    ax.plot(obstacle_pos2[:, 0], obstacle_pos2[:, 1], obstacle_pos2[:, 2], 'g--', label="Obstacle2 Path")
        
    # 标记起点和终点
    start_pos = p.getLinkState(env.robot, 6)[0]
    ax.scatter(positions[0][0], positions[0][1], positions[0][2], c='yellow', label="ee_Start", s=100, marker="o") # 起点
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='green', label="ee_end", s=100, marker="o") # 末端执行器终点
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='red', label="Target", s=100, marker="*")
    # 用文字标记我们的实验结果，即末端执行器与目标点最终距离。
    distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
    ax.text(
        (positions[0][0] + target_pos[0]) / 2,  # X 方向取中点
        (positions[0][1] + target_pos[1]) / 2,  # Y 方向取中点
        (positions[0][2] + target_pos[2]) / 2,  # Z 方向取中点
        f"Distance: {distance:.3f} ",  # 显示两点间的距离
        color="black",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6)  # 添加背景提高可读性
    )    
    # 绘制坐标轴
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Z position")
    ax.set_title("Improved SAC Robot Arm Motion Planning")
    ax.legend()
        
    plt.show()

#--------------------------------上面是所有定义的函数、模型、实体类-------------------------------

#*********************************************************************************************************

#--------------------------------主程序在下面，每训练五万次打印一下结果---------------- -----------  

total_timesteps = 150000
check_freq = 5000
success_rates = []  # 记录成功率
timesteps = []  # 记录时间步数
total_success = 0  #记录累计成功次数
total_episode = 0  #记录累计训练集数
total_reward = 0
total_episode_length = 0
mean_rewards = [] #画图所需要的平均奖励函数的集合
mean_episode_lengths = []  #画图所需要的平均每集长度的集合

# *****训练核心程序*****
for step in range(0, total_timesteps, check_freq):
    logger = EpisodeLogger()
    model.learn(total_timesteps=check_freq, reset_num_timesteps=False, callback=logger)
    total_success+=logger.success_count    #计算累计成功的次数
    total_episode+=logger.episode_count    #计算累计的集数
    print(f"总成功次数:{total_success}, 总集数：{total_episode}")  #调试看看有没有问题
    # 1 计算当前成功率（累计就可以），画折线图
    success_rate = (total_success / total_episode) * 100 if total_episode > 0 else 0
    print(success_rate)

    success_rates.append(success_rate)   # 成功率的图像最终需要绘制
    timesteps.append(step + check_freq)  # 记录当前步数

    # 2 画出整个奖励函数图像的曲线 ,仿照成功率，取每一个间隔平均奖励函数，作折线图
    # plot_training_reward(logger)
    # for reward in logger.episode_rewards:
    #      total_reward+=reward

    # mean_reward = total_reward / total_episode if total_episode>0 else 0
    # mean_rewards.append(mean_reward)     
    # # timesteps.append(step + check_freq)  # 记录当前步数

    # # 3 画出整个训练次数图像的曲线，仿照成功率，取每一个间隔平训练长度，作折线图
    # # plot_exploration_length(logger)
    # for length in logger.episode_lengths:
    #     total_episode_length+=length

    # mean_episode_length = total_episode_length / total_episode  if total_episode>0 else 0
    # mean_episode_lengths.append(mean_episode_length)
    # timesteps.append(step + check_freq)  # 记录当前步数
    
    # 4 这个逻辑就和训练日志一样，记录的是每一个episode里面的平均值，而不是累计
    current_total_reward = sum(logger.episode_rewards)  # 当前 batch 内 episode 的奖励总和
    current_total_length = sum(logger.episode_lengths)  # 当前 batch 内 episode 的长度总和
    current_episode_count = logger.episode_count  # 当前 batch 内 episode 的个数

    mean_reward = current_total_reward / current_episode_count if current_episode_count > 0 else 0
    print(mean_reward)
    mean_episode_length = current_total_length / current_episode_count if current_episode_count > 0 else 0
    mean_rewards.append(mean_reward)  
    mean_episode_lengths.append(mean_episode_length)
# 绘制成功率变化折线图
plt.figure(figsize=(10, 6))
plt.plot(timesteps, success_rates, marker='o', linestyle='-', color='b', label="Success Rate")
plt.xlabel("Timesteps")
plt.ylabel("Success Rate(%)")
plt.title("Success Rate Over Training Steps with RRT")
plt.legend()
plt.grid(True)
plt.show()

# 绘制平均奖励函数变化折线图
plt.figure(figsize=(10, 6))
plt.plot(timesteps,  mean_rewards, marker='o', linestyle='-', color='r', label="Mean reward")
plt.xlabel("Timesteps")
plt.ylabel("Mean reward")
plt.title("Reward functions with RRT")
plt.legend()
plt.grid(True)
plt.show()

# 最后一步，保存数据，以便于绘制对比图。
import pandas as pd

# 1 保存成功率图像
# 创建数据字典
data = {"Timesteps": timesteps, "Success Rate": success_rates}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv("success_rates_with_rrt.csv", index=False)

print("成功率数据已保存到 success_rates_with_rrt.csv")

# 2  保存成功率图像
# 创建数据字典
data = {"Timesteps": timesteps, "Reward": mean_rewards}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv("reward_with_rrt.csv", index=False)

print("奖励函数`数据已保存到 reward_with_rrt.csv")

# 3 轨迹可视化
visualize_trajectory(env)
env.close()

