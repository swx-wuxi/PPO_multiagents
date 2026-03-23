import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
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
# b. for two arms
q_start_1 = np.zeros(7, dtype=np.float32)
q_start_2 = np.zeros(7, dtype=np.float32)

target_pos_1 = np.array([0.4, 0.55, 0.45], dtype=np.float32)
target_pos_2 = np.array([0.4, -0.15, 0.45], dtype=np.float32)
# target_pos_1 = np.array([0.4, 0.6, 0.45], dtype=np.float32)
# target_pos_2 = np.array([0.5 , -0.2 , 0.45], dtype=np.float32)

# 动态障碍物参数
num_obstacles = 2  
obstacle_radius = 0.08
max_time = 50
EE_LINK = 6
DT = 0.1

# 奖励函数参数
beta = 0.2
sigma1 = 1
sigma2 = 1
    
# Obstacle setup
obstacle_start_pos_1 = np.array([0 , -0.75, 0.6])
obstacle_start_pos_2 = np.array([0 , 0.75 , 0.6])  

# For dynamic obstacles, optional
# obstacle_end_pos_1 = np.array([1.3, 2, 0.7])
# obstacle_end_pos_2 = np.array([0.1, -1, 0.7])  # 第二个障碍物结束位置
# obstacle_speed = 0.01

# obstacle_mid_pos_1 = (obstacle_start_pos_1 + obstacle_end_pos_1) / 2  # 中间位置
# obstacle_mid_pos_2 = (obstacle_start_pos_2 + obstacle_end_pos_2) / 2  # 中间位置

# 放宽边界范围的小值
EPSILON = 1e-3

class TwoArmCoopEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=True):
        super().__init__()
        self.render_enabled = render
       
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.robot1_base = [-0.4,  0.35, 0.0]
        self.robot2_base = [-0.4, -0.35, 0.0]

        self.robot1 = p.loadURDF(URDF_PATH, self.robot1_base, useFixedBase=True)
        self.robot2 = p.loadURDF(URDF_PATH, self.robot2_base, useFixedBase=True)
        
        self.use_obstacles = True  # Choose whether to use obstacles

        if self.use_obstacles:
            self.obstacles = []
            self._create_obstacles()
            # 观测：
            # robot1 joints(7) + robot2 joints(7)
            # ee1(3) + ee2(3)
            # target1-ee1(3) + target2-ee2(3) (for dynamic obstacles)
            # time(1) + obstacles(2*3)
            self.obs_dim = 7 + 7 + 3 + 3  + 1 + num_obstacles * 3 + 6
        else:
            self.obs_dim = 7 + 7 + 3 + 3 + 1 + 2*3
        obs_low = np.full(self.obs_dim, -10.0, dtype=np.float32)
        obs_high = np.full(self.obs_dim, 10.0, dtype=np.float32)
        obs_low[-7] = 0.0   # normalized time
        obs_high[-7] = 1.0

        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 动作：两台机械臂各 7 维
        self.action_dim = 14
        self.action_space = Box(
            low=-1.0 * np.ones(self.action_dim, dtype=np.float32),
            high=1.0 * np.ones(self.action_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        #### Core : RRT
        self.current_time = 0.0
        self.global_path_1 = self._generate_global_path_single(-0.4,0.35,1.28,0.8,0.2,-0.83)
        self.global_path_2 = self._generate_global_path_single(-0.4,-0.35,1.28,0.8,0.2,-0.83)
        self.reset()

    def _create_obstacles(self):
        # 创建第一个障碍物
        obstacle_1 = {
            'id': p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius),
            'pos': obstacle_start_pos_1.copy(),
        }
        obstacle_1['body'] = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_1['id'],
            basePosition=obstacle_1["pos"]
        )
        self.obstacles.append(obstacle_1)

        # 创建第二个障碍物
        obstacle_2 = {
            'id': p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius),
            'pos': obstacle_start_pos_2.copy(),
        }
        obstacle_2['body'] = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_2['id'],
            basePosition=obstacle_2['pos']
        )
        self.obstacles.append(obstacle_2)

    def _reset_robot(self, robot_id, q):
        for j in range(7):
            p.resetJointState(robot_id, j, float(q[j]))

    def _get_joint_state(self, robot_id):
        return np.array([p.getJointState(robot_id, j)[0] for j in range(7)], dtype=np.float32)

    def _get_ee_pos(self, robot_id):
        return np.array(p.getLinkState(robot_id, EE_LINK)[0], dtype=np.float32)

    def _get_obs(self):
        q1 = self._get_joint_state(self.robot1)
        q2 = self._get_joint_state(self.robot2)
        ee1 = self._get_ee_pos(self.robot1)
        ee2 = self._get_ee_pos(self.robot2)
        
        # 区分有无障碍物好调试
        if self.use_obstacles:
            obs_pos = np.array([p.getBasePositionAndOrientation(o["body"])[0] for o in self.obstacles], dtype=np.float32).flatten()
            obs = np.concatenate([
                q1,
                q2,
                ee1,
                ee2,
                target_pos_1 - ee1,
                target_pos_2 - ee2,
                np.array([self.current_time / max_time], dtype=np.float32),
                obs_pos
            ]).astype(np.float32)
        else:
            obs = np.concatenate([
                q1,
                q2,
                ee1,
                ee2,
                target_pos_1 - ee1,
                target_pos_2 - ee2,
                np.array([self.current_time / max_time], dtype=np.float32),
            ]).astype(np.float32)  
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_robot(self.robot1, q_start_1)
        self._reset_robot(self.robot2, q_start_2)
        if self.use_obstacles:
            self.obstacles[0]["pos"] = obstacle_start_pos_1.copy()
            self.obstacles[1]["pos"] = obstacle_start_pos_2.copy()

            for obs in self.obstacles:
                p.resetBasePositionAndOrientation(obs["body"], obs["pos"], [0, 0, 0, 1])
        ee1 = self._get_ee_pos(self.robot1)
        ee2 = self._get_ee_pos(self.robot2)

        self.prev_d1 = np.linalg.norm(ee1 - target_pos_1)
        self.prev_d2 = np.linalg.norm(ee2 - target_pos_2)

        self.current_time = 0.0
        return self._get_obs(), {}

    # def _update_obstacles(self):
    #     t = np.sin((self.current_time / max_time) * np.pi)

    #     new_pos_1 = obstacle_mid_pos_1 + t * (obstacle_end_pos_1 - obstacle_mid_pos_1)
    #     new_pos_2 = obstacle_mid_pos_2 + t * (obstacle_end_pos_2 - obstacle_mid_pos_2)

    #     for obstacle, new_pos in zip(self.obstacles, [new_pos_1, new_pos_2]):
    #         obstacle["pos"] = new_pos
    #         p.resetBasePositionAndOrientation(obstacle["body"], new_pos, [0, 0, 0, 1])

    def _apply_action_to_robot(self, robot_id, action7):
        action7 = np.clip(action7, -1.0, 1.0) * 0.25
        q = self._get_joint_state(robot_id)
        new_q = np.clip(q + action7, JOINT_LIMITS["low"], JOINT_LIMITS["high"])

        for j in range(7):
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(new_q[j]),
                force=500
            )

    def _check_robot_obstacle_collision(self, robot_id):
        if not self.use_obstacles:
            return False
        
        for obstacle in self.obstacles:
            closest = p.getClosestPoints(robot_id, obstacle["body"], distance=obstacle_radius + 0.02)
            if len(closest) > 0:
                return True
        return False

    def _check_robot_robot_collision(self):
        closest = p.getClosestPoints(self.robot1, self.robot2, distance=0.03)
        return len(closest) > 0

    def _compute_reward(self, ee_pos, target_pos, action7, q):
        d = np.linalg.norm(ee_pos - target_pos)
       
        reward = 200.0 * np.exp(-8.0 * d)
        reward += -0.05 * np.linalg.norm(action7)
        reward += -0.01 * np.linalg.norm(q)
        if self.use_obstacles: 
            R_obs_1 = -2.0 * np.exp(-np.linalg.norm(ee_pos - obstacle_start_pos_1))
            R_obs_2 = -2.0 * np.exp(-np.linalg.norm(ee_pos - obstacle_start_pos_2))
            R_obs = R_obs_1 + R_obs_2
            reward += R_obs

        if d < 0.15:
            reward += 500.0

        return reward, d

    def step(self, action):
        # print("阮中乐真可爱")
        # ee1_pos = p.getLinkState(self.robot1, 6)[0]
        # ee2_pos = p.getLinkState(self.robot2, 6)[0]

        # print("EE1:", ee1_pos, "EE2:", ee2_pos)
        # input()
        action = np.asarray(action, dtype=np.float32)
        a1 = action[:7]
        a2 = action[7:]

        self.current_time += DT
        # if self.use_obstacles:
        #     self._update_obstacles()
        self._apply_action_to_robot(self.robot1, a1)
        self._apply_action_to_robot(self.robot2, a2)
        p.stepSimulation()

        ee1 = self._get_ee_pos(self.robot1)
        ee2 = self._get_ee_pos(self.robot2)
        q1 = self._get_joint_state(self.robot1)
        q2 = self._get_joint_state(self.robot2)
        
        # Compute and design the reward functions
        r1, d1 = self._compute_reward(ee1, target_pos_1, a1, q1)
        r2, d2 = self._compute_reward(ee2, target_pos_2, a2, q2)
        
        # PS: Add the direction reward
        d1 = np.linalg.norm(ee1 - target_pos_1)
        d2 = np.linalg.norm(ee2 - target_pos_2)

        r_progress_1 = self.prev_d1 - d1
        r_progress_2 = self.prev_d2 - d2
        r_progress = 5 * (r_progress_1 + r_progress_2)

        self.prev_d1 = d1
        self.prev_d2 = d2
        
        # Calculate the total reward 
        reward = r1 + r2 + r_progress
        terminated = False
        truncated = False

        collision = (
            self._check_robot_obstacle_collision(self.robot1) or
            self._check_robot_obstacle_collision(self.robot2) or
            self._check_robot_robot_collision()
        )

        if collision:
            reward -= 300.0
            terminated = True

        both_success = (d1 < 0.15) and (d2 < 0.15)
        if both_success:
            reward += 1000.0
            terminated = True

        if self.current_time >= max_time:
            truncated = True

        info = {
            "is_successful": both_success,
            "d1": float(d1),
            "d2": float(d2),
        }

        return self._get_obs(), float(reward), terminated, truncated, info
    def _generate_global_path_single(self, x0, y0, z0, deltax,deltay,deltaz):
        # 使用 RRT* 生成全局路径（简化版）
        # 这里可以调用外部的 RRT* 实现，返回路径点
        return np.array([
            [x0, y0, z0],
            [x0 + 0.25 * deltax, y0 + 0.25 * deltay, z0 + 0.25 * deltaz],
            [x0 + 0.5 * deltax, y0 + 0.5 * deltay, z0 + 0.25 * deltaz],
            [x0 + 0.75 * deltax, y0 + 0.75 * deltay, z0 + 0.75 * deltaz],
            [x0 + deltax, y0 + deltay, z0 + deltaz]
        ])
    def close(self):
        p.disconnect()

# 添加训练回调类以记录数据
class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

        self._cur_reward = 0.0
        self._cur_len = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        self._cur_reward += float(rewards[0])
        self._cur_len += 1

        if infos[0].get("is_successful", False):
            self.success_count += 1

        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._cur_reward)
            self.episode_lengths.append(self._cur_len)
            self._cur_reward = 0.0
            self._cur_len = 0

        return True

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








