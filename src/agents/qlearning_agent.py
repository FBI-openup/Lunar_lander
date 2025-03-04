# src/agents/qlearning_agent.py
import numpy as np
import time
import random
from .base_agent import BaseAgent
from src.utils.prioritized_buffer import PrioritizedReplayBuffer


class QLearningAgent(BaseAgent):
    """
    进一步增强的Q-Learning智能体实现
    优化状态表示和奖励函数
    针对LunarLander-v3离散动作空间版本特别优化
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.15,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 min_epsilon: float = 0.02,
                 replay_buffer_size: int = 50000,
                 batch_size: int = 32,
                 learning_starts: int = 1000,
                 target_update_freq: int = 500,
                 prioritized_replay_alpha: float = 0.6,
                 prioritized_replay_beta: float = 0.4):
        """
        初始化增强型Q-Learning智能体
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            min_epsilon: 最小探索率
            replay_buffer_size: 经验回放缓冲区大小
            batch_size: 学习批次大小
            learning_starts: 开始学习所需的最小经验数量
            target_update_freq: 目标网络更新频率
            prioritized_replay_alpha: 优先级回放alpha参数
            prioritized_replay_beta: 优先级回放beta参数
        """
        super().__init__(state_dim, action_dim)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq

        # 记录训练统计数据
        self.training_steps = 0
        self.last_update_time = time.time()
        self.successful_episodes = []  # 记录成功着陆的回合

        # 使用不均匀离散化，在关键区域使用更精细的粒度
        # 更精细地离散化关键状态
        # x位置: 着陆区附近更精细
        self.x_bins = self._create_nonuniform_bins(
            -1.0, 1.0,  # 范围
            center=0.0,  # 中心点（着陆区）
            n_bins=40,  # 总区间数
            center_density=4.0  # 中心密度系数
        )

        # y位置: 接近地面时更精细
        self.y_bins = self._create_nonuniform_bins(
            0.0, 1.5,  # 范围
            center=0.05,  # 中心点（接近地面）
            n_bins=40,
            center_density=5.0
        )

        # x速度: 低速时更精细
        self.vx_bins = self._create_nonuniform_bins(
            -2.0, 2.0,
            center=0.0,
            n_bins=20,
            center_density=3.0
        )

        # y速度: 接近零和小负值时更精细 (软着陆速度)
        self.vy_bins = self._create_nonuniform_bins(
            -2.0, 2.0,
            center=-0.1,  # 稍微负一点的速度最好，表示缓慢下降
            n_bins=25,
            center_density=4.0
        )

        # 角度: 接近垂直时更精细
        self.angle_bins = self._create_nonuniform_bins(
            -np.pi / 2, np.pi / 2,
            center=0.0,
            n_bins=40,
            center_density=5.0
        )

        # 角速度: 接近零时更精细
        self.angular_velocity_bins = self._create_nonuniform_bins(
            -3.0, 3.0,
            center=0.0,
            n_bins=20,
            center_density=3.0
        )

        # 腿接触是二值的，无需特殊处理
        self.leg_bins = [0.5]  # 仅需一个阈值

        # 收集所有分箱到一个列表中，方便索引
        self.all_bins = [
            self.x_bins,
            self.y_bins,
            self.vx_bins,
            self.vy_bins,
            self.angle_bins,
            self.angular_velocity_bins,
            self.leg_bins,
            self.leg_bins
        ]

        # 定义状态范围，用于裁剪值
        self.state_ranges = [
            (-1.0, 1.0),  # x位置
            (0.0, 1.5),  # y位置
            (-2.0, 2.0),  # x速度
            (-2.0, 2.0),  # y速度
            (-np.pi / 2, np.pi / 2),  # 角度
            (-3.0, 3.0),  # 角速度
            (0, 1),  # 左腿接触
            (0, 1)  # 右腿接触
        ]

        # 增加附加特征：合成特征标志
        # 这些"虚拟"状态可以帮助Q表更好地捕捉重要的状态组合
        self.use_composite_features = True

        # 使用字典实现Q表，节省内存并处理高维状态
        self.q_table = {}

        # 记录访问次数，用于自适应学习率
        self.state_visit_counts = {}

        # 双重学习系数
        self.double_learning = True
        self.q_table_2 = {}  # 第二个Q表用于双重学习

        # 记录上一次选择的动作，用于实现更复杂的探索策略
        self.last_action = None

        # 初始化优先级经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_buffer_size,
            alpha=prioritized_replay_alpha,
            beta=prioritized_replay_beta
        )

        # 扫描优先级队列 - 存储需要优先更新的状态
        self.priority_queue = []
        self.max_queue_size = 1000

        # 记录回合数据，用于回合结束时的经验学习
        self.episode_experience = []
        self.current_episode_reward = 0

        # 奖励调整权重
        self.reward_weights = {
            'angle': 0.3,  # 角度接近垂直的奖励
            'angular_velocity': 0.3,  # 减小角速度的奖励
            'position': 0.2,  # x位置接近中心的奖励
            'velocity': 0.2,  # 软着陆速度的奖励
            'legs_contact': 0.5,  # 双腿接触的奖励
            'landing_combo': 1.0,  # 良好着陆姿态组合的奖励
            'fuel_penalty': 0.05,  # 燃料使用惩罚
            'wrong_engine': 0.2,  # 使用错误引擎的惩罚
            'proper_landing': 5.0  # 正确着陆的大量奖励
        }

        # 状态判断阈值
        self.thresholds = {
            'good_angle': 0.1,  # 良好角度阈值
            'good_angular_vel': 0.1,  # 良好角速度阈值
            'center_position': 0.2,  # 中心位置阈值
            'soft_landing': 0.3,  # 软着陆速度阈值
            'near_ground': 0.1  # 接近地面阈值
        }

    def _create_nonuniform_bins(self, min_val, max_val, center, n_bins, center_density):
        """
        创建非均匀分箱，在中心点附近使用更高的密度
        Args:
            min_val: 最小值
            max_val: 最大值
            center: 中心点，密度最高的位置
            n_bins: 总区间数
            center_density: 中心密度系数，越大中心越密集
        Returns:
            bins: 分箱边界值的数组
        """
        # 归一化中心点到[0,1]范围
        center_norm = (center - min_val) / (max_val - min_val)

        # 生成[0,1]范围内的基础均匀点
        uniform_points = np.linspace(0, 1, n_bins)

        # 应用变换，使中心附近更密集
        transformed_points = self._transform_to_nonuniform(uniform_points, center_norm, center_density)

        # 还原到原始范围
        bins = min_val + transformed_points * (max_val - min_val)

        return bins

    def _transform_to_nonuniform(self, points, center, density):
        """
        将均匀分布的点变换为非均匀分布，使中心附近更密集
        使用S形函数实现
        """
        # 计算每个点到中心的距离
        distances = np.abs(points - center)

        # S形变换函数
        transformed = points + (np.sign(center - points) *
                                np.sin(np.pi * distances) *
                                (1 - np.exp(-density * (1 - distances))))

        # 确保点仍然在[0,1]范围内并排序
        transformed = np.clip(transformed, 0, 1)
        transformed.sort()

        return transformed

    def clip_state(self, state):
        """
        将状态值限制在合理范围内，避免极端值
        """
        clipped = []
        for i, (value, (low, high)) in enumerate(zip(state, self.state_ranges)):
            # 特殊处理二值状态 (腿接触)
            if i >= 6:  # 腿接触状态
                clipped.append(int(value > 0.5))
            else:
                clipped.append(np.clip(value, low, high))
        return np.array(clipped)

    def discretize_state(self, state):
        """
        将连续状态转换为离散状态，使用非均匀分箱
        并添加合成特征
        """
        # 首先裁剪状态到合理范围
        state = self.clip_state(state)

        # 基本状态离散化
        discretized = []
        for i, (value, bins) in enumerate(zip(state, self.all_bins)):
            if i >= 6:  # 腿接触状态
                bin_index = int(value > 0.5)
            else:
                bin_index = np.digitize(value, bins)
            discretized.append(bin_index)

        base_state = tuple(discretized)

        # 如果不使用合成特征，直接返回基本状态
        if not self.use_composite_features:
            return base_state

        # 添加合成特征标志
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = state

        # 提取关键状态特征，构建额外标志位
        good_angle = int(abs(angle) < self.thresholds['good_angle'])
        low_angular_vel = int(abs(angular_velocity) < self.thresholds['good_angular_vel'])
        center_position = int(abs(x) < self.thresholds['center_position'])
        soft_descent = int(vy > -self.thresholds['soft_landing'] and vy < 0)
        near_ground = int(y < self.thresholds['near_ground'])
        both_legs = int(left_leg > 0 and right_leg > 0)

        # 多个标志位合成一个整数，用作额外状态维度
        # 注意这种合成方式可以确保不同组合映射到不同的整数
        flag1 = (good_angle << 0) | (low_angular_vel << 1) | (center_position << 2)
        flag2 = (soft_descent << 0) | (near_ground << 1) | (both_legs << 2)

        # 合并基本状态和标志位
        return (*base_state, flag1, flag2)

    def get_q_value(self, state_key, action, q_table=None):
        """
        获取状态动作对的Q值，使用自定义默认值
        """
        if q_table is None:
            q_table = self.q_table

        # 对于新状态，优化初始Q值，使其略带乐观
        # 这有助于鼓励探索未知状态
        if state_key not in q_table:
            q_table[state_key] = np.zeros(self.action_dim) + 0.1  # 略带乐观的初始化
        return q_table[state_key][action]

    def select_action(self, state):
        """
        使用改进的探索策略选择动作
        """
        self.training_steps += 1

        # 离散化状态
        state_key = self.discretize_state(state)

        # 增加访问计数
        if state_key not in self.state_visit_counts:
            self.state_visit_counts[state_key] = 0
        self.state_visit_counts[state_key] += 1

        # 确保我们有这个状态的Q值
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim) + 0.1

        # 探索策略
        if np.random.random() < self.epsilon:
            # 分析当前状态，实现更智能的探索
            x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = state

            # 特殊情况探索策略，优化着陆行为
            if y < 0.3:  # 接近地面
                if abs(angle) > 0.3 or abs(angular_velocity) > 0.5:
                    # 角度不正或旋转太快，必须纠正
                    action = 1 if angle > 0 else 3  # 选择正确的侧向引擎
                elif vy < -0.5:  # 下降太快
                    # 优先使用主引擎减速
                    action = 2
                elif abs(x) > 0.3:  # 不在着陆区中心
                    # 尝试向中心移动
                    action = 3 if x > 0 else 1
                else:
                    # 已接近理想着陆状态，小心操作
                    choices = [0, 2]  # 只考虑不操作或主引擎
                    action = np.random.choice(choices)
            else:
                # 高空探索，基于上一个动作的连续性
                if self.last_action is not None and np.random.random() < 0.7:
                    # 30%概率选择相同动作，70%概率选择相邻动作
                    if np.random.random() < 0.3:
                        action = self.last_action
                    else:
                        # 选择相邻动作
                        if self.last_action == 0:
                            # 如果上次是不操作，优先考虑方向修正
                            choices = [1, 2, 3] if abs(angle) < 0.1 else [1, 3]
                        elif self.last_action == 2:
                            # 如果上次是主引擎，可能需要转向调整
                            choices = [0, 1, 3]
                        else:
                            # 如果上次是侧向引擎，可能需要切换方向
                            choices = [0, 2, 1 if self.last_action == 3 else 3]
                        action = np.random.choice(choices)
                else:
                    # 考虑物理状态的随机探索
                    if abs(angular_velocity) > 1.0:
                        # 旋转过快，优先考虑稳定
                        choices = [1, 3] if angular_velocity > 0 else [3, 1]
                        action = np.random.choice(choices)
                    elif abs(angle) > 0.5:
                        # 角度过大，需要纠正
                        action = 1 if angle > 0 else 3
                    else:
                        # 普通状态，完全随机
                        action = np.random.randint(self.action_dim)
        else:
            # 利用最优动作
            # 如果使用双重学习，结合两个Q表的值
            if self.double_learning:
                if state_key not in self.q_table_2:
                    self.q_table_2[state_key] = np.zeros(self.action_dim) + 0.1
                combined_q = self.q_table[state_key] + self.q_table_2[state_key]
                action = int(np.argmax(combined_q))
            else:
                action = int(np.argmax(self.q_table[state_key]))

        self.last_action = action
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        增强型Q-learning学习，结合经验回放和优先级扫描
        """
        # 离散化状态
        current_state = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        # 1. 添加经验到回合记录
        self.episode_experience.append((state, action, reward, next_state, done))
        self.current_episode_reward += reward

        # 2. 计算TD误差，用于优先级回放
        # 确保状态存在于Q表中
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.action_dim) + 0.1
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim) + 0.1

        current_q = self.get_q_value(current_state, action)

        if done:
            target_q = reward
        else:
            if self.double_learning:
                if next_state_key not in self.q_table_2:
                    self.q_table_2[next_state_key] = np.zeros(self.action_dim) + 0.1
                next_best_action = np.argmax(self.q_table[next_state_key])
                next_q = self.get_q_value(next_state_key, next_best_action, self.q_table_2)
            else:
                next_q = np.max(self.q_table[next_state_key])

            target_q = reward + self.gamma * next_q

        # 计算TD误差
        td_error = target_q - current_q

        # 3. 添加到优先级回放缓冲区
        shaped_reward = self.shape_reward(state, action, reward, next_state, done)
        self.replay_buffer.add(state, action, shaped_reward, next_state, done, error=td_error)

        # 4. 将当前状态添加到优先级队列，用于扫描更新
        self.add_to_priority_queue(current_state, abs(td_error))

        # 5. 从回放缓冲区学习
        if self.replay_buffer.is_ready(self.batch_size) and self.training_steps > self.learning_starts:
            self.learn_from_replay_buffer()

        # 6. 如果回合结束，进行特殊处理
        if done:
            # 标记成功的回合，用于后续优先学习
            if self.current_episode_reward > 0:
                self.successful_episodes.append(self.episode_experience)

                # 限制成功回合的最大存储数量
                if len(self.successful_episodes) > 10:
                    self.successful_episodes.pop(0)  # 先进先出

            # 重置回合状态
            self.episode_experience = []
            self.current_episode_reward = 0

            # 从成功回合中额外学习（如果有）
            if len(self.successful_episodes) > 0 and np.random.random() < 0.5:
                # 50%的概率从成功回合中学习
                successful_episode = random.choice(self.successful_episodes)
                for exp in successful_episode:
                    s, a, r, next_s, d = exp
                    # 增加奖励以强化成功经验
                    self.replay_buffer.add(s, a, r * 1.5, next_s, d)

        # 7. 执行优先级扫描更新
        if self.training_steps % 10 == 0:  # 每10步执行一次扫描
            self.priority_sweep(5)  # 一次更新5个高优先级状态

        # 8. 每1000步打印信息
        if self.training_steps % 1000 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            self.last_update_time = current_time
            print(f"Q表大小: {len(self.q_table):,} 状态, "
                  f"缓冲区: {len(self.replay_buffer):,} 经验, "
                  f"成功回合: {len(self.successful_episodes)}, "
                  f"Epsilon: {self.epsilon:.3f}")

    def learn_from_replay_buffer(self):
        """
        从经验回放缓冲区中学习
        """
        # 从缓冲区采样经验批次
        experiences, indices, importance_weights = self.replay_buffer.sample(self.batch_size)

        # 计算TD误差并更新Q值
        td_errors = []

        for i, (state, action, reward, next_state, done) in enumerate(experiences):
            weight = importance_weights[i]
            state_key = self.discretize_state(state)
            next_state_key = self.discretize_state(next_state)

            # 确保状态存在于Q表中
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim) + 0.1
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_dim) + 0.1

            # 自适应学习率
            visit_count = self.state_visit_counts.get(state_key, 0) + 1
            adaptive_lr = self.lr / (1 + 0.05 * np.log(visit_count))

            # 使用双重学习更新
            if self.double_learning and np.random.random() < 0.5:
                # 从第一个表选择动作，用第二个表评估
                if next_state_key not in self.q_table_2:
                    self.q_table_2[next_state_key] = np.zeros(self.action_dim) + 0.1

                current_q = self.get_q_value(state_key, action, self.q_table)

                if done:
                    target_q = reward
                else:
                    next_best_action = np.argmax(self.q_table[next_state_key])
                    next_q = self.get_q_value(next_state_key, next_best_action, self.q_table_2)
                    target_q = reward + self.gamma * next_q

                # 计算TD误差
                td_error = target_q - current_q

                # 使用重要性权重更新Q值
                self.q_table[state_key][action] += adaptive_lr * weight * td_error

            else:
                # 更新第二个Q表或常规更新
                q_table = self.q_table_2 if self.double_learning else self.q_table

                if next_state_key not in q_table:
                    q_table[next_state_key] = np.zeros(self.action_dim) + 0.1

                current_q = self.get_q_value(state_key, action, q_table)

                if done:
                    target_q = reward
                else:
                    if self.double_learning:
                        next_best_action = np.argmax(q_table[next_state_key])
                        next_q = self.get_q_value(next_state_key, next_best_action, self.q_table)
                    else:
                        next_q = np.max(q_table[next_state_key])

                    target_q = reward + self.gamma * next_q

                # 计算TD误差
                td_error = target_q - current_q

                # 使用重要性权重更新Q值
                q_table[state_key][action] += adaptive_lr * weight * td_error

            # 收集TD误差用于更新优先级
            td_errors.append(td_error)

            # 将更新后的状态添加到优先级队列
            self.add_to_priority_queue(state_key, abs(td_error))

        # 更新经验优先级
        self.replay_buffer.update_priorities(indices, td_errors)

    def add_to_priority_queue(self, state_key, priority):
        """
        将状态添加到优先级队列
        """
        # 检查状态是否已在队列中
        for i, (s, p) in enumerate(self.priority_queue):
            if s == state_key:
                # 如果新优先级更高，更新优先级
                if priority > p:
                    self.priority_queue[i] = (state_key, priority)
                return

        # 添加新状态到队列
        self.priority_queue.append((state_key, priority))

        # 如果队列太长，移除优先级最低的状态
        if len(self.priority_queue) > self.max_queue_size:
            self.priority_queue.sort(key=lambda x: x[1], reverse=True)
            self.priority_queue = self.priority_queue[:self.max_queue_size]

    def priority_sweep(self, n_updates=5):
        """
        执行优先级扫描，更新高优先级状态的相关状态
        """
        if not self.priority_queue:
            return

        # 按优先级排序
        self.priority_queue.sort(key=lambda x: x[1], reverse=True)

        # 更新前n个高优先级状态
        for i in range(min(n_updates, len(self.priority_queue))):
            state_key, _ = self.priority_queue[i]

            # 获取该状态下的最佳动作
            if state_key not in self.q_table:
                continue

            best_action = np.argmax(self.q_table[state_key])

            # 更新可能导致这个状态的前置状态
            # 这需要一些启发式规则来猜测可能的前置状态
            # 这里使用简单的启发式：修改状态的某些维度
            for dim in range(len(state_key) - 2 if self.use_composite_features else len(state_key)):
                # 尝试相邻的状态值
                for offset in [-1, 1]:
                    modified_state = list(state_key)
                    if dim < 6:  # 只修改非二值状态
                        # 确保修改后的值在有效范围内
                        if dim == 0:  # x位置
                            bin_max = len(self.x_bins)
                        elif dim == 1:  # y位置
                            bin_max = len(self.y_bins)
                        elif dim == 2:  # x速度
                            bin_max = len(self.vx_bins)
                        elif dim == 3:  # y速度
                            bin_max = len(self.vy_bins)
                        elif dim == 4:  # 角度
                            bin_max = len(self.angle_bins)
                        elif dim == 5:  # 角速度
                            bin_max = len(self.angular_velocity_bins)
                        else:
                            bin_max = 2  # 二值状态

                        modified_state[dim] = max(0, min(bin_max, modified_state[dim] + offset))
                    modified_state = tuple(modified_state)

                    # 跳过不存在的状态
                    if modified_state not in self.q_table:
                        continue

                    # 对于每个可能的动作，检查是否可能转换到目标状态
                    for action in range(self.action_dim):
                        # 这里我们不知道确切的转换概率，所以使用启发式方法
                        # 简单地更新前置状态的Q值，假设它可能导致目标状态
                        current_q = self.q_table[modified_state][action]
                        next_value = self.q_table[state_key][best_action]

                        # 计算可能的目标Q值
                        target_q = 0 + self.gamma * next_value  # 假设奖励为0

                        # 如果目标Q值比当前值大，进行更新
                        if target_q > current_q:
                            # 使用小学习率更新，因为这是基于启发式的
                            self.q_table[modified_state][action] += 0.01 * (target_q - current_q)

        # 移除已处理的状态
        self.priority_queue = self.priority_queue[min(n_updates, len(self.priority_queue)):]

    def shape_reward(self, state, action, reward, next_state, done):
        """
        改进的奖励整形函数，为中间状态添加精细的额外奖励
        """
        # 提取状态信息
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = state
        next_x, next_y, next_vx, next_vy, next_angle, next_angular_vel, next_left, next_right = next_state

        # 初始奖励
        shaped_reward = reward

        # 奖励1: 角度改进 (保持飞行器垂直)
        angle_improvement = abs(angle) - abs(next_angle)
        shaped_reward += self.reward_weights['angle'] * angle_improvement

        # 奖励2: 角速度改进 (减小旋转)
        angular_vel_improvement = abs(angular_velocity) - abs(next_angular_vel)
        shaped_reward += self.reward_weights['angular_velocity'] * angular_vel_improvement

        # 奖励3: 位置改进 (靠近着陆区中心)
        position_improvement = abs(x) - abs(next_x)
        shaped_reward += self.reward_weights['position'] * position_improvement

        # 奖励4: 软着陆速度调整
        if vy < 0 and next_vy > vy:  # 如果正在下降并且速度减小
            velocity_improvement = next_vy - vy
            shaped_reward += self.reward_weights['velocity'] * velocity_improvement

            # 特别奖励接近完美着陆速度
            if -0.3 < next_vy < -0.1 and y < 0.3:
                shaped_reward += 2 * self.reward_weights['velocity']

        # 奖励5: 着陆触地奖励
        if (next_left > 0 or next_right > 0) and (left_leg == 0 and right_leg == 0):
            # 首次触地
            shaped_reward += self.reward_weights['legs_contact']

        if next_left > 0 and next_right > 0:
            # 双腿触地，额外奖励
            shaped_reward += self.reward_weights['legs_contact']

        # 奖励6: 良好着陆姿态组合奖励
        good_landing_attitude = (
                abs(next_angle) < self.thresholds['good_angle'] and
                abs(next_angular_vel) < self.thresholds['good_angular_vel'] and
                abs(next_x) < self.thresholds['center_position']
        )

        if good_landing_attitude:
            shaped_reward += self.reward_weights['landing_combo']

            # 如果还接近地面，额外加分
            if next_y < self.thresholds['near_ground']:
                shaped_reward += self.reward_weights['landing_combo']

        # 奖励7: 成功着陆的巨大奖励
        if next_left > 0 and next_right > 0 and good_landing_attitude and abs(next_vy) < self.thresholds[
            'soft_landing']:
            shaped_reward += self.reward_weights['proper_landing']

        # 惩罚8: 不必要地使用燃料
        if action != 0 and abs(next_angle) < 0.05 and abs(next_x) < 0.1 and abs(next_vy) < 0.1:
            shaped_reward -= self.reward_weights['fuel_penalty']

        # 惩罚9: 在错误的时机使用错误的引擎
        if abs(angle) > 0.2:
            # 角度偏移时应该使用侧向引擎纠正
            if angle > 0 and action == 3:  # 角度为正，应该使用右引擎
                shaped_reward += self.reward_weights['wrong_engine']
            elif angle < 0 and action == 1:  # 角度为负，应该使用左引擎
                shaped_reward += self.reward_weights['wrong_engine']
            elif action == 2:  # 不应该使用主引擎纠正角度
                shaped_reward -= self.reward_weights['wrong_engine'] / 2

        # 惩罚10: 快速下落时不使用主引擎
        if vy < -1.0 and action != 2 and y < 0.5:
            shaped_reward -= self.reward_weights['wrong_engine'] / 2

        # 奖励11: 特殊情况 - 接近着陆但未着陆时的正确操作
        if y < 0.3 and abs(x) < 0.2 and left_leg == 0 and right_leg == 0:
            if abs(angle) > 0.1:
                # 需要调整角度
                correct_action = 1 if angle > 0 else 3
                if action == correct_action:
                    shaped_reward += self.reward_weights['landing_combo'] / 2
            elif vy < -0.5:
                # 需要减速
                if action == 2:
                    shaped_reward += self.reward_weights['landing_combo'] / 2

        # 奖励12: 飞行中的稳定控制
        if abs(angular_velocity) > 1.0:
            correct_action = 1 if angular_velocity > 0 else 3
            if action == correct_action:
                shaped_reward += self.reward_weights['angular_velocity'] / 2

        return shaped_reward

    def decay_epsilon(self, decay_rate: float = 0.996):
        """
        衰减探索率，使用指数衰减，但速度更慢
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * decay_rate)

    def get_state_action_values(self, state):
        """
        获取给定状态下所有动作的Q值
        """
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim) + 0.1

        # 如果使用双重学习，结合两个Q表的值
        if self.double_learning:
            if state_key not in self.q_table_2:
                self.q_table_2[state_key] = np.zeros(self.action_dim) + 0.1
            combined_q = (self.q_table[state_key] + self.q_table_2[state_key]) / 2
            return combined_q

        return self.q_table[state_key]