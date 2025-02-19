"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 00:19
@Author        : FengD
@File          : test
@Description   :
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from typing import List, Tuple, Dict


class LunarLanderEnvironment:
    """
    月球着陆器环境的包装类，提供统一的接口和额外的功能
    """

    def __init__(self, seed: int = 42):
        """
        初始化环境
        Args:
            seed: 随机种子，确保实验可重复性
        """
        self.env = gym.make('LunarLander-v2')
        self.env.reset(seed=seed)

        # 环境信息
        self.state_dim = self.env.observation_space.shape[0]  # 状态空间维度 (8维)
        self.action_dim = self.env.action_space.n  # 动作空间维度 (4个离散动作)

        # 记录训练数据
        self.episode_rewards = []  # 每个回合的总奖励
        self.episode_lengths = []  # 每个回合的长度

    def get_state_info(self) -> str:
        """
        返回状态空间的详细信息
        状态空间包含8个连续值:
        - 位置 x (水平位置)
        - 位置 y (垂直位置)
        - 速度 x (水平速度)
        - 速度 y (垂直速度)
        - 角度 (飞行器倾角)
        - 角速度
        - 左腿接触标志
        - 右腿接触标志
        """
        return """
        状态空间 (8维连续值):
        - x: 水平位置 (-1.5 到 +1.5)
        - y: 垂直位置 (0 到 +1.5)
        - vx: 水平速度 (-5 到 +5)
        - vy: 垂直速度 (-5 到 +5)
        - theta: 角度 (-π 到 +π)
        - omega: 角速度 (-5 到 +5)
        - left_leg: 左腿接触 (0 或 1)
        - right_leg: 右腿接触 (0 或 1)
        """

    def get_action_info(self) -> str:
        """
        返回动作空间的详细信息
        动作空间包含4个离散动作
        """
        return """
        动作空间 (4个离散动作):
        0: 不做任何操作
        1: 开启左引擎
        2: 开启主引擎
        3: 开启右引擎
        """

    def reset(self) -> np.ndarray:
        """
        重置环境
        Returns:
            初始状态
        """
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步动作
        Args:
            action: 选择的动作 (0-3)
        Returns:
            state: 下一个状态
            reward: 获得的奖励
            terminated: 回合是否结束
            truncated: 回合是否被截断
            info: 额外信息
        """
        return self.env.step(action)

    def render(self):
        """渲染环境"""
        self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()


class ExperienceBuffer:
    """
    经验回放缓冲区，用于DQN算法
    存储和采样(state, action, reward, next_state, done)转换
    """

    def __init__(self, capacity: int = 10000):
        """
        初始化经验回放缓冲区
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """
        添加一个转换到缓冲区
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        随机采样一个批次的转换
        Args:
            batch_size: 批次大小
        Returns:
            采样的转换列表
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class MetricsLogger:
    """
    指标记录器，用于记录和可视化训练过程
    """

    def __init__(self):
        self.episode_rewards = []  # 每个回合的总奖励
        self.episode_lengths = []  # 每个回合的步数
        self.avg_rewards = []  # 平均奖励（用于绘图）
        self.avg_lengths = []  # 平均步数（用于绘图）

    def add_episode(self, reward: float, length: int):
        """
        添加一个回合的数据
        Args:
            reward: 回合总奖励
            length: 回合步数
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        # 计算最近100个回合的平均值
        window_size = min(100, len(self.episode_rewards))
        self.avg_rewards.append(np.mean(self.episode_rewards[-window_size:]))
        self.avg_lengths.append(np.mean(self.episode_lengths[-window_size:]))

    def plot_metrics(self):
        """
        绘制训练指标图表
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # 绘制奖励曲线
        ax1.plot(self.avg_rewards, label='平均奖励')
        ax1.set_title('训练过程中的平均奖励')
        ax1.set_xlabel('回合')
        ax1.set_ylabel('奖励')
        ax1.legend()

        # 绘制步数曲线
        ax2.plot(self.avg_lengths, label='平均步数')
        ax2.set_title('训练过程中的平均步数')
        ax2.set_xlabel('回合')
        ax2.set_ylabel('步数')
        ax2.legend()

        plt.tight_layout()
        plt.show()


def evaluate_agent(env: LunarLanderEnvironment,
                   agent,  # 这里agent的类型取决于具体的算法实现
                   n_episodes: int = 100,
                   render: bool = False) -> Dict:
    """
    评估智能体的性能
    Args:
        env: 环境实例
        agent: 要评估的智能体
        n_episodes: 评估的回合数
        render: 是否渲染环境
    Returns:
        包含评估指标的字典
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            if render:
                env.render()

            # 根据策略选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # 计算评估指标
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }

    return metrics


# 示例使用
if __name__ == "__main__":
    # 创建环境
    env = LunarLanderEnvironment(seed=42)

    # 打印环境信息
    print("状态空间信息:")
    print(env.get_state_info())
    print("\n动作空间信息:")
    print(env.get_action_info())

    # 创建指标记录器
    logger = MetricsLogger()

    # 创建经验回放缓冲区（用于DQN）
    buffer = ExperienceBuffer()

    # 环境测试
    state = env.reset()
    print("\n初始状态:", state)

    # 执行随机动作
    action = random.randint(0, env.action_dim - 1)
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"执行动作 {action} 后:")
    print(f"下一状态: {next_state}")
    print(f"奖励: {reward}")
    print(f"是否结束: {terminated}")
    print(f"是否截断: {truncated}")

    env.close()