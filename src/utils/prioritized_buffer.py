"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-03-04 08:48
@Author        : FengD
@File          : prioritized_buffer
@Description   :
"""
# src/utils/prioritized_buffer.py
import numpy as np
from collections import deque
import random


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    存储转换(state, action, reward, next_state, done)并根据TD误差赋予优先级
    """

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先级回放缓冲区
        Args:
            capacity: 缓冲区容量
            alpha: 确定优先级使用程度的指数 (0 - 均匀采样, 1 - 完全优先级采样)
            beta: 重要性采样的指数，校正优先级采样偏差 (0 - 无校正, 1 - 完全校正)
            beta_increment: 每次采样时beta的增量
        """
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # 新经验的初始优先级

    def add(self, state, action, reward, next_state, done, error=None):
        """
        添加新经验到缓冲区
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 回合是否结束
            error: TD误差，如果为None则使用最大优先级
        """
        # 创建新经验
        experience = (state, action, reward, next_state, done)

        # 如果缓冲区未满，添加新条目
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # 否则覆盖旧条目
            self.memory[self.position] = experience

        # 设置优先级
        priority = self.max_priority if error is None else (abs(error) + 1e-5) ** self.alpha
        self.priorities[self.position] = priority

        # 更新位置
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        按优先级采样批次经验
        Args:
            batch_size: 采样批次大小
        Returns:
            采样的经验批次、采样索引和重要性权重
        """
        # 如果缓冲区中没有足够的经验，减少批次大小
        n_samples = min(batch_size, len(self.memory))

        # 确保probabilities数组只包含有效的优先级
        valid_priorities = self.priorities[:len(self.memory)]

        # 计算采样概率
        probabilities = valid_priorities / np.sum(valid_priorities)

        # 按概率采样索引
        indices = np.random.choice(len(self.memory), n_samples, p=probabilities, replace=False)

        # 获取经验
        samples = [self.memory[idx] for idx in indices]

        # 计算重要性权重
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重

        # 增加beta值（随时间接近1）
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        """
        更新指定经验的优先级
        Args:
            indices: 要更新的经验索引
            errors: 对应的TD误差
        """
        for idx, error in zip(indices, errors):
            # 确保索引有效
            if idx < len(self.memory):
                priority = (abs(error) + 1e-5) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        返回缓冲区中经验的数量
        """
        return len(self.memory)

    def is_ready(self, batch_size):
        """
        检查缓冲区是否准备好进行采样
        """
        return len(self.memory) >= batch_size