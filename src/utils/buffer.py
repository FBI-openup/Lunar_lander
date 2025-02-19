"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:43
@Author        : FengD
@File          : buffer
@Description   :
"""
# src/utils/buffer.py
from collections import deque
import random
import numpy as np
from typing import List, Tuple


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