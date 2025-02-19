"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:57
@Author        : FengD
@File          : random_agent
@Description   :
"""
from .base_agent import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):
    """随机策略智能体"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)

    def select_action(self, state: np.ndarray) -> int:
        """随机选择动作"""
        return np.random.randint(0, self.action_dim)

    def learn(self, *args, **kwargs):
        """
        随机智能体不需要学习，但必须实现这个方法以满足基类要求
        """
        pass