"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:43
@Author        : FengD
@File          : base_agent
@Description   :
"""
# src/agents/base_agent.py
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """所有智能体的基类"""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """选择动作"""
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """学习更新"""
        pass