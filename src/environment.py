"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:43
@Author        : FengD
@File          : environment.py
@Description   :
"""
# src/environment.py
import gymnasium as gym
import numpy as np
from typing import Tuple


class LunarLanderEnvironment:
    """
    月球着陆器环境的包装类，提供统一的接口和额外的功能
    """

    def __init__(self, seed: int = 42, render_mode: str = None):
        """
        初始化环境
        Args:
            seed: 随机种子，确保实验可重复性
        """
        self.env = gym.make('LunarLander-v3', render_mode=render_mode)
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