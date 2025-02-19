"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:44
@Author        : FengD
@File          : logger
@Description   :
"""
import numpy as np
import json
import os
from datetime import datetime
from src.utils.visualizer import Visualizer


class MetricsLogger:
    """指标记录器，用于记录和可视化训练过程"""

    def __init__(self, save_dir: str = None):
        """
        初始化记录器
        Args:
            save_dir: 结果保存路径
        """
        self.save_dir = save_dir
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

    def save_metrics(self, save_dir: str = None):
        """
        保存训练指标
        Args:
            save_dir: 保存路径，如果为None则使用初始化时的路径
        """
        save_path = save_dir if save_dir else self.save_dir
        if save_path:
            metrics = {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'avg_rewards': self.avg_rewards,
                'avg_lengths': self.avg_lengths,
                'final_avg_reward': self.avg_rewards[-1] if self.avg_rewards else 0,
                'final_avg_length': self.avg_lengths[-1] if self.avg_lengths else 0
            }

            metrics_path = os.path.join(save_path, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

    def plot_metrics(self, save_path: str = None):
        """
        绘制训练指标图表
        Args:
            save_path: 图表保存路径
        """
        if save_path is None and self.save_dir:
            save_path = os.path.join(self.save_dir, 'training_plot.png')

        Visualizer.plot_training_history(
            self.episode_rewards,
            self.episode_lengths,
            save_path=save_path
        )