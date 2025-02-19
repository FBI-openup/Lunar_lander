"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:44
@Author        : FengD
@File          : visualizer
@Description   :
"""
# src/utils/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class Visualizer:
    """可视化工具类，用于绘制各种训练和评估结果图表"""

    @staticmethod
    def set_style():
        """设置图表样式"""
        # 使用内置的样式
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_theme(style="whitegrid")

    @staticmethod
    def plot_training_history(rewards: List[float],
                              steps: List[float],
                              window_size: int = 100,
                              save_path: Optional[str] = None):
        """
        绘制训练历史
        Args:
            rewards: 每个回合的奖励列表
            steps: 每个回合的步数列表
            window_size: 移动平均窗口大小
            save_path: 保存路径
        """
        Visualizer.set_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 计算移动平均
        rewards_mean = np.convolve(rewards,
                                   np.ones(window_size) / window_size,
                                   mode='valid')
        steps_mean = np.convolve(steps,
                                 np.ones(window_size) / window_size,
                                 mode='valid')

        # 绘制奖励曲线
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
        ax1.plot(range(window_size - 1, len(rewards)),
                 rewards_mean,
                 color='red',
                 label=f'{window_size}-Episode Moving Average')
        ax1.set_title('Training Rewards over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True)

        # 绘制步数曲线
        ax2.plot(steps, alpha=0.3, color='blue', label='Raw Steps')
        ax2.plot(range(window_size - 1, len(steps)),
                 steps_mean,
                 color='red',
                 label=f'{window_size}-Episode Moving Average')
        ax2.set_title('Episode Lengths over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_evaluation_results(metrics: Dict,
                                rewards: List[float],
                                lengths: List[float],
                                save_path: Optional[str] = None):
        """
        绘制评估结果
        Args:
            metrics: 评估指标字典
            rewards: 评估回合的奖励列表
            lengths: 评估回合的步数列表
            save_path: 保存路径
        """
        Visualizer.set_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 奖励分布直方图
        ax1.hist(rewards, bins=30, density=True, alpha=0.7)
        ax1.axvline(metrics['mean_reward'], color='r', linestyle='--',
                    label=f'Mean: {metrics["mean_reward"]:.2f}')
        ax1.axvline(metrics['mean_reward'] + metrics['std_reward'],
                    color='g', linestyle=':', label='±1 STD')
        ax1.axvline(metrics['mean_reward'] - metrics['std_reward'],
                    color='g', linestyle=':')
        ax1.set_title('Reward Distribution')
        ax1.set_xlabel('Reward')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True)

        # 步数分布直方图
        ax2.hist(lengths, bins=30, density=True, alpha=0.7)
        ax2.axvline(metrics['mean_steps'], color='r', linestyle='--',
                    label=f'Mean: {metrics["mean_steps"]:.2f}')
        ax2.axvline(metrics['mean_steps'] + metrics['std_steps'],
                    color='g', linestyle=':', label='±1 STD')
        ax2.axvline(metrics['mean_steps'] - metrics['std_steps'],
                    color='g', linestyle=':')
        ax2.set_title('Episode Length Distribution')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True)

        # 奖励时间序列
        ax3.plot(rewards, label='Reward')
        ax3.axhline(y=metrics['mean_reward'], color='r', linestyle='--',
                    label=f'Mean: {metrics["mean_reward"]:.2f}')
        ax3.fill_between(range(len(rewards)),
                         [metrics['mean_reward'] - metrics['std_reward']] * len(rewards),
                         [metrics['mean_reward'] + metrics['std_reward']] * len(rewards),
                         alpha=0.2, label='±1 STD')
        ax3.set_title('Evaluation Rewards over Episodes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.legend()
        ax3.grid(True)

        # 步数时间序列
        ax4.plot(lengths, label='Steps')
        ax4.axhline(y=metrics['mean_steps'], color='r', linestyle='--',
                    label=f'Mean: {metrics["mean_steps"]:.2f}')
        ax4.fill_between(range(len(lengths)),
                         [metrics['mean_steps'] - metrics['std_steps']] * len(lengths),
                         [metrics['mean_steps'] + metrics['std_steps']] * len(lengths),
                         alpha=0.2, label='±1 STD')
        ax4.set_title('Evaluation Steps over Episodes')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_agent_comparison(agent_results: Dict[str, Dict],
                              save_path: Optional[str] = None):
        """
        比较不同智能体的性能
        Args:
            agent_results: 不同智能体的评估结果字典
                {agent_name: {'mean_reward': float, 'std_reward': float, ...}}
            save_path: 保存路径
        """
        Visualizer.set_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 准备数据
        agents = list(agent_results.keys())
        means_reward = [results['mean_reward'] for results in agent_results.values()]
        stds_reward = [results['std_reward'] for results in agent_results.values()]
        means_steps = [results['mean_steps'] for results in agent_results.values()]
        stds_steps = [results['std_steps'] for results in agent_results.values()]

        # 绘制奖励对比条形图
        ax1.bar(agents, means_reward, yerr=stds_reward, capsize=5)
        ax1.set_title('Average Rewards Comparison')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True, axis='y')

        # 绘制步数对比条形图
        ax2.bar(agents, means_steps, yerr=stds_steps, capsize=5)
        ax2.set_title('Average Steps Comparison')
        ax2.set_ylabel('Mean Steps')
        ax2.grid(True, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def plot_results(rewards: List[float],
                 lengths: List[int],
                 save_path: Optional[str] = None):
    """
    便捷函数：绘制基本的评估结果
    Args:
        rewards: 奖励列表
        lengths: 步数列表
        save_path: 保存路径
    """
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(lengths)),
        "std_steps": float(np.std(lengths))
    }

    Visualizer.plot_evaluation_results(metrics, rewards, lengths, save_path)