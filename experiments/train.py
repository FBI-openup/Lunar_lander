"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:46
@Author        : FengD
@File          : train
@Description   :
"""
# experiments/train.py
from typing import Dict, Any
import numpy as np
from src.utils.logger import MetricsLogger


def train_agent(env, agent, config: Dict[str, Any]):
    """通用训练函数"""
    logger = MetricsLogger(config['save_dir'])

    for episode in range(config['num_episodes']):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 智能体学习
            if hasattr(agent, 'learn'):
                agent.learn(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

        # 记录日志
        logger.add_episode(episode_reward, episode_steps)

        # 打印进度
        if (episode + 1) % config['log_interval'] == 0:
            avg_reward = np.mean(logger.episode_rewards[-config['log_interval']:])
            avg_steps = np.mean(logger.episode_lengths[-config['log_interval']:])
            print(f"Episode {episode + 1}/{config['num_episodes']}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Steps: {avg_steps:.2f}")
            print("-" * 50)

    return logger
