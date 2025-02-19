"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:46
@Author        : FengD
@File          : evaluate
@Description   :
"""
from typing import Dict, Any
import numpy as np


def evaluate_agent(env, agent, config: Dict[str, Any]):
    """通用评估函数"""
    episode_rewards = []
    episode_lengths = []

    for episode in range(config['eval_episodes']):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if config['render']:
                env.render()

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)

        if (episode + 1) % 10 == 0:
            print(f"Evaluation Episode {episode + 1}/{config['eval_episodes']}")
            print(f"Current Average Reward: {np.mean(episode_rewards[-10:]):.2f}")

    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
        "std_steps": float(np.std(episode_lengths))
    }

    return metrics, episode_rewards, episode_lengths