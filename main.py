"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 10:04
@Author        : FengD
@File          : main
@Description   :
"""
# main.py
import os
from datetime import datetime
import json
from src.environment import LunarLanderEnvironment
from src.agents.random_agent import RandomAgent
# from src.agents.td_agent import TDAgent
# from src.agents.sarsa_agent import SARSAAgent
from src.agents.qlearning_agent import QLearningAgent
# from src.agents.dqn_agent import DQNAgent
from experiments.train import train_agent
from experiments.evaluate import evaluate_agent
from src.utils.visualizer import Visualizer


# 更新 main.py 中的 Config 类
# 添加新的参数，支持优先级经验回放和优先级扫描

# 更新 main.py 中的 Config 类
# 添加新的参数，支持优化状态表示和奖励函数

class Config:
    """配置类，集中管理所有可调参数"""
    # 基础设置
    AGENT_TYPE = 'qlearning'  # 可选: 'random', 'qlearning', 'dqn'
    MODE = 'train_and_evaluate'  # 可选: 'train', 'evaluate', 'train_and_evaluate'
    SEED = 42

    # 训练参数
    TRAIN_EPISODES = 10000  # 增加训练回合数
    LOG_INTERVAL = 1000

    # 评估参数
    EVAL_EPISODES = 100
    RENDER = True  # 是否渲染环境

    # 保存设置
    BASE_SAVE_DIR = 'results'

    # 算法特定参数
    class QLearning:
        LEARNING_RATE = 0.15
        GAMMA = 0.995  # 增加折扣因子，更重视长期回报
        EPSILON = 1.0  # 起始探索率
        MIN_EPSILON = 0.02
        REPLAY_BUFFER_SIZE = 50000
        BATCH_SIZE = 64  # 增加批量大小
        LEARNING_STARTS = 1000
        TARGET_UPDATE_FREQ = 500
        PRIORITIZED_REPLAY_ALPHA = 0.6
        PRIORITIZED_REPLAY_BETA = 0.4


# 更新 main.py 中的 get_agent 函数

def get_agent(agent_type: str, state_dim: int, action_dim: int):
    """根据类型创建智能体"""
    agents = {
        'random': lambda: RandomAgent(state_dim, action_dim),
        'qlearning': lambda: QLearningAgent(
            state_dim,
            action_dim,
            learning_rate=Config.QLearning.LEARNING_RATE,
            gamma=Config.QLearning.GAMMA,
            epsilon=Config.QLearning.EPSILON,
            min_epsilon=Config.QLearning.MIN_EPSILON,
            replay_buffer_size=Config.QLearning.REPLAY_BUFFER_SIZE,
            batch_size=Config.QLearning.BATCH_SIZE,
            learning_starts=Config.QLearning.LEARNING_STARTS,
            target_update_freq=Config.QLearning.TARGET_UPDATE_FREQ,
            prioritized_replay_alpha=Config.QLearning.PRIORITIZED_REPLAY_ALPHA,
            prioritized_replay_beta=Config.QLearning.PRIORITIZED_REPLAY_BETA
        ),
    }

    if agent_type not in agents:
        raise ValueError(f"未知的智能体类型: {agent_type}")

    return agents[agent_type]()


def setup_experiment():
    """设置实验环境并返回相关配置"""
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(Config.BASE_SAVE_DIR, Config.AGENT_TYPE, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 保存实验配置
    config_dict = {
        'agent_type': Config.AGENT_TYPE,
        'mode': Config.MODE,
        'seed': Config.SEED,
        'train_episodes': Config.TRAIN_EPISODES,
        'eval_episodes': Config.EVAL_EPISODES,
        'render': Config.RENDER
    }

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    return {
        'save_dir': save_dir,
        'num_episodes': Config.TRAIN_EPISODES,
        'log_interval': Config.LOG_INTERVAL,
        'eval_episodes': Config.EVAL_EPISODES,
        'render': Config.RENDER
    }


def run_experiment(env, agent, config):
    """运行实验"""
    if Config.MODE in ['train', 'train_and_evaluate']:
        print(f"\n开始训练 {Config.AGENT_TYPE} 智能体...")
        logger = train_agent(env, agent, config)
        Visualizer.plot_training_history(
            logger.episode_rewards,
            logger.episode_lengths,
            save_path=os.path.join(config['save_dir'], 'training_history.png')
        )

    if Config.MODE in ['evaluate', 'train_and_evaluate']:
        print(f"\n开始评估 {Config.AGENT_TYPE} 智能体...")
        metrics, rewards, lengths = evaluate_agent(env, agent, config)

        # 保存评估结果
        with open(os.path.join(config['save_dir'], 'eval_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # 绘制评估结果
        Visualizer.plot_evaluation_results(
            metrics, rewards, lengths,
            save_path=os.path.join(config['save_dir'], 'eval_results.png')
        )

        print("\n评估结果:")
        print(f"平均奖励: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"奖励范围: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print(f"平均步数: {metrics['mean_steps']:.2f} ± {metrics['std_steps']:.2f}")


def main():
    """主函数"""
    # 设置实验
    config = setup_experiment()

    # 创建环境和智能体
    env = LunarLanderEnvironment(seed=Config.SEED)
    agent = get_agent(Config.AGENT_TYPE, env.state_dim, env.action_dim)

    # 运行实验
    try:
        run_experiment(env, agent, config)
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    # except Exception as e:
    #     print(f"\n实验出错: {str(e)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()