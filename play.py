"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 10:28
@Author        : FengD
@File          : play.py
@Description   :
"""
# play.py
import time
from src.environment import LunarLanderEnvironment
from src.agents.random_agent import RandomAgent
# from src.agents.td_agent import TDAgent
# from src.agents.sarsa_agent import SARSAAgent
# from src.agents.qlearning_agent import QLearningAgent
# from src.agents.dqn_agent import DQNAgent


def get_agent(agent_type: str, state_dim: int, action_dim: int):
    """创建指定类型的智能体"""
    agents = {
        'random': lambda: RandomAgent(state_dim, action_dim),
        # 'td': lambda: TDAgent(state_dim, action_dim),
        # 'sarsa': lambda: SARSAAgent(state_dim, action_dim),
        # 'qlearning': lambda: QLearningAgent(state_dim, action_dim),
        # 'dqn': lambda: DQNAgent(state_dim, action_dim)
    }
    return agents[agent_type]()


def play_episode(env, agent, render_delay: float = 0.05):
    """
    运行一个回合并渲染
    Args:
        env: 游戏环境
        agent: 智能体
        render_delay: 每步渲染的延迟时间（秒）
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # 渲染环境（现在会显示游戏窗口）
        env.render()
        time.sleep(render_delay)  # 添加延迟使动画更容易观察

        # 选择动作
        action = agent.select_action(state)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        state = next_state

        # 打印当前状态（可选）
        if steps % 20 == 0:  # 每20步打印一次，避免输出太多
            print(f"Step: {steps}, Total Reward: {total_reward:.2f}")

    return total_reward, steps


def main():
    # 配置
    AGENT_TYPE = 'random'  # 选择要运行的智能体类型
    NUM_EPISODES = 3  # 要运行的回合数
    RENDER_DELAY = 0.01  # 渲染延迟（秒）

    # 创建环境和智能体，设置render_mode='human'
    env = LunarLanderEnvironment(seed=42, render_mode='human')
    agent = get_agent(AGENT_TYPE, env.state_dim, env.action_dim)

    try:
        # 运行多个回合
        for episode in range(NUM_EPISODES):
            print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
            print("-" * 50)

            reward, steps = play_episode(env, agent, RENDER_DELAY)

            print(f"\nEpisode finished after {steps} steps")
            print(f"Total reward: {reward:.2f}")
            print("-" * 50)

            # 每回合之间稍作暂停
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n游戏被用户中断")
    finally:
        env.close()


if __name__ == "__main__":
    main()