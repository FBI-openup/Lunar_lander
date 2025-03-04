"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 17:58
@Author        : FengD
@File          : e-greddy_agent
@Description   :
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray

from src.agents.base_agent import BaseAgent


class EGreedyAgent(BaseAgent):
    """所有智能体的基类"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def e_greedy_policy(self, state:ndarray, q_values:np.ndarray, epsilon:float):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(q_values[state])

    # DISPLAY_EVERY_N_EPISODES = 50

    # def q_learning(
    #         alpha: float = 0.1,
    #         alpha_factor: float = 0.9995,
    #         gamma: float = 0.99,
    #         epsilon: float = 0.5,
    #         display: bool = False) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    #     q_array_history = []
    #     alpha_history = []
    #     observation_space = cast(gym.spaces.Discrete, environment.observation_space)
    #     action_space = cast(gym.spaces.Discrete, environment.action_space)
    #     num_states = observation_space.n
    #     num_actions = action_space.n
    #     q_array = np.zeros([num_states, num_actions])
    #     for episode_index in tqdm(range(1, num_episodes)):
    #         if display and episode_index % DISPLAY_EVERY_N_EPISODES == 0:
    #             display_qtable(q_array, title="Q table")
    #         q_array_history.append(q_array.copy())
    #         alpha_history.append(alpha)
    #         if alpha_factor is not None:
    #             alpha = alpha * alpha_factor
    #         state, _ = environment.reset()
    #         terminated = False
    #         truncated = False
    #         while not (terminated or truncated):
    #             action = epsilon_greedy_policy(state, q_array, epsilon)
    #             next_state, reward, terminated, truncated, _ = environment.step(action)
    #             target = reward + gamma * np.max(q_array[next_state])
    #             td_error = target - q_array[state, action]
    #             q_array[state, action] = q_array[state, action] + alpha * td_error
    #             state = next_state
    #     return q_array, q_array_history, alpha_history

    @abstractmethod
    def select_action(self, q_values:np.ndarray, state: ndarray, epsilon:float) -> int:
        """选择动作"""
        return self.e_greedy_policy(state, q_values, epsilon)

    @abstractmethod
    def learn(self, *args, **kwargs):
        """学习更新"""
        pass