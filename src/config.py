"""
!/usr/bin/env python
-*- coding: utf-8 -*-
@CreateTime    : 2025-02-19 09:45
@Author        : FengD
@File          : config
@Description   :
"""


# src/config.py
class Config:
    """项目配置"""
    # 环境参数
    SEED = 42

    # 训练参数
    EPISODES = 1000
    MAX_STEPS = 1000

    # 算法参数
    class TD:
        LEARNING_RATE = 0.01
        GAMMA = 0.99

    class SARSA:
        LEARNING_RATE = 0.01
        GAMMA = 0.99
        EPSILON = 0.1

    # ... 其他算法的参数