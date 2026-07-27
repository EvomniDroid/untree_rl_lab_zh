# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # 每个环境每轮采样的步数，影响每轮训练数据量和显存占用
    max_iterations = 5000   # 最大训练迭代次数
    save_interval = 100     # 每隔多少轮保存一次模型
    experiment_name = ""    # 实验名称，默认与任务名一致
    empirical_normalization = False  # 是否使用经验归一化（如观测归一化）

    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                # 初始动作噪声标准差
        actor_hidden_dims=[512, 256, 128], # 策略网络隐藏层结构
        critic_hidden_dims=[512, 256, 128],# 价值网络隐藏层结构
        activation="elu",                 # 激活函数类型
    )

    # PPO算法相关参数
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # 价值损失的权重
        use_clipped_value_loss=True,   # 是否对价值损失裁剪
        clip_param=0.2,                # PPO裁剪参数，防止策略更新过大
        entropy_coef=0.01,             # 熵系数，鼓励探索
        num_learning_epochs=5,         # 每轮采样后PPO优化的epoch数
        num_mini_batches=4,            # 每轮训练分成的mini-batch数量
        learning_rate=1.0e-3,          # 学习率
        schedule="adaptive",          # 学习率调整策略
        gamma=0.99,                    # 奖励折扣因子
        lam=0.95,                      # GAE优势估计参数
        desired_kl=0.01,               # 期望KL散度（用于自适应学习率）
        max_grad_norm=1.0,             # 梯度裁剪最大范数
    )
