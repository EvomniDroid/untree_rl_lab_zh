"""RSL-RL PPO settings matching the successful B2RM Parkour optimizer."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class B2RMParkourRewardPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 500
    experiment_name = "unitree_b2rm_velocity_parkour_reward"
    clip_actions = 20.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=7.5e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.97,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )
