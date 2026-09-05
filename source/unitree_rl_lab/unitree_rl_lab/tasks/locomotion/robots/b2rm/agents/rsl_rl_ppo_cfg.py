"""PPO settings for the bounded-action B2RM flat velocity task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class B2RMVelocityPPORunnerCfg(BasePPORunnerCfg):
    """Use gentler exploration than the generic locomotion default."""

    # The action manager still applies per-joint absolute target limits. This
    # bound also keeps last_action observations and exported deployment inputs
    # consistent instead of allowing raw actor outputs to grow beyond +/-3.
    clip_actions = 2.0

    policy = RslRlPpoActorCriticCfg(
        # Policy-phase PD is intentionally much softer than the 1000/10
        # target2 stand controller. It therefore needs enough exploration to
        # discover a swing phase instead of converging to a static stance.
        init_noise_std=0.40,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Preserve moderate exploration while the first walking gait forms.
        # The bounded actions and soft policy PD keep this below the earlier
        # high-PD instability regime.
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
