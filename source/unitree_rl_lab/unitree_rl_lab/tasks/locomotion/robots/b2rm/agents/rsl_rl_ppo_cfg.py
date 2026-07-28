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
        # With Kp=1000 and action_scale=0.20, std=0.35 already produces
        # roughly 70 Nm RMS exploratory corrections. Start conservatively so
        # the policy can discover a stable gait instead of saturating joints.
        init_noise_std=0.15,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Do not reward increasing action noise. The previous run grew from
        # std=0.35 to about 0.91 and terminated on orientation every episode.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
