"""Minimal B2RM flat-ground velocity task."""

import gymnasium as gym


gym.register(
    id="Unitree-B2RM-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B2RMVelocityEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B2RMVelocityPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
