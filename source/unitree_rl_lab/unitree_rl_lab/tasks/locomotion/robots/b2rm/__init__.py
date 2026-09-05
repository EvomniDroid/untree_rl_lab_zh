"""Minimal B2RM flat-ground velocity task."""

import gymnasium as gym


gym.register(
    id="Unitree-B2RM-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B2RMVelocityEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B2RMVelocityPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:B2RMVelocityPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-B2RM-Velocity-ParkourPose",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B2RMVelocityParkourPoseEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B2RMVelocityParkourPosePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:B2RMVelocityPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-B2RM-Velocity-ParkourReward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_reward_env_cfg:B2RMParkourRewardEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.parkour_reward_env_cfg:B2RMParkourRewardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_parkour_reward_ppo_cfg:B2RMParkourRewardPPORunnerCfg"
        ),
    },
)
