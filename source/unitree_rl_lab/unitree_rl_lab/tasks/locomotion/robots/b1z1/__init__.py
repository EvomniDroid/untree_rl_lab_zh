import gymnasium as gym


gym.register(
	id="Unitree-B1Z1-Velocity",
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B1Z1VelocityEnvCfg",
		"play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:B1Z1VelocityPlayEnvCfg",
		"rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
	},
)
