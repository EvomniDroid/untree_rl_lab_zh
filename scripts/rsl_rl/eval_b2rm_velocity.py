"""Evaluate a B2RM velocity checkpoint against one fixed command."""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a B2RM policy with a fixed body-frame velocity command.")
parser.add_argument("--task", type=str, default="Unitree-B2RM-Velocity")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt.")
parser.add_argument("--cmd-vx", type=float, default=0.30, help="Forward command in m/s.")
parser.add_argument("--cmd-vy", type=float, default=0.0, help="Lateral command in m/s.")
parser.add_argument("--cmd-wz", type=float, default=0.0, help="Yaw command in rad/s.")
parser.add_argument("--duration", type=float, default=20.0, help="Evaluation duration in seconds.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    command = env_cfg.commands.base_velocity
    command.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    command.ranges.lin_vel_y = (args_cli.cmd_vy, args_cli.cmd_vy)
    command.ranges.ang_vel_z = (args_cli.cmd_wz, args_cli.cmd_wz)
    command.resampling_time_range = (args_cli.duration + 1.0, args_cli.duration + 1.0)
    command.debug_vis = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    checkpoint = os.path.abspath(args_cli.checkpoint)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if agent_cfg.experiment_name == "":
        agent_cfg.experiment_name = args_cli.task.lower().replace("-", "_")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        # The current IsaacLab wrapper returns a batched TensorDict directly.
        # Indexing it with ``[0]`` strips the one-environment batch dimension,
        # causing RSL-RL to emit a 1-D action that the action manager rejects.
        obs = env.get_observations()
        steps = max(1, round(args_cli.duration / env.unwrapped.step_dt))
        print(
            f"Evaluating {os.path.basename(checkpoint)} for {args_cli.duration:.1f}s: "
            f"cmd=(vx={args_cli.cmd_vx:+.2f}, vy={args_cli.cmd_vy:+.2f}, wz={args_cli.cmd_wz:+.2f})"
        )
        for step in range(steps):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                if actions.ndim == 1:
                    actions = actions.unsqueeze(0)
                obs, _, dones, _ = env.step(actions)
            if step % max(1, round(1.0 / env.unwrapped.step_dt)) == 0:
                robot = env.unwrapped.scene["robot"]
                active_command = env.unwrapped.command_manager.get_command("base_velocity")[0]
                velocity = robot.data.root_lin_vel_b[0]
                yaw_rate = robot.data.root_ang_vel_b[0, 2]
                print(
                    f"[eval] t={(step + 1) * env.unwrapped.step_dt:5.1f}s "
                    f"cmd=({active_command[0].item():+.2f},{active_command[1].item():+.2f},"
                    f"{active_command[2].item():+.2f}) "
                    f"vel=({velocity[0].item():+.2f},{velocity[1].item():+.2f}) "
                    f"wz={yaw_rate.item():+.2f} action=[{actions.min().item():+.2f},"
                    f"{actions.max().item():+.2f}] reset={bool(dones.any().item())}"
                )
            if args_cli.real_time:
                sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
