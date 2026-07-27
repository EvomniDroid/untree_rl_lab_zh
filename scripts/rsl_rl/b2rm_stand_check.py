"""Verify that the B2RM flat-ground reset pose can stand without a policy.

The script sends zero residual actions only.  With the velocity task action
definition, that means every leg target stays exactly at the B2RM baseline
reset pose.
It is intentionally separate from training so a reset-pose/PD problem is not
mistaken for failed RL exploration.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Hold B2RM at its flat-ground reset pose with zero policy residual.")
parser.add_argument("--task", type=str, default="Unitree-B2RM-Velocity")
parser.add_argument("--duration", type=float, default=8.0, help="Check duration in seconds.")
parser.add_argument("--print-interval", type=float, default=0.25, help="Diagnostic print interval in seconds.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
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
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        base_sensor = env.unwrapped.scene["base_ground_contact"]
        robot = env.unwrapped.scene["robot"]
        action_dim = env.unwrapped.action_manager.total_action_dim
        actions = torch.zeros((1, action_dim), device=env.unwrapped.device)
        steps = max(1, round(args_cli.duration / env.unwrapped.step_dt))
        print_every = max(1, round(args_cli.print_interval / env.unwrapped.step_dt))

        print("Holding B2RM at the flat-ground reset pose with zero leg residual actions.")
        print(f"action_dim={action_dim}, dt={env.unwrapped.step_dt:.4f}s, steps={steps}")
        for name, actuator_cfg in env.unwrapped.cfg.scene.robot.actuators.items():
            print(
                f"[stand] actuator={name} type={actuator_cfg.class_type.__name__} "
                f"stiffness={actuator_cfg.stiffness} damping={actuator_cfg.damping} "
                f"delay=({getattr(actuator_cfg, 'min_delay', 0)},{getattr(actuator_cfg, 'max_delay', 0)})"
            )
        for step in range(steps):
            _, _, terminated, truncated, _ = env.step(actions)
            if step % print_every == 0 or bool((terminated | truncated).any().item()):
                root_pos = robot.data.root_pos_w[0]
                projected_gravity = robot.data.projected_gravity_b[0]
                base_force = torch.linalg.vector_norm(base_sensor.data.net_forces_w[0, 0]).item()
                print(
                    f"[stand] t={(step + 1) * env.unwrapped.step_dt:.2f}s "
                    f"root_z={root_pos[2].item():+.3f} "
                    f"gravity_b=({projected_gravity[0].item():+.2f},"
                    f"{projected_gravity[1].item():+.2f},{projected_gravity[2].item():+.2f}) "
                    f"base_ground_force={base_force:.1f}N "
                    f"terminated={bool(terminated.any().item())} "
                    f"truncated={bool(truncated.any().item())}"
                )
            if bool((terminated | truncated).any().item()):
                print("[stand] FAILED: the reset pose cannot hold under the current simulated PD/dynamics.")
                return

        print("[stand] PASSED: the reset pose held for the full check duration.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
