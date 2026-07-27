"""Reproduce the Unitree B2 SDK2 low-level stand/down sequence in Isaac Lab.

This is a PD-position-control demo, not an RL policy.  It follows the SDK2
sequence: current pose -> target1 crouch -> target2 stand -> target3 down.
The B2RM arm stays at its fixed folded target throughout.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the B2 SDK2 stand/down PD sequence in Isaac Lab.")
parser.add_argument("--task", type=str, default="Unitree-B2RM-Velocity")
parser.add_argument("--crouch-seconds", type=float, default=1.0)
parser.add_argument("--stand-seconds", type=float, default=1.8)
parser.add_argument("--hold-seconds", type=float, default=2.0)
parser.add_argument("--down-seconds", type=float, default=1.8)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.assets.robots.b2rm import LEG_JOINT_NAMES
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# SDK motor order is FR, FL, RR, RL.  Store the pose by name so it is safely
# reordered to the policy/action order (FL, FR, RL, RR) below.
SDK_TARGET_1 = {
    "FR": (0.0, 1.36, -2.65),
    "FL": (0.0, 1.36, -2.65),
    "RR": (-0.2, 1.36, -2.65),
    "RL": (0.2, 1.36, -2.65),
}
SDK_TARGET_2 = {leg: (0.0, 0.67, -1.30) for leg in ("FR", "FL", "RR", "RL")}
SDK_TARGET_3 = {
    "FR": (-0.5, 1.36, -2.65),
    "FL": (0.5, 1.36, -2.65),
    "RR": (-0.5, 1.36, -2.65),
    "RL": (0.5, 1.36, -2.65),
}


def pose_tensor(pose: dict[str, tuple[float, float, float]], device: torch.device) -> torch.Tensor:
    """Return a single 12-D target in LEG_JOINT_NAMES order."""
    values: list[float] = []
    for joint_name in LEG_JOINT_NAMES:
        leg_name, joint_type, _ = joint_name.split("_")
        joint_index = {"hip": 0, "thigh": 1, "calf": 2}[joint_type]
        values.append(pose[leg_name][joint_index])
    return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # The SDK sample sends a fresh PD target at 500 Hz (dt=0.002 s).
    env_cfg.sim.dt = 0.002
    env_cfg.decimation = 1
    env_cfg.sim.render_interval = 1
    env_cfg.scene.contact_forces.update_period = env_cfg.sim.dt
    env_cfg.scene.base_ground_contact.update_period = env_cfg.sim.dt
    # Permit the exact SDK crouch/down poses. These limits apply after the
    # residual action has been converted back into a joint target.
    env_cfg.actions.leg_joint_pos.clip = {
        ".*_hip_joint": (-0.55, 0.55),
        ".*_thigh_joint": (0.0, 1.50),
        ".*_calf_joint": (-2.80, -0.50),
    }
    # A deliberate sit/down can touch the base. Do not auto-reset mid-demo.
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.bad_orientation = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        device = env.unwrapped.device
        action_scale = env_cfg.actions.leg_joint_pos.scale
        action_dim = env.unwrapped.action_manager.total_action_dim
        if action_dim != 12:
            raise RuntimeError(f"Expected 12 leg actions, got {action_dim}.")

        leg_action = env.unwrapped.action_manager.get_term("leg_joint_pos")
        leg_joint_ids = leg_action._joint_ids
        start_target = robot.data.joint_pos[:, leg_joint_ids].clone()
        target_1 = pose_tensor(SDK_TARGET_1, device)
        target_2 = pose_tensor(SDK_TARGET_2, device)
        target_3 = pose_tensor(SDK_TARGET_3, device)
        default_target = robot.data.default_joint_pos[:, leg_joint_ids]
        phases = (
            ("crouch", start_target, target_1, args_cli.crouch_seconds),
            ("stand", target_1, target_2, args_cli.stand_seconds),
            ("hold", target_2, target_2, args_cli.hold_seconds),
            ("down", target_2, target_3, args_cli.down_seconds),
        )

        print("B2RM SDK2-style PD stand/down demo")
        print("PD gains: Kp=1000, Kd=10; control_dt=0.002 s; arm=fixed folded pose")
        for phase_name, phase_start, phase_end, seconds in phases:
            steps = max(1, round(seconds / env.unwrapped.step_dt))
            print(f"[pd-demo] phase={phase_name} duration={seconds:.2f}s steps={steps}")
            for step in range(steps):
                alpha = min(1.0, (step + 1) / steps)
                joint_target = torch.lerp(phase_start, phase_end, alpha)
                # JointPositionAction consumes residuals around default_target.
                residual_action = (joint_target - default_target) / action_scale
                env.step(residual_action)
                if step % max(1, round(0.25 / env.unwrapped.step_dt)) == 0:
                    root_z = robot.data.root_pos_w[0, 2].item()
                    gravity = robot.data.projected_gravity_b[0]
                    print(
                        f"[pd-demo] {phase_name} t={(step + 1) * env.unwrapped.step_dt:.2f}s "
                        f"root_z={root_z:.3f} gravity_b=({gravity[0].item():+.2f},"
                        f"{gravity[1].item():+.2f},{gravity[2].item():+.2f})"
                    )
        print("[pd-demo] complete")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
