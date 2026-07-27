# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import pickle
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data):
    """Save a config pickle across IsaacLab versions that removed this helper."""
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def _run_play_mode():
    """Run a lightweight visualization-only loop.

    This intentionally bypasses Hydra because the shorthand override `env=play`
    would otherwise replace the `env` dict with a string and crash during `env_cfg.from_dict(...)`.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # Prefer the task's play config if it exists (usually fewer envs, simpler terrain, better defaults for GUI).
    try:
        play_cfg = load_cfg_from_registry(args_cli.task, "play_env_cfg_entry_point")
    except Exception:
        play_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    play_cfg.scene.num_envs = 1 if args_cli.num_envs is None else args_cli.num_envs
    play_cfg.sim.device = args_cli.device if args_cli.device is not None else play_cfg.sim.device
    # Make sure rendering updates regularly in GUI.
    if getattr(play_cfg.sim, "render_interval", None) is None or play_cfg.sim.render_interval <= 0:
        play_cfg.sim.render_interval = 1
    # In play mode, prefer realtime stepping if supported.
    if hasattr(play_cfg.sim, "use_fabric"):
        pass

    env = gym.make(args_cli.task, cfg=play_cfg)
    try:
        env.reset()
        # Kick the simulation once so that spawns and rendering settle.
        # IMPORTANT: IsaacLab manager-based env expects actions shaped (num_envs, total_action_dim).
        num_envs = int(getattr(play_cfg.scene, "num_envs", 1))
        action_dim = getattr(getattr(env.unwrapped, "action_manager", None), "total_action_dim", None)
        if action_dim is None:
            # Fallback to action_space shape (usually (action_dim,))
            act_shape = getattr(env.action_space, "shape", None)
            if act_shape is None:
                act_shape = torch.as_tensor(env.action_space.sample()).shape
            action_dim = int(act_shape[0])

        zero_action = torch.zeros((num_envs, int(action_dim)), device=play_cfg.sim.device, dtype=torch.float32)
        env.step(zero_action)
        while simulation_app.is_running():
            # IsaacLab env expects torch Tensor actions (it calls action.to(self.device)).
            # Gymnasium spaces typically return numpy arrays.
            action_np = env.action_space.sample()
            action = torch.as_tensor(action_np, dtype=torch.float32, device=play_cfg.sim.device)
            # Normalize action shape to (num_envs, action_dim).
            if action.ndim == 1:
                action = action.unsqueeze(0)
            # Some spaces may return shape (1, action_dim) even when num_envs==1.
            if action.shape[0] != num_envs:
                if action.shape[0] == 1:
                    action = action.repeat(num_envs, 1)
                else:
                    action = action[:num_envs]
            # Last line of defense: enforce correct width.
            if action.shape[1] != int(action_dim):
                # If space gives (num_envs, 1) but env expects (num_envs, action_dim), expand zeros.
                fixed = torch.zeros((num_envs, int(action_dim)), device=play_cfg.sim.device, dtype=torch.float32)
                w = min(int(action_dim), int(action.shape[1]))
                fixed[:, :w] = action[:, :w]
                action = fixed
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


# NOTE:
# Hydra 中的 shorthand override `env=play` 会把 env 从 dict 覆盖成 str，导致
# isaaclab_tasks/utils/hydra.py 里 env_cfg.from_dict(hydra_env_cfg["env"]) 崩溃。
# 因此如果检测到 `env=play`，我们直接绕过 Hydra，进入可视化 loop。
if any(arg.strip() == "env=play" for arg in hydra_args):
    _run_play_mode()
    simulation_app.close()
    raise SystemExit(0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
