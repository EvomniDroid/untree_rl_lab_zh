"""Export a trained B2RM RSL-RL actor checkpoint to a standalone ONNX policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_*.pt.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run>/exported_velocity.",
    )
    return parser.parse_args()


def build_actor(state_dict: dict[str, torch.Tensor]) -> torch.nn.Sequential:
    actor_state = {
        key.removeprefix("actor."): value.cpu()
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    weight_indices = sorted(
        int(key.split(".", maxsplit=1)[0]) for key in actor_state if key.endswith(".weight")
    )
    if not weight_indices:
        raise ValueError("Checkpoint does not contain actor linear layers.")

    modules: list[torch.nn.Module] = []
    for layer_number, index in enumerate(weight_indices):
        weight = actor_state[f"{index}.weight"]
        modules.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if layer_number < len(weight_indices) - 1:
            modules.append(torch.nn.ELU())

    actor = torch.nn.Sequential(*modules)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    return actor


def write_metadata(path: Path, checkpoint: Path, actor: torch.nn.Sequential) -> None:
    linear_layers = [module for module in actor if isinstance(module, torch.nn.Linear)]
    metadata = {
        "source_checkpoint": str(checkpoint.resolve()),
        "policy": "RSL-RL PPO actor MLP",
        "visual_input": False,
        "input_dim": linear_layers[0].in_features,
        "output_dim": linear_layers[-1].out_features,
        "control_dt": 0.02,
        "observation_order": [
            {"name": "base_lin_vel_body", "size": 3, "scale": 1.0},
            {"name": "base_ang_vel_body", "size": 3, "scale": 0.2},
            {"name": "projected_gravity_body", "size": 3, "scale": 1.0},
            {"name": "velocity_command", "size": 3, "scale": 1.0},
            {"name": "leg_joint_pos_relative", "size": 12, "scale": 1.0},
            {"name": "leg_joint_vel", "size": 12, "scale": 0.05},
            {"name": "last_raw_action", "size": 12, "scale": 1.0},
            {"name": "gait_phase_sin_cos", "size": 2, "scale": 1.0, "period": 0.6},
        ],
        "isaac_leg_order": [
            "FL_hip",
            "FR_hip",
            "RL_hip",
            "RR_hip",
            "FL_thigh",
            "FR_thigh",
            "RL_thigh",
            "RR_thigh",
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
        ],
        "default_leg_joint_pos": [0.0] * 4 + [0.67] * 4 + [-1.30] * 4,
        "action_scale": 0.2,
        "leg_kp": 1000.0,
        "leg_kd": 10.0,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint.parent / "exported_velocity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(checkpoint_data["model_state_dict"])
    first_layer = next(module for module in actor if isinstance(module, torch.nn.Linear))
    sample = torch.zeros(1, first_layer.in_features, dtype=torch.float32)
    onnx_path = output_dir / "policy.onnx"

    torch.onnx.export(
        actor,
        sample,
        onnx_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    write_metadata(output_dir / "metadata.json", checkpoint, actor)

    with torch.inference_mode():
        output = actor(sample).numpy()
    print(f"Exported: {onnx_path}")
    print(f"Actor: {first_layer.in_features} observations -> {output.shape[1]} leg actions")
    print(f"Zero-observation output: [{np.min(output):+.4f}, {np.max(output):+.4f}]")


if __name__ == "__main__":
    main()
