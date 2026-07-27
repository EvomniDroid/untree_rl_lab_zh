"""Action terms for joints that must be held outside the learned policy."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class FixedJointPositionAction(ActionTerm):
    """Continuously hold selected joints at fixed targets without policy actions.

    The term deliberately has zero action dimensions.  It lets a locomotion policy
    control only the legs while, for example, a mounted arm remains in a known
    folded pose and still receives its own PD target every physics step.
    """

    def __init__(self, cfg: FixedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if not self._joint_ids:
            raise ValueError(f"FixedJointPositionAction matched no joints: {cfg.joint_names}")

        missing_targets = [name for name in self._joint_names if name not in cfg.joint_positions]
        if missing_targets:
            raise ValueError(f"Missing fixed targets for joints: {missing_targets}")

        target = torch.tensor(
            [cfg.joint_positions[name] for name in self._joint_names], device=self.device, dtype=torch.float
        )
        self._targets = target.unsqueeze(0).repeat(self.num_envs, 1)
        self._empty_actions = torch.zeros((self.num_envs, 0), device=self.device)
        # A zero-width action is an implementation detail, not a deployable policy I/O.
        self._export_IO_descriptor = False

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty_actions

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != 0:
            raise ValueError("FixedJointPositionAction expects a zero-dimensional action tensor.")

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)


@configclass
class FixedJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`FixedJointPositionAction`."""

    class_type: type[ActionTerm] = FixedJointPositionAction
    joint_names: list[str] = MISSING
    joint_positions: dict[str, float] = MISSING
