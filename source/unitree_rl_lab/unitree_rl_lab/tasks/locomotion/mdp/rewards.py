from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from the nominal pose while velocity commands are near zero."""
    asset: Articulation = env.scene[asset_cfg.name]

    dof_error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    is_standing_command = (torch.norm(command[:, :2], dim=1) < threshold) & (
        torch.abs(command[:, 2]) < threshold
    )
    return (dof_error - offset) * is_standing_command.float()


def dont_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when a meaningful forward command is present."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    velocity_x = asset.data.root_lin_vel_b[:, 0]
    return (command_x > 0.2) * (
        (velocity_x < 0.2).float() + (velocity_x < 0.0).float() + (velocity_x < -0.15).float()
    )


def forward_velocity_deficit(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the normalized forward-speed shortfall for nonzero forward commands.

    Unlike an exponential tracking reward, this keeps a standing robot costly
    even when the requested speed is modest. It is used only by the first
    B2RM gait-learning stage, where every command requests forward motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    target_speed = torch.clamp(command_x, min=0.0)
    shortfall = torch.clamp(target_speed - asset.data.root_lin_vel_b[:, 0], min=0.0)
    return torch.where(target_speed > 1.0e-3, shortfall / target_speed, torch.zeros_like(target_speed))


def forward_velocity_error_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return squared body-frame forward-speed tracking error."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(asset.data.root_lin_vel_b[:, 0] - command_x)


def lateral_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize uncommanded body-frame lateral velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def max_foot_air_time_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air_time: float = 0.60
) -> torch.Tensor:
    """Penalize any foot remaining airborne long enough to create a three-leg gait."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.square(torch.clamp(air_time - max_air_time, min=0.0)), dim=1)


def diagonal_gait_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Favor two-foot diagonal support and penalize synchronized hopping.

    Foot order is explicitly configured as FL, FR, RL, RR. FL/RR and FR/RL
    are the allowed diagonal support pairs; front, rear, and same-side pairs
    are penalized. A contact count far from two also penalizes all-feet flight
    and all-feet impact without imposing an episode-time gait clock.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    contact = (torch.linalg.vector_norm(forces, dim=-1) > contact_force_threshold).float()

    non_diagonal_contact = (
        contact[:, 0] * contact[:, 1]  # front pair: FL-FR
        + contact[:, 2] * contact[:, 3]  # rear pair: RL-RR
        + contact[:, 0] * contact[:, 2]  # left pair: FL-RL
        + contact[:, 1] * contact[:, 3]  # right pair: FR-RR
    )
    contact_count = torch.sum(contact, dim=1)
    count_error = torch.square(torch.clamp(torch.abs(contact_count - 2.0) - 0.5, min=0.0))
    has_forward_command = env.command_manager.get_command(command_name)[:, 0] > 0.05
    return (non_diagonal_contact + count_error) * has_forward_command.float()


def mechanical_power_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize absolute leg mechanical power; synchronized jumps are costly."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * joint_velocity), dim=1)


def yaw_rate_error_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize yaw-rate error directly instead of only reducing a reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_command = env.command_manager.get_command(command_name)[:, 2]
    return torch.square(asset.data.root_ang_vel_b[:, 2] - yaw_command)


def must_turn(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.05,
    min_turn_rate: float = 0.05,
    target_ratio: float = 0.6,
) -> torch.Tensor:
    """Penalize failure to turn in the commanded yaw direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_command = env.command_manager.get_command(command_name)[:, 2]
    signed_turn_rate = torch.sign(yaw_command) * asset.data.root_ang_vel_b[:, 2]
    target_rate = torch.maximum(
        torch.full_like(yaw_command, min_turn_rate), target_ratio * torch.abs(yaw_command)
    )
    penalty = torch.clamp(target_rate - signed_turn_rate, min=0.0) / torch.clamp(target_rate, min=1.0e-6)
    return (torch.abs(yaw_command) > cmd_threshold).float() * penalty


def heading_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize absolute yaw-rate tracking error."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.abs(asset.data.root_ang_vel_b[:, 2] - command[:, 2])


def parkour_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    vel_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward controlled swing time while discouraging one foot staying airborne."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    swing_air_time = torch.where(in_contact, torch.zeros_like(air_time), air_time)
    num_contact = torch.sum(in_contact.int(), dim=1)
    has_swing = torch.logical_and(num_contact > 0, num_contact < in_contact.shape[1])
    reward = torch.mean(swing_air_time, dim=1) * has_swing.float()

    max_swing = torch.max(air_time, dim=1).values
    mean_swing = torch.mean(air_time, dim=1)
    asymmetry = torch.clamp(max_swing - 1.5 * (mean_swing + 0.1), min=0.0) ** 2
    reward = reward - 0.3 * asymmetry

    command = env.command_manager.get_command(command_name)
    has_command = torch.logical_or(
        torch.norm(command[:, :2], dim=1) > vel_threshold,
        torch.abs(command[:, 2]) > vel_threshold,
    )
    return reward * has_command.float()


def foot_contact_balance(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air_time: float = 0.5
) -> torch.Tensor:
    """Require every foot to return to contact within a bounded time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(air_time - max_air_time, min=0.0) ** 2, dim=1)


def feet_air_time_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    vel_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize unequal air time between the two diagonal pairs."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    diagonal_a = 0.5 * (air_time[:, 0] + air_time[:, 3])
    diagonal_b = 0.5 * (air_time[:, 1] + air_time[:, 2])
    error = torch.square(diagonal_a - diagonal_b)

    command = env.command_manager.get_command(command_name)
    linear_command = torch.norm(command[:, :2], dim=1)
    forward = linear_command > vel_threshold
    turning = torch.abs(command[:, 2]) > vel_threshold
    gate = torch.where(
        forward & ~turning,
        torch.ones_like(linear_command),
        torch.where(
            turning & ~forward,
            torch.full_like(linear_command, 0.2),
            torch.full_like(linear_command, 0.5),
        ),
    )
    return error * gate * torch.logical_or(forward, turning).float()


def contact_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize planar foot velocity while the foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_velocity.norm(dim=-1) * contacts, dim=1)


def roll_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute base roll, matching the Parkour B2RM objective."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.abs(roll)


def positive_pitch_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize backward base pitch while allowing slight forward lean."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(torch.clamp(pitch, min=0.0))


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def parkour_feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.3,
    base_to_ground_height: float = 0.4,
    vel_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward swing feet near the clearance used by the B2RM Parkour task."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    foot_height = foot_z - (base_z - base_to_ground_height)
    swing_height = torch.where(in_contact, torch.zeros_like(foot_height), foot_height)
    reward = torch.mean(torch.exp(-torch.square(swing_height - target_height) / 0.04), dim=1)

    command = env.command_manager.get_command(command_name)
    has_command = torch.logical_or(
        torch.norm(command[:, :2], dim=1) > vel_threshold,
        torch.abs(command[:, 2]) > vel_threshold,
    )
    return reward * has_command.float()


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.10,
    std: float = 0.04,
) -> torch.Tensor:
    """Reward airborne feet for reaching a useful flat-ground clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    swing = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    per_foot_reward = torch.exp(-torch.square(foot_height - target_height) / (std * std))
    swing_count = torch.clamp(torch.sum(swing.float(), dim=1), min=1.0)
    reward = torch.sum(per_foot_reward * swing.float(), dim=1) / swing_count

    command = env.command_manager.get_command(command_name)
    has_command = torch.logical_or(
        torch.norm(command[:, :2], dim=1) > 0.1,
        torch.abs(command[:, 2]) > 0.1,
    )
    has_swing = torch.any(swing, dim=1)
    return reward * has_command.float() * has_swing.float()


def feet_height_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    max_height: float = 0.36,
    base_to_ground_height: float = 0.4,
) -> torch.Tensor:
    """Penalize diagonal swing-height asymmetry and excessive clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    swing = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    foot_height = foot_z - (base_z - base_to_ground_height)

    symmetry_error = torch.zeros(env.num_envs, device=env.device)
    for first, second in ((0, 3), (1, 2)):
        pair_swing = swing[:, first] & swing[:, second]
        symmetry_error += torch.square(foot_height[:, first] - foot_height[:, second]) * pair_swing.float()

    excessive_height = torch.clamp(foot_height - max_height, min=0.0)
    height_error = torch.sum(torch.square(excessive_height) * swing.float(), dim=1)

    command = env.command_manager.get_command(command_name)
    linear_command = torch.norm(command[:, :2], dim=1)
    forward = linear_command > 0.1
    turning = torch.abs(command[:, 2]) > 0.1
    gate = torch.where(
        forward & ~turning,
        torch.ones_like(linear_command),
        torch.where(
            turning & ~forward,
            torch.full_like(linear_command, 0.2),
            torch.full_like(linear_command, 0.5),
        ),
    )
    return (symmetry_error + height_error) * gate * torch.logical_or(forward, turning).float()


def net_mechanical_work(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize absolute net joint power, as in the Parkour task."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.abs(torch.sum(power, dim=1))


class delta_torques(ManagerTermBase):
    """Penalize step-to-step changes in commanded joint torque."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[asset_cfg.name]
        self._last_torques = torch.zeros_like(self.asset.data.applied_torque[:, asset_cfg.joint_ids])

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        current = self.asset.data.applied_torque[:, asset_cfg.joint_ids]
        penalty = torch.sum(torch.square(current - self._last_torques), dim=1)
        self._last_torques[:] = current
        return penalty

    def reset(self, env_ids):
        self._last_torques[env_ids] = 0.0


class feet_jerk(ManagerTermBase):
    """Penalize step-to-step changes in foot contact force."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]
        self.sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._last_forces = torch.zeros(env.num_envs, len(sensor_cfg.body_ids), 3, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        forces = self.sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
        penalty = torch.sum(torch.norm(forces - self._last_forces, dim=-1), dim=1)
        self._last_forces[:] = forces
        return penalty

    def reset(self, env_ids):
        self._last_forces[env_ids] = 0.0


def contact_forces_penalty(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize foot contact force above a fixed threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    excess = torch.clamp(torch.norm(forces, dim=-1) - threshold, min=0.0)
    return torch.sum(excess, dim=1)


def tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    sigma: float = 0.5,
    kappa: float = 0.07,
) -> torch.Tensor:
    """Favor diagonal support and smooth changes in mean foot force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    normalized_force = torch.clamp(torch.norm(forces, dim=-1) / 120.0, 0.0, 1.0)

    non_diagonal_error = (
        normalized_force[:, 0] * normalized_force[:, 1]
        + normalized_force[:, 0] * normalized_force[:, 2]
        + normalized_force[:, 1] * normalized_force[:, 3]
        + normalized_force[:, 2] * normalized_force[:, 3]
    )
    mean_force = torch.mean(normalized_force, dim=1)
    mean_error = torch.clamp(torch.abs(mean_force - 0.5) - sigma, min=0.0) ** 2

    if not hasattr(env, "_b2rm_last_contact_mean"):
        env._b2rm_last_contact_mean = torch.zeros_like(mean_force)
    phase_jitter = torch.square(mean_force - env._b2rm_last_contact_mean)
    env._b2rm_last_contact_mean = (
        (1.0 - kappa) * env._b2rm_last_contact_mean + kappa * mean_force
    )

    command = env.command_manager.get_command(command_name)
    linear_command = torch.norm(command[:, :2], dim=1)
    forward = linear_command > 0.1
    turning = torch.abs(command[:, 2]) > 0.1
    gate = torch.where(
        forward & ~turning,
        torch.ones_like(linear_command),
        torch.where(
            turning & ~forward,
            torch.full_like(linear_command, 0.3),
            torch.full_like(linear_command, 0.7),
        ),
    )
    return (non_diagonal_error + mean_error + phase_jitter) * gate * torch.logical_or(forward, turning).float()


def tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Keep stance feet still and require swing feet to move."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    in_contact = torch.norm(forces, dim=-1) > 1.0
    foot_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    swing_penalty = torch.clamp(
        -torch.where(in_contact, torch.zeros_like(foot_speed), foot_speed - sigma), min=0.0
    )
    stance_penalty = torch.where(in_contact, foot_speed, torch.zeros_like(foot_speed))

    command = env.command_manager.get_command(command_name)
    has_command = torch.logical_or(
        torch.norm(command[:, :2], dim=1) > 0.1,
        torch.abs(command[:, 2]) > 0.1,
    )
    return torch.sum(swing_penalty + stance_penalty, dim=1) * has_command.float()


def walking_dof(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    vel_threshold: float = 0.15,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Reward a compact gait near the nominal standing pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    has_command = torch.logical_or(
        torch.norm(command[:, :2], dim=1) > vel_threshold,
        torch.abs(command[:, 2]) > vel_threshold,
    )
    return torch.exp(-sigma * error) * has_command.float()


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
