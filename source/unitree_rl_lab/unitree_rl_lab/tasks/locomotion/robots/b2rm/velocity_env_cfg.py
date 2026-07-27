"""Minimal, deterministic flat-ground velocity task for the B2RM."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.b2rm import ARM_FOLDED_POS, ARM_JOINT_NAMES, LEG_JOINT_NAMES, UNITREE_B2RM_CFG
from unitree_rl_lab.tasks.locomotion import mdp


@configclass
class B2RMSceneCfg(InteractiveSceneCfg):
    """Flat terrain, B2RM, contact sensing and a neutral light only."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # A global plane gives every cloned environment a unique grid origin.
        # A 1x1 terrain generator would assign all 1024 robots the same origin.
        terrain_type="plane",
        env_spacing=2.5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_B2RM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # This must be a separate, single-body sensor.  The all-body sensor above
    # also sees B2RM self-contacts (notably from the mounted arm), which must
    # never be treated as the base hitting the floor.
    base_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        filter_prim_paths_expr=["/World/ground/.*"],
        history_length=3,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.78, 0.85)),
    )


@configclass
class EventCfg:
    """Only deterministic reset; no pushes, mass/friction randomization or noise events."""

    reset_to_standing_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        # Stage 1 deliberately contains only meaningful forward commands. A
        # zero/yaw command mix lets a high-PD quadruped earn reward by merely
        # finding a static stance before it has learned a gait.
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 0.40), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.50, 0.80), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.60, 0.60)
        ),
    )


@configclass
class ActionsCfg:
    # Exactly twelve policy outputs: the four legs. The arm is driven by arm_hold below.
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        # Parkour uses a wide residual target range. +/-0.05 rad is enough
        # to find a static compensation pose, but not enough hip/thigh travel
        # for a meaningful gait under the official high-PD stand controller.
        scale=0.20,
        use_default_offset=True,
        # These are absolute joint-position limits after scale and offset.
        # In particular, a generic [-1, 1] limit clips target2's calf=-1.30
        # and encourages the policy to exploit saturation instead of gait.
        clip={
            ".*_hip_joint": (-0.45, 0.45),
            ".*_thigh_joint": (0.15, 1.20),
            ".*_calf_joint": (-1.85, -0.75),
        },
    )
    arm_hold = mdp.FixedJointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, joint_positions=ARM_FOLDED_POS
    )


LEG_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
FEET_CFG = SceneEntityCfg("contact_forces", body_names=".*_foot")
ORDERED_FEET_CFG = SceneEntityCfg(
    "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # A velocity policy must observe the velocity it is trying to track.
        # This mirrors the successful B2RM Parkour actor observation.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEG_CFG})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": LEG_CFG})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEG_CFG})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": LEG_CFG})
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, params={"asset_cfg": LEG_CFG})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # Flat-ground, deployable subset of the B2RM Parkour and Go2 velocity
    # objectives. It has no terrain, vision, random pushes, or contacts other
    # than feet/ground, but keeps the gait-quality constraints.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    forward_velocity_deficit = RewTerm(
        func=mdp.forward_velocity_deficit, weight=-3.0, params={"command_name": "base_velocity"}
    )
    forward_velocity_error = RewTerm(
        func=mdp.forward_velocity_error_l2, weight=-4.0, params={"command_name": "base_velocity"}
    )
    # Keep the policy mean in a finite, deployable action range. Action-rate
    # alone cannot prevent a constant but saturated output.
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.10)
    yaw_rate_error = RewTerm(
        func=mdp.yaw_rate_error_l2, weight=-2.0, params={"command_name": "base_velocity"}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.25)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4, params={"asset_cfg": LEG_CFG})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6, params={"asset_cfg": LEG_CFG})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5, params={"asset_cfg": LEG_CFG})
    mechanical_power = RewTerm(func=mdp.mechanical_power_l1, weight=-2.0e-4, params={"asset_cfg": LEG_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": LEG_CFG})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # The validated target2 PD stand settles at base_link z about 0.58 m.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.58})
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.15, params={"asset_cfg": LEG_CFG})
    diagonal_gait_contact = RewTerm(
        func=mdp.diagonal_gait_contact_penalty,
        weight=-1.0,
        params={"command_name": "base_velocity", "sensor_cfg": ORDERED_FEET_CFG},
    )
    max_foot_air_time = RewTerm(
        func=mdp.max_foot_air_time_penalty, weight=-2.0, params={"sensor_cfg": FEET_CFG, "max_air_time": 0.60}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "sensor_cfg": FEET_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.3})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_ground_contact", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class B2RMVelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: B2RMSceneCfg = B2RMSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class B2RMVelocityPlayEnvCfg(B2RMVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
