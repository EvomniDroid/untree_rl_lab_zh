"""Flat-ground B2RM velocity task using the validated Parkour reward recipe."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.b2rm import ARM_FOLDED_POS, ARM_JOINT_NAMES, LEG_JOINT_NAMES
from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import B2RMSceneCfg, EventCfg, PARKOUR_LEG_POS


LEG_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
FEET_SENSOR_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    preserve_order=True,
)
FEET_BODY_CFG = SceneEntityCfg(
    "robot",
    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    preserve_order=True,
)


@configclass
class ParkourRewardCommandsCfg:
    """The flat-ground equivalent of the Parkour Perlin command range."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.8, 0.8),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


@configclass
class ParkourRewardActionsCfg:
    """Twelve learned leg targets plus a zero-dimensional fixed-arm term."""

    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.4,
        use_default_offset=True,
    )
    arm_hold = mdp.FixedJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        joint_positions=ARM_FOLDED_POS,
    )


@configclass
class ParkourRewardObservationsCfg:
    """The successful Parkour proprioceptive terms, without depth or gait phase."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": LEG_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-10.0, 10.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": LEG_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-50.0, 50.0),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-10.0, 10.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-20.0, 20.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEG_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": LEG_CFG})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ParkourRewardRewardsCfg:
    """Parkour locomotion rewards, excluding terrain-, vision- and arm-specific terms."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    heading_error = RewTerm(
        func=mdp.heading_error,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )
    dont_wait = RewTerm(
        func=mdp.dont_wait,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    must_turn = RewTerm(
        func=mdp.must_turn,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.05,
            "min_turn_rate": 0.05,
            "target_ratio": 0.6,
        },
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "offset": 0.0, "threshold": 0.05, "asset_cfg": LEG_CFG},
    )

    feet_air_time = RewTerm(
        func=mdp.parkour_feet_air_time,
        weight=0.5,
        params={"command_name": "base_velocity", "sensor_cfg": FEET_SENSOR_CFG, "vel_threshold": 0.15},
    )
    foot_contact_balance = RewTerm(
        func=mdp.foot_contact_balance,
        weight=-2.0,
        params={"sensor_cfg": FEET_SENSOR_CFG, "max_air_time": 1.0},
    )
    feet_air_time_balance = RewTerm(
        func=mdp.feet_air_time_balance,
        weight=-1.0,
        params={"command_name": "base_velocity", "sensor_cfg": FEET_SENSOR_CFG, "vel_threshold": 0.15},
    )
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.5,
        params={"sensor_cfg": FEET_SENSOR_CFG, "asset_cfg": FEET_BODY_CFG, "threshold": 1.0},
    )

    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    roll = RewTerm(func=mdp.roll_l1, weight=-2.0)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5, params={"asset_cfg": LEG_CFG})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-7.5e-7, params={"asset_cfg": LEG_CFG})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": LEG_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_pitch = RewTerm(func=mdp.positive_pitch_l2, weight=-2.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-4.0, params={"target_height": 0.55})
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.3, params={"asset_cfg": LEG_CFG})
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": LEG_CFG})

    feet_height = RewTerm(
        func=mdp.parkour_feet_height,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": FEET_BODY_CFG,
            "sensor_cfg": FEET_SENSOR_CFG,
            "target_height": 0.3,
            "vel_threshold": 0.15,
        },
    )
    feet_height_balance = RewTerm(
        func=mdp.feet_height_balance,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": FEET_BODY_CFG,
            "sensor_cfg": FEET_SENSOR_CFG,
            "max_height": 0.36,
        },
    )
    work = RewTerm(func=mdp.net_mechanical_work, weight=-0.003, params={"asset_cfg": LEG_CFG})
    delta_torques = RewTerm(func=mdp.delta_torques, weight=-1.0e-7, params={"asset_cfg": LEG_CFG})
    feet_jerk = RewTerm(func=mdp.feet_jerk, weight=-2.0e-4, params={"sensor_cfg": FEET_SENSOR_CFG})
    contact_forces = RewTerm(
        func=mdp.contact_forces_penalty,
        weight=-0.001,
        params={"threshold": 120.0, "sensor_cfg": FEET_SENSOR_CFG},
    )
    tracking_contacts_force = RewTerm(
        func=mdp.tracking_contacts_shaped_force,
        weight=-2.0,
        params={"command_name": "base_velocity", "sensor_cfg": FEET_SENSOR_CFG, "sigma": 0.5, "kappa": 0.07},
    )
    tracking_contacts_vel = RewTerm(
        func=mdp.tracking_contacts_shaped_vel,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": FEET_BODY_CFG,
            "sensor_cfg": FEET_SENSOR_CFG,
            "sigma": 0.5,
        },
    )
    walking_dof = RewTerm(
        func=mdp.walking_dof,
        weight=0.5,
        params={"command_name": "base_velocity", "asset_cfg": LEG_CFG},
    )


@configclass
class ParkourRewardTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_ground_contact", body_names="base_link"), "threshold": 1.0},
    )
    leg_link_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh"]), "threshold": 15.0},
    )
    calf_link_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf"]), "threshold": 50.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.3})


@configclass
class B2RMParkourRewardEnvCfg(ManagerBasedRLEnvCfg):
    """Standard IsaacLab/RSL-RL implementation of the proprioceptive baseline."""

    scene: B2RMSceneCfg = B2RMSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ParkourRewardObservationsCfg = ParkourRewardObservationsCfg()
    actions: ParkourRewardActionsCfg = ParkourRewardActionsCfg()
    commands: ParkourRewardCommandsCfg = ParkourRewardCommandsCfg()
    rewards: ParkourRewardRewardsCfg = ParkourRewardRewardsCfg()
    terminations: ParkourRewardTerminationsCfg = ParkourRewardTerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

        # Match the nominal pose stored with the stable Parkour checkpoint.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
        self.scene.robot.init_state.joint_pos = {
            **PARKOUR_LEG_POS,
            **ARM_FOLDED_POS,
        }
        self.scene.robot.init_state.joint_vel = {".*": 0.0}


@configclass
class B2RMParkourRewardPlayEnvCfg(B2RMParkourRewardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.debug_vis = True
