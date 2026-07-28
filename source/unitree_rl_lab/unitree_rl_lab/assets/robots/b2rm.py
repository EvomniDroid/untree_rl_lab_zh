"""B2RM asset configuration shared by the minimal velocity task."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg


# Keep using the exact URDF and mesh tree that produced the existing B2RM policy.
# Override this for a copied/deployed asset tree with UNITREE_B2RM_URDF_PATH.
B2RM_URDF_PATH = os.environ.get(
    "UNITREE_B2RM_URDF_PATH",
    "/home/zh/isaac/instinctlab/source/instinctlab/instinctlab/assets/resources/unitree_b2rm/urdf/b2rm.urdf",
)

LEG_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

ARM_JOINT_NAMES = [
    "arm_joint_1", "arm_joint_2", "arm_joint_3",
    "arm_joint_4", "arm_joint_5", "arm_joint_6",
]

# The same folded arm pose used by the current InstinctLab B2RM asset.
ARM_FOLDED_POS = {
    "arm_joint_1": 0.0,
    "arm_joint_2": 1.5707963267948966,
    "arm_joint_3": -3.036872898470133,
    "arm_joint_4": 0.0,
    "arm_joint_5": -0.15707963267948966,
    "arm_joint_6": 0.017453292519943295,
}

# Exact B2 SDK ``targetPos_2``: the real-robot stand demo reaches this pose
# before handing locomotion to a higher-level controller.
B2_TARGET_POS_2_LEG_POS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.67,
    "FL_calf_joint": -1.30,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.67,
    "FR_calf_joint": -1.30,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.67,
    "RL_calf_joint": -1.30,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.67,
    "RR_calf_joint": -1.30,
}


UNITREE_B2RM_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=B2RM_URDF_PATH,
        fix_base=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        # Disable URDF importer drives. The explicit actuator groups below own
        # all B2RM PD gains and keep train/deploy values visible in one place.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # The B2RM targetPos_2 zero-action check settles at base_link z=0.63 m.
    # Spawn at that measured equilibrium to avoid a reset-time ground impulse.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={
            **B2_TARGET_POS_2_LEG_POS,
            **ARM_FOLDED_POS,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Policy-phase gains match the converged InstinctLab B2RM Parkour
        # controller. Deployment uses 1000/10 only for target2 stand-up, then
        # blends to these gains before enabling learned residual actions.
        "legs_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"], effort_limit=200.0, velocity_limit=23.0,
            stiffness=99.09843, damping=6.30880, armature=0.025101925, min_delay=0, max_delay=0,
        ),
        "legs_thigh": DelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint"], effort_limit=200.0, velocity_limit=23.0,
            stiffness=40.17924, damping=2.55789, armature=0.010177520, min_delay=0, max_delay=0,
        ),
        "legs_calf": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"], effort_limit=320.0, velocity_limit=14.0,
            stiffness=99.09843, damping=6.30880, armature=0.025101925, min_delay=0, max_delay=0,
        ),
        # The arm is not a policy action. Keep its folded target firmly fixed.
        "arm_hold": DelayedPDActuatorCfg(
            joint_names_expr=["arm_joint_.*"],
            effort_limit={
                "arm_joint_1": 60.0, "arm_joint_2": 60.0,
                "arm_joint_3": 30.0, "arm_joint_4": 30.0,
                "arm_joint_5": 10.0, "arm_joint_6": 10.0,
            },
            velocity_limit=3.14,
            stiffness={
                "arm_joint_1": 60.0, "arm_joint_2": 60.0,
                "arm_joint_3": 50.0, "arm_joint_4": 50.0,
                "arm_joint_5": 40.0, "arm_joint_6": 40.0,
            },
            damping={
                "arm_joint_1": 6.0, "arm_joint_2": 6.0,
                "arm_joint_3": 5.0, "arm_joint_4": 5.0,
                "arm_joint_5": 4.0, "arm_joint_6": 4.0,
            },
            armature=0.001,
            min_delay=0,
            max_delay=0,
        ),
    },
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        *ARM_JOINT_NAMES,
    ],
)
