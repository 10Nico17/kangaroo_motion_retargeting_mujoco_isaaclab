# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Kangaroo robot configuration with passive closed-chain leg joints."""

from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabPhysXParams,
    IsaacLabSimParams,
)
from protomotions.simulator.newton.config import NewtonSimParams


# Ordered exactly like the kinematic DOFs, excluding the four passive linkage
# joints.  Policy actions use this order; observations and motion states retain
# all 32 DOFs.
ACTUATED_DOF_NAMES = [
    "pelvis_1_joint",
    "pelvis_2_joint",
    *[f"arm_left_{i}_joint" for i in range(1, 8)],
    *[f"arm_right_{i}_joint" for i in range(1, 8)],
    "leg_left_1_joint",
    "leg_left_2_joint",
    "leg_left_3_joint",
    "leg_left_length_joint",
    "leg_left_4_joint",
    "leg_left_5_joint",
    "leg_right_1_joint",
    "leg_right_2_joint",
    "leg_right_3_joint",
    "leg_right_length_joint",
    "leg_right_4_joint",
    "leg_right_5_joint",
]

PASSIVE_DOF_NAMES = [
    "leg_left_femur_joint",
    "leg_left_knee_joint",
    "leg_right_femur_joint",
    "leg_right_knee_joint",
]


# qpos values from the MJCF ``init_state`` keyframe (root pose omitted).
DEFAULT_JOINT_POS = {
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
    "arm_left_1_joint": 0.24,
    "arm_left_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_left_4_joint": 0.8,
    "arm_left_5_joint": 0.0,
    "arm_left_6_joint": 0.0,
    "arm_left_7_joint": 0.0,
    "arm_right_1_joint": -0.24,
    "arm_right_2_joint": 1.32,
    "arm_right_3_joint": -1.57,
    "arm_right_4_joint": 0.8,
    "arm_right_5_joint": 0.0,
    "arm_right_6_joint": 0.0,
    "arm_right_7_joint": 0.0,
    "leg_left_1_joint": -0.012,
    "leg_left_2_joint": 0.054,
    "leg_left_3_joint": 0.04,
    "leg_left_length_joint": 0.6,
    "leg_left_4_joint": -0.053,
    "leg_left_5_joint": 0.0,
    "leg_left_femur_joint": 0.9,
    "leg_left_knee_joint": 1.8,
    "leg_right_1_joint": 0.012,
    "leg_right_2_joint": 0.054,
    "leg_right_3_joint": -0.04,
    "leg_right_length_joint": 0.6,
    "leg_right_4_joint": -0.053,
    "leg_right_5_joint": 0.0,
    "leg_right_femur_joint": 0.9,
    "leg_right_knee_joint": 1.8,
}


def _control_info(
    stiffness: float, damping: float, effort_limit: float
) -> ControlInfo:
    return ControlInfo(
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
    )


@dataclass
class KangarooRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["leg_left_foot_link"],
            "all_right_foot_bodies": ["leg_right_foot_link"],
            "all_left_hand_bodies": ["arm_left_7_link"],
            "all_right_hand_bodies": ["arm_right_7_link"],
            # Kangaroo has no separate head body.  The base/torso is the most
            # stable semantic substitute required by the shared interface.
            "head_body_name": ["base_link"],
            "torso_body_name": ["base_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "base_link",
            "arm_left_7_link",
            "arm_right_7_link",
            "leg_left_foot_link",
            "leg_right_foot_link",
        ]
    )

    default_root_height: float = 0.9
    default_dof_pos: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_POS)
    )
    actuated_dof_names: List[str] = field(
        default_factory=lambda: list(ACTUATED_DOF_NAMES)
    )
    anchor_body_name: str = "base_link"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name=(
                "Kangaroo/kangaroo_grippers_ias.xml"
            ),
            usd_asset_file_name=(
                "Kangaroo/usd/kangaroo_grippers_ias/"
                "kangaroo_grippers_ias_configured.usd"
            ),
            usd_bodies_root_prim_path=(
                "/World/envs/env_.*/Robot/base_link/"
            ),
            self_collisions=False,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "pelvis_.*_joint": _control_info(49.94, 3.179, 50.0),
                "arm_.*_[12]_joint": _control_info(49.94, 3.179, 50.0),
                "arm_.*_[3-7]_joint": _control_info(26.177, 1.666, 25.0),
                "leg_.*_1_joint": _control_info(100.0, 6.366, 80.0),
                "leg_.*_2_joint": _control_info(100.0, 6.366, 230.0),
                "leg_.*_3_joint": _control_info(100.0, 6.366, 139.0),
                "leg_.*_length_joint": _control_info(
                    1600.0, 101.859, 1100.0
                ),
                "leg_.*_4_joint": _control_info(30.0, 1.91, 140.0),
                "leg_.*_5_joint": _control_info(30.0, 1.91, 82.0),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1.0,
                ),
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
            ),
        )
    )
