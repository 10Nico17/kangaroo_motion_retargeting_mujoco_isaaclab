# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Retarget semantic SOMA keypoints to the closed-chain Kangaroo MJCF.

The optimizer works directly on MuJoCo qpos.  This is intentional: a URDF-only
retargeter cannot represent Kangaroo's two equality-constrained leg loops.
"""

from pathlib import Path

import mujoco
import numpy as np
import typer
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm import tqdm


app = typer.Typer(pretty_exceptions_enable=False)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = (
    ROOT
    / "protomotions/data/assets/Kangaroo/kangaroo_grippers_ias.xml"
)

# Order matches the first 15 semantic keypoints produced by keypoint_utils.py.
TARGET_BODY_NAMES = [
    "base_link",
    "leg_left_1_link",
    "leg_right_1_link",
    "leg_left_knee_link",
    "leg_right_knee_link",
    "leg_left_5_link",
    "leg_right_5_link",
    "leg_left_foot_link",
    "leg_right_foot_link",
    "arm_left_2_link",
    "arm_right_2_link",
    "arm_left_4_link",
    "arm_right_4_link",
    "arm_left_7_link",
    "arm_right_7_link",
]

# End effectors and root matter most; intermediate keypoints guide limb shape.
POSITION_WEIGHTS = np.asarray(
    [4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0,
     1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
    dtype=np.float64,
)

# Semantic segments used by the G1 PyRoki retargeter.  Matching their
# directions makes the result much less dependent on the exact location of a
# robot link origin.
BONE_PAIRS = [
    (9, 11),   # left shoulder -> elbow
    (10, 12),  # right shoulder -> elbow
    (11, 13),  # left elbow -> wrist
    (12, 14),  # right elbow -> wrist
    (1, 3),    # left hip -> knee
    (2, 4),    # right hip -> knee
    (3, 5),    # left knee -> ankle
    (4, 6),    # right knee -> ankle
    (5, 7),    # left ankle -> foot
    (6, 8),    # right ankle -> foot
]

# Semantic tree in parent-before-child order. Motion changes are transferred
# along this tree while Kangaroo keeps its own rest-pose segment vectors.
SEMANTIC_TREE_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
    (5, 7), (6, 8),
    (0, 9), (0, 10),
    (9, 11), (10, 12),
    (11, 13), (12, 14),
]

# Track pelvis and wrist orientation. Source foot frames are bone-local and do
# not represent Kangaroo's sole orientation; sole flatness is handled by a
# dedicated world-up residual below.
ORIENTATION_INDICES = np.asarray([0, 13, 14], dtype=np.int32)
ORIENTATION_WEIGHTS = np.asarray([1.0, 0.15, 0.15])

# Fixed pairs keep the residual dimension constant for scipy least_squares.
# Adjacent bodies are deliberately omitted; those are allowed to touch around
# their physical joints and the closed linkage.
SELF_COLLISION_GEOM_PAIRS = [
    ("pelvis_2_collision", "arm_left_4_collision"),
    ("pelvis_2_collision", "arm_right_4_collision"),
    ("arm_left_4_collision", "arm_right_4_collision"),
    *[
        (left, right)
        for left in (
            "leg_left1_1_collision",
            "leg_left_femur_collision",
            "leg_left_knee_collision",
            "leg_left_ankle_collision",
        )
        for right in (
            "leg_right1_1_collision",
            "leg_right_femur_collision",
            "leg_right_knee_collision",
            "leg_right_ankle_collision",
        )
    ],
    *[
        (arm, leg)
        for arm in ("arm_left_4_collision", "arm_right_4_collision")
        for leg in (
            "leg_left_femur_collision",
            "leg_right_femur_collision",
            "leg_left_knee_collision",
            "leg_right_knee_collision",
        )
    ],
]


def _smooth_contacts(contacts: np.ndarray, window: int = 5) -> np.ndarray:
    """Convert ankle/toe flags to the cross-faded contact strength used by G1."""
    strength = contacts.mean(axis=1) if contacts.ndim == 2 else contacts
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(strength.astype(np.float64), kernel, mode="same")


def _stabilize_contact_targets(
    targets: np.ndarray, contacts: np.ndarray, foot_index: int
) -> None:
    """Keep a source foot target fixed inside each contiguous stance phase."""
    active = contacts > 0.2
    start = 0
    while start < len(active):
        if not active[start]:
            start += 1
            continue
        end = start + 1
        while end < len(active) and active[end]:
            end += 1
        weights = contacts[start:end]
        anchor = np.average(
            targets[start:end, foot_index], axis=0, weights=np.maximum(weights, 1e-6)
        )
        blend = contacts[start:end, None]
        targets[start:end, foot_index] = (
            targets[start:end, foot_index] * (1.0 - blend) + anchor * blend
        )
        start = end


def _rotation_residual(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) error vector from target to actual."""
    return Rotation.from_matrix(target.T @ actual).as_rotvec()


def _keyframe_id(model: mujoco.MjModel, name: str) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"MJCF keyframe not found: {name}")
    return key_id


def _joint_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(model.nv - 6, -np.inf, dtype=np.float64)
    upper = np.full(model.nv - 6, np.inf, dtype=np.float64)
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        dof_index = model.jnt_dofadr[joint_id] - 6
        if model.jnt_limited[joint_id]:
            lower[dof_index], upper[dof_index] = model.jnt_range[joint_id]
    return lower, upper


def _estimate_scale(
    source_first: np.ndarray, robot_first: np.ndarray
) -> float:
    # Locomotion quality is dominated by matching pelvis-to-foot reach.  Segment
    # ratios are misleading for Kangaroo because its passive linkage "knee" is
    # not located like a human serial-chain knee.
    source_drop = source_first[0, 2] - np.mean(source_first[[7, 8], 2])
    robot_drop = robot_first[0, 2] - np.mean(robot_first[[7, 8], 2])
    if source_drop <= 1.0e-5 or robot_drop <= 1.0e-5:
        return 1.0
    return float(np.clip(robot_drop / source_drop, 0.75, 1.1))


def _semantic_world_alignment(
    source_first: np.ndarray, target_root_rotation: np.ndarray
) -> np.ndarray:
    """Align source left/right and up axes with the robot world frame.

    SOMA pelvis orientation matrices use a bone-local convention whose yaw is
    not the semantic facing direction of the keypoint cloud. Using that matrix
    directly rotates the source's left/right axis onto Kangaroo's forward axis.
    Deriving yaw from hips and shoulders is both explicit and robust.
    """
    left_center = np.mean(source_first[[1, 9]], axis=0)
    right_center = np.mean(source_first[[2, 10]], axis=0)
    source_left = left_center - right_center
    source_left[2] = 0.0
    source_left /= max(np.linalg.norm(source_left), 1.0e-8)
    source_up = np.asarray([0.0, 0.0, 1.0])
    source_forward = np.cross(source_left, source_up)
    source_forward /= max(np.linalg.norm(source_forward), 1.0e-8)
    source_basis = np.column_stack([source_forward, source_left, source_up])
    return target_root_rotation @ source_basis.T


def _semantic_torso_frames(positions: np.ndarray) -> np.ndarray:
    """Build an upright root frame from the hip left/right axis.

    Kangaroo has no human spine. Shoulder motion must therefore not tilt the
    entire floating base; arms are retargeted independently. We retain source
    heading from the hips while keeping root roll and pitch upright.
    """
    frames = np.empty((len(positions), 3, 3), dtype=np.float64)
    for frame, points in enumerate(positions):
        left = points[1] - points[2]
        left[2] = 0.0
        left /= max(np.linalg.norm(left), 1.0e-8)
        up = np.asarray([0.0, 0.0, 1.0])
        forward = np.cross(left, up)
        forward /= max(np.linalg.norm(forward), 1.0e-8)
        left = np.cross(up, forward)
        frames[frame] = np.column_stack([forward, left, up])
    return frames


def _morph_to_robot_proportions(
    aligned_source: np.ndarray, robot_default: np.ndarray
) -> np.ndarray:
    """Transfer source segment direction changes onto Kangaroo proportions.

    At frame zero every semantic target exactly matches the robot's clean
    initial pose. Later frames rotate each robot rest-pose segment by the
    corresponding source segment's change relative to its first frame. This
    avoids forcing human shoulder height and human knee geometry onto the
    parallel-linkage robot.
    """
    targets = np.zeros(
        (len(aligned_source), len(TARGET_BODY_NAMES), 3), dtype=np.float64
    )
    targets[:, 0] = (
        robot_default[0][None, :]
        + aligned_source[:, 0]
        - aligned_source[0, 0][None, :]
    )
    for parent, child in SEMANTIC_TREE_EDGES:
        source_initial = aligned_source[0, child] - aligned_source[0, parent]
        robot_segment = robot_default[child] - robot_default[parent]
        if np.linalg.norm(source_initial) < 1.0e-8:
            targets[:, child] = targets[:, parent] + robot_segment
            continue
        for frame in range(len(aligned_source)):
            source_current = (
                aligned_source[frame, child] - aligned_source[frame, parent]
            )
            if np.linalg.norm(source_current) < 1.0e-8:
                rotated_segment = robot_segment
            else:
                delta_rotation, _ = Rotation.align_vectors(
                    source_current[None, :], source_initial[None, :]
                )
                rotated_segment = delta_rotation.apply(robot_segment)
            targets[frame, child] = targets[frame, parent] + rotated_segment
    return targets


@app.command()
def main(
    keypoints_file: Path = typer.Argument(..., exists=True),
    output_file: Path = typer.Argument(...),
    mjcf: Path = typer.Option(DEFAULT_MJCF, exists=True),
    max_frames: int = typer.Option(
        0, help="Limit frames for debugging; zero processes the complete motion."
    ),
    max_nfev: int = typer.Option(100, help="Maximum least-squares evaluations/frame."),
    constraint_weight: float = typer.Option(80.0),
    smoothness_weight: float = typer.Option(0.5),
    acceleration_weight: float = typer.Option(0.5),
    bone_alignment_weight: float = typer.Option(1.5),
    contact_weight: float = typer.Option(30.0),
    orientation_weight: float = typer.Option(1.0),
    root_orientation_weight: float = typer.Option(
        10.0, help="Keep robot root upright and aligned with the visible hip axis."
    ),
    sole_flatness_weight: float = typer.Option(
        8.0, help="Keep each foot sole normal aligned with world up."
    ),
    sole_height_weight: float = typer.Option(
        100.0, help="Keep stance sole capsules at ground-contact height."
    ),
    self_collision_weight: float = typer.Option(
        100.0, help="Penalty for selected non-adjacent collision geometries."
    ),
    self_collision_margin: float = typer.Option(
        0.005, help="Minimum separation for selected self-collision pairs."
    ),
    hand_clearance_weight: float = typer.Option(
        100.0, help="Penalty for virtual gripper clearance violations."
    ),
    hand_hand_margin: float = typer.Option(
        0.25, help="Minimum distance between left and right gripper origins."
    ),
    hand_torso_margin: float = typer.Option(
        0.30, help="Minimum distance from each gripper origin to torso center."
    ),
    hand_forearm_margin: float = typer.Option(
        0.18, help="Minimum distance from a gripper to the opposite forearm."
    ),
    velocity_limit_weight: float = typer.Option(
        30.0, help="Penalty for root/joint velocity above kinematic limits."
    ),
    pelvis_stability_weight: float = typer.Option(
        20.0, help="Keep the two internal pelvis joints near the clean init pose."
    ),
    fps: float = typer.Option(30.0, help="Motion frame rate used for velocity limits."),
    rest_weight: float = typer.Option(0.015),
) -> None:
    source = np.load(keypoints_file, allow_pickle=True).item()
    positions = np.asarray(source["positions"], dtype=np.float64)
    orientations = np.asarray(source["orientations"], dtype=np.float64)
    left_contacts = np.asarray(source["left_foot_contacts"], dtype=np.float64)
    right_contacts = np.asarray(source["right_foot_contacts"], dtype=np.float64)
    if max_frames > 0:
        positions = positions[:max_frames]
        orientations = orientations[:max_frames]
        left_contacts = left_contacts[:max_frames]
        right_contacts = right_contacts[:max_frames]

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    key_id = _keyframe_id(model, "init_state")
    default_qpos = model.key_qpos[key_id].copy()
    default_joints = default_qpos[7:].copy()
    joint_lower, joint_upper = _joint_bounds(model)

    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in TARGET_BODY_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        missing = [n for n, i in zip(TARGET_BODY_NAMES, body_ids) if i < 0]
        raise ValueError(f"MJCF bodies not found: {missing}")

    torso_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_2_link"
    )
    if torso_body_id < 0:
        raise ValueError("MJCF body not found: pelvis_2_link")

    collision_geom_pairs = []
    for first_name, second_name in SELF_COLLISION_GEOM_PAIRS:
        first_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, first_name
        )
        second_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, second_name
        )
        if first_id < 0 or second_id < 0:
            raise ValueError(
                f"Self-collision geometry missing: {first_name}, {second_name}"
            )
        collision_geom_pairs.append((first_id, second_id))

    # Root translation, root rotation-vector, then all scalar model joints.
    max_velocity = np.full(6 + model.nv - 6, 6.0, dtype=np.float64)
    max_velocity[:3] = 2.0  # m/s
    max_velocity[3:6] = 3.0  # rad/s
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
            max_velocity[6 + model.jnt_dofadr[joint_id] - 6] = 0.8  # m/s
    max_frame_delta = max_velocity / fps

    data.qpos[:] = default_qpos
    mujoco.mj_forward(model, data)
    robot_default_positions = data.xpos[body_ids].copy()
    robot_default_rotations = data.xmat[body_ids].reshape(-1, 3, 3).copy()
    scale = _estimate_scale(positions[0, :15], robot_default_positions)

    source_root_initial = positions[0, 0].copy()
    root_default_xyzw = np.asarray(
        [default_qpos[4], default_qpos[5], default_qpos[6], default_qpos[3]]
    )
    root_default_rotation = Rotation.from_quat(root_default_xyzw).as_matrix()
    # Align semantic forward/left/up axes. The SOMA pelvis orientation is a
    # bone-local frame and must not be used as the keypoint-cloud facing frame.
    world_alignment = _semantic_world_alignment(
        positions[0, :15], root_default_rotation
    )
    source_delta = positions - source_root_initial[None, None, :]
    aligned_source_positions = (
        default_qpos[:3][None, None, :]
        + np.einsum("ij,tkj->tki", world_alignment, source_delta) * scale
    )
    desired_positions = _morph_to_robot_proportions(
        aligned_source_positions[:, :15], robot_default_positions
    )
    desired_root = desired_positions[:, 0]

    aligned_source_rotations = np.einsum(
        "ij,tkjl->tkil", world_alignment, orientations
    )
    orientation_offsets = np.empty((15, 3, 3), dtype=np.float64)
    for index in range(15):
        orientation_offsets[index] = (
            aligned_source_rotations[0, index].T @ robot_default_rotations[index]
        )
    desired_orientations = np.einsum(
        "tkij,kjl->tkil", aligned_source_rotations[:, :15], orientation_offsets
    )
    semantic_torso_frames = _semantic_torso_frames(positions[:, :15])
    torso_alignment = root_default_rotation @ semantic_torso_frames[0].T
    desired_orientations[:, 0] = np.einsum(
        "ij,tjk->tik", torso_alignment, semantic_torso_frames
    )

    left_contact_strength = _smooth_contacts(left_contacts)
    right_contact_strength = _smooth_contacts(right_contacts)
    _stabilize_contact_targets(desired_positions, left_contact_strength, 7)
    _stabilize_contact_targets(desired_positions, right_contact_strength, 8)
    # The foot collision capsules have radius 1 cm and lie in the foot body's
    # local z=0 plane. Place the body center just above the floor in stance.
    for foot_index, strength in (
        (7, left_contact_strength),
        (8, right_contact_strength),
    ):
        desired_positions[:, foot_index, 2] = (
            desired_positions[:, foot_index, 2] * (1.0 - strength)
            + 0.012 * strength
        )

    frame_count = len(positions)
    root_positions = np.zeros((frame_count, 3), dtype=np.float64)
    root_quaternions = np.zeros((frame_count, 4), dtype=np.float64)
    joint_positions = np.zeros((frame_count, model.nv - 6), dtype=np.float64)
    constraint_errors = np.zeros(frame_count, dtype=np.float64)
    hand_separations = np.zeros(frame_count, dtype=np.float64)

    # x = root xyz, root rotation vector, all 32 scalar joint positions.
    x_previous = np.concatenate(
        [
            default_qpos[:3],
            Rotation.from_quat(
                [default_qpos[4], default_qpos[5], default_qpos[6], default_qpos[3]]
            ).as_rotvec(),
            default_joints,
        ]
    )
    x_previous_previous = x_previous.copy()
    previous_foot_positions = robot_default_positions[[7, 8]].copy()
    lower = np.concatenate(
        [np.full(3, -np.inf), np.full(3, -np.pi), joint_lower]
    )
    upper = np.concatenate(
        [np.full(3, np.inf), np.full(3, np.pi), joint_upper]
    )

    for frame in tqdm(range(frame_count), desc="Retargeting Kangaroo"):
        pelvis_quat_xyzw = Rotation.from_matrix(
            desired_orientations[frame, 0]
        ).as_quat()
        root_rotvec_target = Rotation.from_quat(pelvis_quat_xyzw).as_rotvec()
        contact_strength = np.asarray(
            [left_contact_strength[frame], right_contact_strength[frame]]
        )

        if frame == 0:
            x_previous[:3] = desired_root[frame]
            x_previous[3:6] = root_rotvec_target
            x_previous_previous = x_previous.copy()

        def residual(x: np.ndarray) -> np.ndarray:
            quat_xyzw = Rotation.from_rotvec(x[3:6]).as_quat()
            data.qpos[:3] = x[:3]
            data.qpos[3:7] = np.asarray(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )
            data.qpos[7:] = x[6:]
            mujoco.mj_forward(model, data)

            position_error = (
                data.xpos[body_ids] - desired_positions[frame, :15]
            ) * POSITION_WEIGHTS[:, None]

            # Match semantic bone directions as well as individual points.
            actual_points = data.xpos[body_ids]
            target_points = desired_positions[frame, :15]
            bone_errors = []
            for start, end in BONE_PAIRS:
                actual_bone = actual_points[end] - actual_points[start]
                target_bone = target_points[end] - target_points[start]
                actual_bone /= max(np.linalg.norm(actual_bone), 1.0e-8)
                target_bone /= max(np.linalg.norm(target_bone), 1.0e-8)
                bone_errors.append(actual_bone - target_bone)

            equality_mask = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            equality_error = data.efc_pos[: data.nefc][equality_mask]

            actual_rotations = data.xmat[body_ids].reshape(-1, 3, 3)
            orientation_errors = []
            for target_slot, body_index in enumerate(ORIENTATION_INDICES):
                weight = ORIENTATION_WEIGHTS[target_slot]
                if body_index == 0:
                    weight *= root_orientation_weight
                orientation_errors.append(
                    _rotation_residual(
                        actual_rotations[body_index],
                        desired_orientations[frame, body_index],
                    ) * weight
                )

            # The sole collision capsules lie in local XY, so local +Z is the
            # sole normal. Penalize only tilt, leaving foot yaw unconstrained.
            foot_normals = actual_rotations[[7, 8], :, 2]
            sole_weights = 0.15 + 3.85 * contact_strength
            sole_flatness = (
                foot_normals - np.asarray([0.0, 0.0, 1.0])[None, :]
            ) * sole_weights[:, None]

            from_to = np.zeros(6, dtype=np.float64)
            collision_distances = np.asarray(
                [
                    mujoco.mj_geomDistance(
                        model, data, first, second, 10.0, from_to
                    )
                    for first, second in collision_geom_pairs
                ]
            )
            self_collision = np.maximum(
                self_collision_margin - collision_distances, 0.0
            )

            # The MJCF has no collision geoms for the large grippers. Virtual
            # clearances prevent visually intersecting hands while still
            # allowing the optimizer to approximate impossible human targets.
            hand_positions = actual_points[[13, 14]]
            hand_hand_clearance = np.asarray(
                [
                    max(
                        hand_hand_margin
                        - np.linalg.norm(hand_positions[0] - hand_positions[1]),
                        0.0,
                    )
                ]
            )
            torso_rotation = data.xmat[torso_body_id].reshape(3, 3)
            torso_center = (
                data.xpos[torso_body_id]
                + torso_rotation @ np.asarray([0.0, 0.0, 0.23])
            )
            hand_torso_clearance = np.maximum(
                hand_torso_margin
                - np.linalg.norm(hand_positions - torso_center[None, :], axis=1),
                0.0,
            )
            opposite_forearms = actual_points[[12, 11]]
            hand_forearm_clearance = np.maximum(
                hand_forearm_margin
                - np.linalg.norm(hand_positions - opposite_forearms, axis=1),
                0.0,
            )
            hand_clearance = np.concatenate(
                [
                    hand_hand_clearance,
                    hand_torso_clearance,
                    hand_forearm_clearance,
                ]
            )

            velocity_excess = np.maximum(
                np.abs(x - x_previous) - max_frame_delta, 0.0
            )
            pelvis_stability = x[6:8] - default_joints[:2]

            foot_positions = actual_points[[7, 8]]
            contact_velocity = (
                foot_positions - previous_foot_positions
            ) * contact_strength[:, None]
            # The sole capsules have a 10 mm radius around the foot body's
            # local z=0 plane. Keep their centers just above that radius so a
            # flat stance touches instead of penetrating the ground.
            sole_height = (
                foot_positions[:, 2] - 0.011
            ) * contact_strength
            smoothness = (x - x_previous) * smoothness_weight
            acceleration = (
                x - 2.0 * x_previous + x_previous_previous
            ) * acceleration_weight
            rest = (x[6:] - default_joints) * rest_weight
            return np.concatenate(
                [
                    position_error.ravel(),
                    np.asarray(bone_errors).ravel() * bone_alignment_weight,
                    equality_error * constraint_weight,
                    np.asarray(orientation_errors).ravel() * orientation_weight,
                    sole_flatness.ravel() * sole_flatness_weight,
                    self_collision.ravel() * self_collision_weight,
                    hand_clearance.ravel() * hand_clearance_weight,
                    velocity_excess.ravel() * velocity_limit_weight,
                    pelvis_stability.ravel() * pelvis_stability_weight,
                    contact_velocity.ravel() * contact_weight,
                    sole_height.ravel() * sole_height_weight,
                    smoothness,
                    acceleration,
                    rest,
                ]
            )

        solution = least_squares(
            residual,
            np.clip(x_previous, lower, upper),
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1.0e-5,
            ftol=1.0e-5,
            gtol=1.0e-5,
        )
        residual(solution.x)  # refresh MuJoCo state for diagnostics
        previous_foot_positions = data.xpos[body_ids[[7, 8]]].copy()
        x_previous_previous = x_previous.copy()
        x_previous = solution.x

        quat_xyzw = Rotation.from_rotvec(solution.x[3:6]).as_quat()
        root_positions[frame] = solution.x[:3]
        root_quaternions[frame] = [
            quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        ]
        joint_positions[frame] = solution.x[6:]
        equality_mask = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        equality = data.efc_pos[: data.nefc][equality_mask]
        constraint_errors[frame] = np.max(np.abs(equality), initial=0.0)
        hand_separations[frame] = np.linalg.norm(
            data.xpos[body_ids[13]] - data.xpos[body_ids[14]]
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        base_frame_pos=root_positions,
        base_frame_wxyz=root_quaternions,
        joint_angles=joint_positions,
        left_foot_contacts=np.asarray(source["left_foot_contacts"])[:frame_count],
        right_foot_contacts=np.asarray(source["right_foot_contacts"])[:frame_count],
        source_scale=np.asarray(scale),
        max_constraint_error=np.asarray(constraint_errors.max()),
        min_hand_separation=np.asarray(hand_separations.min()),
    )
    print(f"Source-to-Kangaroo scale: {scale:.4f}")
    print(
        "Maximum equality residual: "
        f"{constraint_errors.max() * 1000.0:.3f} mm"
    )
    print(f"Minimum hand separation: {hand_separations.min():.3f} m")
    print(f"Wrote retargeted trajectory: {output_file}")


if __name__ == "__main__":
    app()
