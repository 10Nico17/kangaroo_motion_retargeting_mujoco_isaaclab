"""Visualize SOMA23-to-Kangaroo retargeting correspondences in MuJoCo."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import signal
import threading
import time
import xml.etree.ElementTree as ET
import tempfile

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


MAPPINGS = [
    ("pelvis", "base_link"),
    ("left_hip", "leg_left_1_link"),
    ("right_hip", "leg_right_1_link"),
    ("left_knee", "leg_left_knee_link"),
    ("right_knee", "leg_right_knee_link"),
    ("left_ankle", "leg_left_5_link"),
    ("right_ankle", "leg_right_5_link"),
    ("left_foot", "leg_left_foot_link"),
    ("right_foot", "leg_right_foot_link"),
    ("left_shoulder", "arm_left_2_link"),
    ("right_shoulder", "arm_right_2_link"),
    ("left_elbow", "arm_left_4_link"),
    ("right_elbow", "arm_right_4_link"),
    ("left_wrist", "arm_left_7_link"),
    ("right_wrist", "arm_right_7_link"),
]

# Every semantic keypoint contributes a direct optimization residual.
MATCH_ENABLED = np.asarray(
    [True, True, True, True, True, True, True, True, True,
     True, True, True, True, True, True],
    dtype=bool,
)

SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (3, 5), (5, 7),
    (2, 4), (4, 6), (6, 8),
    (0, 9), (0, 10), (9, 10),
    (9, 11), (11, 13),
    (10, 12), (12, 14),
]

SEMANTIC_TREE_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
    (5, 7), (6, 8),
    (0, 9), (0, 10),
    (9, 11), (10, 12),
    (11, 13), (12, 14),
]

DEFAULT_MODEL = Path(
    "protomotions/data/assets/Kangaroo/kangaroo_grippers_ias.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--retargeted", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Display the animated source skeleton beside the Kangaroo.",
    )
    parser.add_argument("--start-paused", action="store_true")
    return parser.parse_args()


def load_model(path: Path) -> mujoco.MjModel:
    """Add a display floor without changing the source MJCF."""
    path = path.resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is not None:
        ET.SubElement(
            worldbody,
            "geom",
            name="mapping_floor",
            type="plane",
            size="10 10 0.1",
            rgba="0.18 0.20 0.22 1",
            contype="0",
            conaffinity="0",
        )
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".xml", prefix="mapping_", dir=path.parent, delete=False
    ) as temporary:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        temporary_path = Path(temporary.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        temporary_path.unlink(missing_ok=True)

    # Kangaroo visual meshes use groups 1 and 2. Group 3 contains the collision
    # primitives that otherwise obscure the real body.
    model.geom_rgba[model.geom_group == 3, 3] = 0.0
    return model


def keyframe_id(model: mujoco.MjModel, name: str) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"MJCF keyframe not found: {name}")
    return key_id


def smooth_contacts(contacts: np.ndarray, window: int = 5) -> np.ndarray:
    strength = contacts.mean(axis=1) if contacts.ndim == 2 else contacts
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(strength.astype(np.float64), kernel, mode="same")


def stabilize_contact_targets(
    targets: np.ndarray, contacts: np.ndarray, foot_index: int
) -> None:
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
            targets[start:end, foot_index],
            axis=0,
            weights=np.maximum(weights, 1.0e-6),
        )
        blend = contacts[start:end, None]
        targets[start:end, foot_index] = (
            targets[start:end, foot_index] * (1.0 - blend) + anchor * blend
        )
        start = end


def semantic_world_alignment(
    source_first: np.ndarray, target_root_rotation: np.ndarray
) -> np.ndarray:
    """Match the semantic left/right and up axes used by the optimizer."""
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


def morph_to_robot_proportions(
    aligned_source: np.ndarray, robot_default: np.ndarray
) -> np.ndarray:
    targets = np.zeros((len(aligned_source), len(MAPPINGS), 3), dtype=np.float64)
    targets[:, 0] = (
        robot_default[0][None, :]
        + aligned_source[:, 0]
        - aligned_source[0, 0][None, :]
    )
    for parent, child in SEMANTIC_TREE_EDGES:
        source_initial = aligned_source[0, child] - aligned_source[0, parent]
        robot_segment = robot_default[child] - robot_default[parent]
        for frame in range(len(aligned_source)):
            source_current = aligned_source[frame, child] - aligned_source[frame, parent]
            if min(np.linalg.norm(source_initial), np.linalg.norm(source_current)) < 1.0e-8:
                rotated_segment = robot_segment
            else:
                delta_rotation, _ = Rotation.align_vectors(
                    source_current[None, :], source_initial[None, :]
                )
                rotated_segment = delta_rotation.apply(robot_segment)
            targets[frame, child] = targets[frame, parent] + rotated_segment
    return targets


def build_source_targets(
    source: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Reproduce the target positions used by the Kangaroo optimizer."""
    positions = np.asarray(source["positions"], dtype=np.float64)
    orientations = np.asarray(source["orientations"], dtype=np.float64)
    default_qpos = model.key_qpos[keyframe_id(model, "init_state")].copy()
    data.qpos[:] = default_qpos
    mujoco.mj_forward(model, data)

    default_xyzw = np.asarray(
        [default_qpos[4], default_qpos[5], default_qpos[6], default_qpos[3]]
    )
    default_rotation = Rotation.from_quat(default_xyzw).as_matrix()
    world_alignment = semantic_world_alignment(positions[0, :15], default_rotation)
    source_delta = positions - positions[0, 0][None, None, :]
    aligned_source = (
        default_qpos[:3][None, None, :]
        + np.einsum("ij,tkj->tki", world_alignment, source_delta) * scale
    )
    robot_default = data.xpos[body_ids].copy()
    targets = morph_to_robot_proportions(aligned_source[:, :15], robot_default)

    left = smooth_contacts(np.asarray(source["left_foot_contacts"]))
    right = smooth_contacts(np.asarray(source["right_foot_contacts"]))
    stabilize_contact_targets(targets, left, 7)
    stabilize_contact_targets(targets, right, 8)
    for foot_index, strength in ((7, left), (8, right)):
        targets[:, foot_index, 2] = (
            targets[:, foot_index, 2] * (1.0 - strength) + 0.012 * strength
        )
    return targets


def colors() -> np.ndarray:
    return np.asarray(
        [(*colorsys.hsv_to_rgb(i / len(MAPPINGS), 0.8, 1.0), 1.0) for i in range(len(MAPPINGS))],
        dtype=np.float32,
    )


def add_sphere(scene, position, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_line(scene, start, end, width, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def main() -> None:
    args = parse_args()
    source = np.load(args.keypoints, allow_pickle=True).item()
    result = np.load(args.retargeted)
    root_pos = np.asarray(result["base_frame_pos"], dtype=np.float64)
    root_quat = np.asarray(result["base_frame_wxyz"], dtype=np.float64)
    joints = np.asarray(result["joint_angles"], dtype=np.float64)
    scale = float(result["source_scale"])

    # Match the exact source slice used to create this NPZ. This matters for
    # contact-stance anchoring when a short --max-frames probe is visualized.
    source_frame_count = len(np.asarray(source["positions"]))
    frame_count = min(source_frame_count, len(root_pos), len(joints))
    source = {
        key: (
            value[:frame_count]
            if isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == source_frame_count
            else value
        )
        for key, value in source.items()
    }

    model = load_model(args.model)
    data = mujoco.MjData(model)
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for _, body_name in MAPPINGS
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        missing = [name for (_, name), idx in zip(MAPPINGS, body_ids) if idx < 0]
        raise ValueError(f"Mapped bodies not found: {missing}")
    if model.nq != 7 + joints.shape[1]:
        raise ValueError(
            f"Model expects {model.nq - 7} joints, NPZ contains {joints.shape[1]}"
        )

    source_targets = build_source_targets(source, model, data, body_ids, scale)
    palette = colors()

    print("\nSOMA23 keypoint -> Kangaroo target body")
    print("-" * 58)
    for index, ((source_name, target_name), color) in enumerate(zip(MAPPINGS, palette)):
        rgb = tuple(int(channel * 255) for channel in color[:3])
        if MATCH_ENABLED[index]:
            print(f"{index:2d}  {source_name:16s} -> {target_name:27s} RGB{rgb}")
        else:
            print(f"{index:2d}  {source_name:16s} -> DISABLED (structure only)")
    print("\nSpace: pause/play | Left/Right: one frame | R: restart | Q/Esc: close")
    print("Large spheres: source targets | Small spheres: optimized Kangaroo bodies")
    print("Cyan lines: Kangaroo semantic chain | White/cyan markers: loop anchors")
    if args.side_by_side:
        print("Side-by-side: source skeleton shifted 1.2 m beside the Kangaroo")

    state = {"frame": 0, "playing": not args.start_paused}
    lock = threading.Lock()
    stop_requested = threading.Event()

    def key_callback(keycode: int) -> None:
        with lock:
            if keycode == glfw.KEY_SPACE:
                state["playing"] = not state["playing"]
            elif keycode == glfw.KEY_RIGHT:
                state["playing"] = False
                state["frame"] = (state["frame"] + 1) % frame_count
            elif keycode == glfw.KEY_LEFT:
                state["playing"] = False
                state["frame"] = (state["frame"] - 1) % frame_count
            elif keycode == glfw.KEY_R:
                state["playing"] = False
                state["frame"] = 0
            elif keycode == glfw.KEY_Q:
                stop_requested.set()
            else:
                return
            print(
                f"Frame {state['frame']}/{frame_count - 1} "
                f"({state['frame'] / args.fps:.2f}s), "
                f"playing={state['playing']}"
            )

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def request_clean_shutdown(_signum, _frame) -> None:
        if not stop_requested.is_set():
            print("\nCtrl+C received: closing MuJoCo viewer cleanly...")
        stop_requested.set()

    signal.signal(signal.SIGINT, request_clean_shutdown)
    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = np.array(
                [0.0, 0.6 if args.side_by_side else 0.0, 0.85]
            )
            viewer.cam.distance = 4.0 if args.side_by_side else 3.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0
            next_frame_time = time.monotonic()

            while viewer.is_running() and not stop_requested.is_set():
                with lock:
                    frame = state["frame"]
                    playing = state["playing"]
                shared_offset = np.zeros(3)
                if args.in_place:
                    shared_offset[:2] = root_pos[frame, :2]

                data.qpos[:3] = root_pos[frame] - shared_offset
                data.qpos[3:7] = root_quat[frame]
                data.qpos[7:] = joints[frame]
                mujoco.mj_forward(model, data)
                points = source_targets[frame, :15] - shared_offset
                if args.side_by_side:
                    points = points.copy()
                    points[:, 1] += 1.2
                targets = data.xpos[body_ids]

                loop_anchors = []
                for equality_id in range(model.neq):
                    if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_CONNECT:
                        continue
                    body_a = int(model.eq_obj1id[equality_id])
                    body_b = int(model.eq_obj2id[equality_id])
                    rotation_a = data.xmat[body_a].reshape(3, 3)
                    rotation_b = data.xmat[body_b].reshape(3, 3)
                    anchor_a = (
                        data.xpos[body_a]
                        + rotation_a @ model.eq_data[equality_id, :3]
                    )
                    anchor_b = (
                        data.xpos[body_b]
                        + rotation_b @ model.eq_data[equality_id, 3:6]
                    )
                    loop_anchors.append((anchor_a, anchor_b))

                with viewer.lock():
                    scene = viewer.user_scn
                    scene.ngeom = 0
                    for start, end in SKELETON_EDGES:
                        add_line(
                            scene,
                            points[start],
                            points[end],
                            0.006,
                            [0.7, 0.7, 0.7, 0.75],
                        )
                        add_line(
                            scene,
                            targets[start],
                            targets[end],
                            0.005,
                            [0.1, 0.85, 1.0, 0.8],
                        )
                for index, color in enumerate(palette):
                    source_color = color if MATCH_ENABLED[index] else np.asarray(
                        [0.45, 0.45, 0.45, 0.8], dtype=np.float32
                    )
                    add_sphere(scene, points[index], 0.027, source_color)
                    if not MATCH_ENABLED[index]:
                        continue
                    target_color = color.copy()
                    target_color[3] = 0.85
                    add_sphere(scene, targets[index], 0.017, target_color)
                    line_color = color.copy()
                    line_color[3] = 0.72
                    add_line(
                        scene,
                        points[index],
                        targets[index],
                        0.004,
                        line_color,
                    )
                for anchor_a, anchor_b in loop_anchors:
                    add_sphere(scene, anchor_a, 0.014, [0.0, 1.0, 1.0, 1.0])
                    add_sphere(scene, anchor_b, 0.009, [1.0, 1.0, 1.0, 1.0])
                    add_line(
                        scene,
                        anchor_a,
                        anchor_b,
                        0.007,
                        [1.0, 1.0, 1.0, 1.0],
                    )
                viewer.sync()

                now = time.monotonic()
                if playing and now >= next_frame_time:
                    with lock:
                        state["frame"] = (state["frame"] + 1) % frame_count
                    next_frame_time = now + 1.0 / args.fps
                time.sleep(0.002)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        print("MuJoCo viewer closed.")


if __name__ == "__main__":
    main()
