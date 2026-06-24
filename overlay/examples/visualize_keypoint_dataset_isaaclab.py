"""Visualize source motion keypoints in Isaac Lab without loading a robot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (3, 5), (5, 7),
    (2, 4), (4, 6), (6, 8),
    (0, 9), (0, 10), (9, 10),
    (9, 11), (11, 13),
    (10, 12), (12, 14),
    (13, 15), (14, 16), (0, 17),
]

SOMA23_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (3, 7), (7, 8), (8, 9), (9, 10),
    (3, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18),
    (0, 19), (19, 20), (20, 21), (21, 22),
]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("keypoints_file", type=Path)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--playback-speed", type=float, default=1.0)
parser.add_argument(
    "--frame",
    type=int,
    default=None,
    help="Display one fixed frame instead of playing the animation.",
)
parser.add_argument(
    "--in-place",
    action="store_true",
    help="Remove horizontal pelvis translation during playback.",
)
parser.add_argument("--point-size", type=float, default=12.0)
parser.add_argument(
    "--semantic-only",
    action="store_true",
    help="Show exactly the 15 retargeting landmarks, without auxiliary points.",
)
parser.add_argument("--motion-index", type=int, default=0)
parser.add_argument(
    "--motion-name",
    type=str,
    default=None,
    help="Unique filename substring when opening a packaged .pt MotionLib.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
import omni.kit.app  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.util.debug_draw", True
)
from isaacsim.util.debug_draw import _debug_draw  # noqa: E402


def contact_active(values: np.ndarray, frame: int) -> bool:
    return bool(np.asarray(values[frame]).mean() > 0.5)


def draw_frame(
    draw,
    positions,
    left_contacts,
    right_contacts,
    edges,
    left_contact_points,
    right_contact_points,
    frame: int,
) -> None:
    points = positions[frame]
    point_count = len(points)

    point_colors = [(0.12, 0.45, 0.95, 1.0)] * point_count
    if contact_active(left_contacts, frame):
        for index in left_contact_points:
            point_colors[index] = (0.1, 0.9, 0.2, 1.0)
    if contact_active(right_contacts, frame):
        for index in right_contact_points:
            point_colors[index] = (0.1, 0.9, 0.2, 1.0)

    starts = [tuple(points[start]) for start, _ in edges]
    ends = [tuple(points[end]) for _, end in edges]

    draw.clear_points()
    draw.clear_lines()
    draw.draw_points(
        [tuple(point) for point in points],
        point_colors,
        [args.point_size] * point_count,
    )
    draw.draw_lines(
        starts,
        ends,
        [(0.05, 0.05, 0.05, 1.0)] * len(edges),
        [3.0] * len(edges),
    )


def load_source_motion():
    """Load processed .npy keypoints or one raw SOMA23 motion from a .pt library."""
    if args.keypoints_file.suffix.lower() != ".pt":
        source = np.load(args.keypoints_file, allow_pickle=True).item()
        positions = np.asarray(source["positions"], dtype=np.float32)
        if args.semantic_only:
            positions = positions[:, :15]
        edges = [
            (a, b)
            for a, b in SKELETON_EDGES
            if a < positions.shape[1] and b < positions.shape[1]
        ]
        return (
            positions,
            np.asarray(source["left_foot_contacts"]),
            np.asarray(source["right_foot_contacts"]),
            edges,
            (5, 7),
            (6, 8),
            f"processed keypoints {args.keypoints_file.name}",
        )

    import torch

    library = torch.load(args.keypoints_file, weights_only=False, map_location="cpu")
    required = {"gts", "contacts", "length_starts", "motion_num_frames", "motion_files"}
    missing = required - set(library)
    if missing:
        raise ValueError(f"Unsupported packaged MotionLib; missing keys: {sorted(missing)}")

    motion_files = list(library["motion_files"])
    motion_index = args.motion_index
    if args.motion_name is not None:
        matches = [i for i, name in enumerate(motion_files) if args.motion_name in str(name)]
        if len(matches) != 1:
            names = [Path(str(motion_files[i])).name for i in matches[:10]]
            raise ValueError(
                f"--motion-name must match exactly one motion, got {len(matches)}: {names}"
            )
        motion_index = matches[0]
    if not 0 <= motion_index < len(motion_files):
        raise IndexError(f"motion index {motion_index} outside 0..{len(motion_files) - 1}")

    start = int(library["length_starts"][motion_index])
    frame_count = int(library["motion_num_frames"][motion_index])
    stop = start + frame_count
    positions = library["gts"][start:stop].cpu().numpy().astype(np.float32)
    contacts = library["contacts"][start:stop].cpu().numpy()
    if positions.shape[1] != 23:
        raise ValueError(
            f"Raw mode currently expects SOMA23, got {positions.shape[1]} bodies"
        )

    if args.semantic_only:
        # Match KEYPOINT_MAPPING_SOMA23 and the clean .npy ordering exactly.
        semantic_indices = [0, 19, 15, 20, 16, 21, 17, 22, 18, 12, 8, 13, 9, 14, 10]
        positions = positions[:, semantic_indices]
        edges = [(a, b) for a, b in SKELETON_EDGES if a < 15 and b < 15]
        left_points = (5, 7)
        right_points = (6, 8)
        description = (
            f"raw SOMA23 semantic motion {motion_index}: "
            f"{Path(str(motion_files[motion_index])).name}"
        )
    else:
        edges = SOMA23_EDGES
        left_points = (21, 22)
        right_points = (17, 18)
        description = (
            f"raw SOMA23 motion {motion_index}: "
            f"{Path(str(motion_files[motion_index])).name}"
        )

    return (
        positions,
        contacts[:, [21, 22]],
        contacts[:, [17, 18]],
        edges,
        left_points,
        right_points,
        description,
    )


def main() -> None:
    (
        positions,
        left_contacts,
        right_contacts,
        edges,
        left_contact_points,
        right_contact_points,
        description,
    ) = load_source_motion()

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions [frames, points, 3], got {positions.shape}")
    if args.in_place:
        positions = positions.copy()
        positions[:, :, :2] -= positions[:, :1, :2]

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args.fps))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    sim.reset()

    draw = _debug_draw.acquire_debug_draw_interface()
    sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 0.9))

    duration = len(positions) / args.fps
    print(
        f"Loaded {description}: {len(positions)} frames, "
        f"{positions.shape[1]} points, {duration:.2f}s at {args.fps:g} FPS"
    )
    print("Blue: keypoints | Green: foot contact | No robot and no physics")

    start_time = time.perf_counter()
    previous_frame = -1
    while simulation_app.is_running():
        if args.frame is None:
            elapsed = (time.perf_counter() - start_time) * args.playback_speed
            frame = int(elapsed * args.fps) % len(positions)
        else:
            frame = args.frame % len(positions)
        if frame != previous_frame:
            draw_frame(
                draw,
                positions,
                left_contacts,
                right_contacts,
                edges,
                left_contact_points,
                right_contact_points,
                frame,
            )
            if not args.in_place:
                root = positions[frame, 0]
                sim.set_camera_view(
                    eye=tuple(root + np.asarray([3.0, 3.0, 1.1])),
                    target=tuple(root + np.asarray([0.0, 0.0, 0.6])),
                )
            previous_frame = frame
        sim.render()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
