"""Render static and optimized SOMA23-to-Kangaroo comparison videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from visualize_kangaroo_retarget_mapping_mujoco import (
    DEFAULT_MODEL,
    MATCH_ENABLED,
    MAPPINGS,
    SKELETON_EDGES,
    add_line,
    add_sphere,
    build_source_targets,
    colors,
    keyframe_id,
    load_model,
    semantic_world_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--retargeted", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("media"))
    parser.add_argument("--prefix", default="kangaroo_arm")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--static-seconds", type=float, default=5.0)
    parser.add_argument("--source-offset-y", type=float, default=1.2)
    parser.add_argument(
        "--in-place",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove horizontal root translation from the optimized playback.",
    )
    return parser.parse_args()


def body_ids(model: mujoco.MjModel) -> np.ndarray:
    result = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for _, body_name in MAPPINGS
        ],
        dtype=np.int32,
    )
    if np.any(result < 0):
        missing = [name for (_, name), index in zip(MAPPINGS, result) if index < 0]
        raise ValueError(f"Mapped bodies not found: {missing}")
    return result


def make_camera(source_offset_y: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([0.0, source_offset_y * 0.5, 0.85])
    camera.distance = 3.4
    camera.azimuth = 135.0
    camera.elevation = -15.0
    return camera


def brighten_scene(scene) -> None:
    """Use repo-friendly lighting for the dark Kangaroo materials."""
    if scene.nlight:
        scene.lights[0].ambient[:] = (0.35, 0.35, 0.35)
        scene.lights[0].diffuse[:] = (0.9, 0.9, 0.9)
        scene.lights[0].specular[:] = (0.25, 0.25, 0.25)
    scene.nlight = max(scene.nlight, 2)
    fill = scene.lights[1]
    fill.id = 1
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.headlight = 0
    fill.castshadow = 0
    fill.dir[:] = (-0.5, 0.4, -1.0)
    fill.ambient[:] = (0.15, 0.15, 0.15)
    fill.diffuse[:] = (0.75, 0.75, 0.75)
    fill.specular[:] = (0.15, 0.15, 0.15)


def writer(path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )


def draw_comparison(
    scene,
    source_points: np.ndarray,
    robot_points: np.ndarray,
    palette: np.ndarray,
) -> None:
    for start, end in SKELETON_EDGES:
        add_line(
            scene,
            source_points[start],
            source_points[end],
            0.006,
            [0.08, 0.08, 0.08, 0.9],
        )
        add_line(
            scene,
            robot_points[start],
            robot_points[end],
            0.005,
            [0.1, 0.85, 1.0, 0.85],
        )
    for index, color in enumerate(palette):
        source_color = color if MATCH_ENABLED[index] else np.asarray(
            [0.45, 0.45, 0.45, 0.8], dtype=np.float32
        )
        add_sphere(scene, source_points[index], 0.027, source_color)
        if not MATCH_ENABLED[index]:
            continue
        target_color = color.copy()
        target_color[3] = 0.85
        add_sphere(scene, robot_points[index], 0.017, target_color)
        line_color = color.copy()
        line_color[3] = 0.5
        add_line(
            scene,
            source_points[index],
            robot_points[index],
            0.003,
            line_color,
        )


def static_source_points(
    source: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_body_ids: np.ndarray,
    source_offset_y: float,
) -> np.ndarray:
    positions = np.asarray(source["positions"], dtype=np.float64)[0, :15]
    init_qpos = model.key_qpos[keyframe_id(model, "init_state")]
    data.qpos[:] = init_qpos
    mujoco.mj_forward(model, data)
    robot_points = data.xpos[target_body_ids]

    source_height = positions[0, 2] - np.mean(positions[[7, 8], 2])
    robot_height = robot_points[0, 2] - np.mean(robot_points[[7, 8], 2])
    scale = float(np.clip(robot_height / max(source_height, 1.0e-8), 0.75, 1.1))
    default_xyzw = np.asarray(
        [init_qpos[4], init_qpos[5], init_qpos[6], init_qpos[3]]
    )
    from scipy.spatial.transform import Rotation

    default_rotation = Rotation.from_quat(default_xyzw).as_matrix()
    alignment = semantic_world_alignment(positions, default_rotation)
    points = (
        robot_points[0][None, :]
        + np.einsum("ij,kj->ki", alignment, positions - positions[0]) * scale
    )
    points[:, 1] += source_offset_y
    return points


def render_static_video(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source: dict,
    target_body_ids: np.ndarray,
    palette: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> None:
    init_qpos = model.key_qpos[keyframe_id(model, "init_state")]
    data.qpos[:] = init_qpos
    mujoco.mj_forward(model, data)
    source_points = static_source_points(
        source, model, data, target_body_ids, args.source_offset_y
    )
    robot_points = data.xpos[target_body_ids].copy()
    camera = make_camera(args.source_offset_y)
    frame_count = max(1, int(round(args.static_seconds * args.fps)))

    with writer(output, args.fps) as video:
        for frame in range(frame_count):
            phase = frame / max(frame_count - 1, 1)
            camera.azimuth = 125.0 + 20.0 * phase
            renderer.update_scene(data, camera=camera)
            brighten_scene(renderer.scene)
            draw_comparison(renderer.scene, source_points, robot_points, palette)
            video.append_data(renderer.render())
    print(f"Wrote static mapping video: {output}")


def render_optimized_video(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source: dict,
    result,
    target_body_ids: np.ndarray,
    palette: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> None:
    root_pos = np.asarray(result["base_frame_pos"], dtype=np.float64)
    root_quat = np.asarray(result["base_frame_wxyz"], dtype=np.float64)
    joints = np.asarray(result["joint_angles"], dtype=np.float64)
    scale = float(result["source_scale"])
    frame_count = min(len(source["positions"]), len(root_pos), len(joints))
    sliced_source = {
        key: (
            value[:frame_count]
            if isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == len(source["positions"])
            else value
        )
        for key, value in source.items()
    }
    source_targets = build_source_targets(
        sliced_source, model, data, target_body_ids, scale
    )
    camera = make_camera(args.source_offset_y)

    with writer(output, args.fps) as video:
        for frame in range(frame_count):
            shared_offset = np.zeros(3)
            if args.in_place:
                shared_offset[:2] = root_pos[frame, :2]
            data.qpos[:3] = root_pos[frame] - shared_offset
            data.qpos[3:7] = root_quat[frame]
            data.qpos[7:] = joints[frame]
            mujoco.mj_forward(model, data)
            source_points = source_targets[frame, :15] - shared_offset
            source_points = source_points.copy()
            source_points[:, 1] += args.source_offset_y
            robot_points = data.xpos[target_body_ids].copy()

            # Keep travelling motions framed instead of leaving the camera at
            # the world origin.  The source skeleton receives the same world
            # translation, so the side-by-side comparison remains aligned.
            if not args.in_place:
                camera.lookat[:] = np.asarray(
                    [root_pos[frame, 0], root_pos[frame, 1] + args.source_offset_y * 0.5, 0.85]
                )

            renderer.update_scene(data, camera=camera)
            brighten_scene(renderer.scene)
            draw_comparison(renderer.scene, source_points, robot_points, palette)
            video.append_data(renderer.render())
    print(f"Wrote optimized motion video: {output}")


def main() -> None:
    args = parse_args()
    source = np.load(args.keypoints, allow_pickle=True).item()
    result = np.load(args.retargeted)
    model = load_model(args.model)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    if model.nmat:
        model.mat_rgba[:, :3] = np.maximum(model.mat_rgba[:, :3], 0.5)
    visual_mask = np.isin(model.geom_group, [1, 2])
    model.geom_rgba[visual_mask, :3] = np.maximum(
        model.geom_rgba[visual_mask, :3], 0.5
    )
    data = mujoco.MjData(model)
    target_body_ids = body_ids(model)
    palette = colors()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_output = args.output_dir / f"{args.prefix}_mapping_before.mp4"
    optimized_output = args.output_dir / f"{args.prefix}_retargeted.mp4"
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        render_static_video(
            renderer,
            model,
            data,
            source,
            target_body_ids,
            palette,
            args,
            static_output,
        )
        render_optimized_video(
            renderer,
            model,
            data,
            source,
            result,
            target_body_ids,
            palette,
            args,
            optimized_output,
        )


if __name__ == "__main__":
    main()
