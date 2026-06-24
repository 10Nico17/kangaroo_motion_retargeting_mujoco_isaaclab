"""Inspect one SOMA23-to-Kangaroo mapping frame without optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from visualize_kangaroo_retarget_mapping_mujoco import (
    DEFAULT_MODEL,
    MAPPINGS,
    MATCH_ENABLED,
    SKELETON_EDGES,
    add_line,
    add_sphere,
    colors,
    keyframe_id,
    load_model,
    semantic_world_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Move the source skeleton 1.2 m to the viewer's left.",
    )
    parser.add_argument(
        "--opaque-robot",
        action="store_true",
        help="Keep robot meshes opaque instead of exposing internal markers.",
    )
    parser.add_argument(
        "--legend-mode",
        choices=("window", "overlay", "none"),
        default="window",
        help="Show legends in a separate Matplotlib window, as MuJoCo overlays, or hide them.",
    )
    return parser.parse_args()


def estimate_scale(source: np.ndarray, robot: np.ndarray) -> float:
    source_drop = source[0, 2] - np.mean(source[[7, 8], 2])
    robot_drop = robot[0, 2] - np.mean(robot[[7, 8], 2])
    if source_drop <= 1.0e-8 or robot_drop <= 1.0e-8:
        return 1.0
    return float(np.clip(robot_drop / source_drop, 0.75, 1.1))


def create_legend_window(
    palette: np.ndarray,
    joint_entries: list[tuple[int, str, str]],
):
    """Create a non-blocking color legend that leaves the 3D view unobscured."""
    import matplotlib.pyplot as plt

    figure, (key_ax, joint_ax) = plt.subplots(1, 2, figsize=(13, 8))
    figure.canvas.manager.set_window_title("Kangaroo retargeting legend")
    figure.suptitle("SOMA23 → Kangaroo mapping (no optimization)", fontsize=14)

    for axis in (key_ax, joint_ax):
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")

    key_ax.set_title("SOMA23 keypoints → target bodies", loc="left")
    key_y = np.linspace(0.94, 0.06, len(MAPPINGS))
    for index, ((source_name, target_name), color, y) in enumerate(
        zip(MAPPINGS, palette, key_y)
    ):
        display_color = color if MATCH_ENABLED[index] else [0.45, 0.45, 0.45, 1.0]
        key_ax.scatter([0.05], [y], s=110, color=display_color, edgecolor="black")
        mapping_text = (
            f"{source_name}  →  {target_name}"
            if MATCH_ENABLED[index]
            else f"{source_name}  →  DISABLED (structure only)"
        )
        key_ax.text(
            0.11,
            y,
            f"{index:02d}  {mapping_text}",
            va="center",
            fontsize=9,
        )

    joint_ax.set_title("Kangaroo joints", loc="left")
    joint_colors = {
        "floating root": (0.95, 0.1, 0.1, 1.0),
        "actuated": (0.1, 0.35, 1.0, 1.0),
        "passive": (1.0, 0.45, 0.05, 1.0),
    }
    joint_codes = {"floating root": "R", "actuated": "A", "passive": "P"}
    joint_y = np.linspace(0.96, 0.04, len(joint_entries))
    for (joint_id, joint_name, marker_type), y in zip(joint_entries, joint_y):
        color = joint_colors[marker_type]
        joint_ax.scatter([0.05], [y], s=55, color=color, edgecolor="black")
        joint_ax.text(
            0.10,
            y,
            f"{joint_codes[marker_type]} {joint_id:02d}  {joint_name}",
            va="center",
            fontsize=7.5,
        )

    figure.text(
        0.5,
        0.015,
        "A = actuated   P = passive linkage   R = floating root",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    plt.show(block=False)
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    return figure, plt


def main() -> None:
    args = parse_args()
    source = np.load(args.keypoints, allow_pickle=True).item()
    positions = np.asarray(source["positions"], dtype=np.float64)
    if args.frame < 0 or args.frame >= len(positions):
        raise ValueError(
            f"--frame must be between 0 and {len(positions) - 1}, got {args.frame}"
        )

    model = load_model(args.model)
    if not args.opaque_robot:
        visual_mask = np.isin(model.geom_group, [1, 2])
        model.geom_rgba[visual_mask, 3] = 0.28
    data = mujoco.MjData(model)
    init_key = keyframe_id(model, "init_state")
    data.qpos[:] = model.key_qpos[init_key]
    mujoco.mj_forward(model, data)

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

    robot_targets = data.xpos[body_ids].copy()
    default_root_rotation = data.xmat[body_ids[0]].reshape(3, 3).copy()
    world_alignment = semantic_world_alignment(
        positions[0, :15], default_root_rotation
    )
    scale = estimate_scale(positions[0, :15], robot_targets)
    source_points = (
        robot_targets[0][None, :]
        + np.einsum(
            "ij,kj->ki",
            world_alignment,
            positions[args.frame, :15] - positions[args.frame, 0],
        )
        * scale
    )
    mapping_distances = np.linalg.norm(source_points - robot_targets, axis=1)
    if args.side_by_side:
        source_points = source_points.copy()
        source_points[:, 1] += 1.2

    palette = colors()
    actuated_joint_ids = set(int(joint) for joint in model.actuator_trnid[:, 0])
    keypoint_legend_lines = ["SOMA23 KEYPOINTS"]
    joint_legend_lines = ["KANGAROO JOINTS", "A=actuated  P=passive  R=root"]
    joint_entries: list[tuple[int, str, str]] = []

    print("\nSTATIC VIEW: no optimization, Kangaroo init_state")
    print(f"Source frame: {args.frame}/{len(positions) - 1}, scale={scale:.4f}")
    print("\nSOMA23 keypoint -> Kangaroo target body")
    print("-" * 58)
    for index, ((source_name, target_name), color) in enumerate(
        zip(MAPPINGS, palette)
    ):
        rgb = tuple(int(channel * 255) for channel in color[:3])
        error = mapping_distances[index]
        if MATCH_ENABLED[index]:
            print(
                f"{index:2d}  {source_name:16s} -> {target_name:27s} "
                f"distance={error:6.3f} m RGB{rgb}"
            )
            keypoint_legend_lines.append(
                f"{index:02d} {source_name} -> {target_name}"
            )
        else:
            print(f"{index:2d}  {source_name:16s} -> DISABLED (structure only)")
            keypoint_legend_lines.append(
                f"{index:02d} {source_name} -> DISABLED"
            )

    print("\nRobot joint markers")
    print("-" * 58)
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            marker_type = "floating root"
        elif joint_id in actuated_joint_ids:
            marker_type = "actuated"
        else:
            marker_type = "passive"
        print(f"{joint_id:2d}  {joint_name:30s} {marker_type}")
        joint_entries.append((joint_id, joint_name, marker_type))
        marker_code = {
            "floating root": "R",
            "actuated": "A",
            "passive": "P",
        }[marker_type]
        joint_legend_lines.append(f"{marker_code} {joint_id:02d} {joint_name}")

    print("\nVisualization legend")
    print("Large colored spheres : source keypoints")
    print("Small colored spheres : mapped Kangaroo body origins")
    print("Colored lines         : direct keypoint mapping")
    print("Gray lines            : source skeleton")
    print("Cyan lines            : Kangaroo semantic skeleton")
    print("Blue joints/axes      : actuated joints")
    print("Orange joints/axes    : passive linkage joints")
    print("Red joint             : floating base")
    print("White/cyan anchors    : equality loop closures")
    print("Q or Esc              : close cleanly")
    if not args.opaque_robot:
        print("Robot meshes          : transparent to expose internal targets")

    legend_figure = None
    legend_plt = None
    if args.legend_mode == "window":
        legend_figure, legend_plt = create_legend_window(palette, joint_entries)

    stop_requested = threading.Event()

    def key_callback(keycode: int) -> None:
        if keycode == glfw.KEY_Q:
            stop_requested.set()

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
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])
            viewer.cam.distance = 3.2 if not args.side_by_side else 4.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0
            if args.legend_mode == "overlay":
                viewer.set_texts(
                    [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_LEFT,
                            "\n".join(keypoint_legend_lines),
                            "",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_RIGHT,
                            "\n".join(joint_legend_lines),
                            "",
                        ),
                    ]
                )

            while viewer.is_running() and not stop_requested.is_set():
                mujoco.mj_forward(model, data)
                robot_targets = data.xpos[body_ids].copy()

                loop_anchors = []
                for equality_id in range(model.neq):
                    if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_CONNECT:
                        continue
                    body_a = int(model.eq_obj1id[equality_id])
                    body_b = int(model.eq_obj2id[equality_id])
                    anchor_a = data.xpos[body_a] + data.xmat[body_a].reshape(
                        3, 3
                    ) @ model.eq_data[equality_id, :3]
                    anchor_b = data.xpos[body_b] + data.xmat[body_b].reshape(
                        3, 3
                    ) @ model.eq_data[equality_id, 3:6]
                    loop_anchors.append((anchor_a, anchor_b))

                with viewer.lock():
                    scene = viewer.user_scn
                    scene.ngeom = 0
                    for start, end in SKELETON_EDGES:
                        add_line(
                            scene,
                            source_points[start],
                            source_points[end],
                            0.006,
                            [0.7, 0.7, 0.7, 0.8],
                        )
                        add_line(
                            scene,
                            robot_targets[start],
                            robot_targets[end],
                            0.005,
                            [0.1, 0.85, 1.0, 0.85],
                        )
                    for index, color in enumerate(palette):
                        source_color = color if MATCH_ENABLED[index] else np.asarray(
                            [0.45, 0.45, 0.45, 0.8], dtype=np.float32
                        )
                        add_sphere(scene, source_points[index], 0.028, source_color)
                        if not MATCH_ENABLED[index]:
                            continue
                        add_sphere(scene, robot_targets[index], 0.023, color)
                        add_line(
                            scene,
                            source_points[index],
                            robot_targets[index],
                            0.004,
                            [*color[:3], 0.7],
                        )

                    for joint_id in range(model.njnt):
                        joint_type = model.jnt_type[joint_id]
                        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                            joint_color = [1.0, 0.1, 0.1, 1.0]
                        elif joint_id in actuated_joint_ids:
                            joint_color = [0.1, 0.35, 1.0, 1.0]
                        else:
                            joint_color = [1.0, 0.45, 0.05, 1.0]
                        anchor = data.xanchor[joint_id]
                        add_sphere(scene, anchor, 0.011, joint_color)
                        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                            add_line(
                                scene,
                                anchor - data.xaxis[joint_id] * 0.045,
                                anchor + data.xaxis[joint_id] * 0.045,
                                0.005,
                                joint_color,
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
                if legend_figure is not None:
                    legend_figure.canvas.flush_events()
                time.sleep(0.01)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if legend_figure is not None and legend_plt is not None:
            legend_plt.close(legend_figure)
        print("MuJoCo static mapping viewer closed.")


if __name__ == "__main__":
    main()
