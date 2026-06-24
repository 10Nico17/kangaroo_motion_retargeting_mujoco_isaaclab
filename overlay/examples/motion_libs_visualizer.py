# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Motion Visualizer with Smoothness Metrics
# Supports normalized jerk, oscillation index, and purposeful jerk metrics
# Uses threshold-based highlighting similar to the original visualizer

from typing import Dict, List
import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

FPS = 30

# Parse arguments first (argparse is safe, doesn't import torch)
parser = argparse.ArgumentParser(
    description="Motion Visualizer with Smoothness Metrics"
)
parser.add_argument(
    "--motion_files",
    type=str,
    nargs="+",
    required=True,
    help="Paths to MotionLib .pt files (e.g., predicted_motion_lib.pt motion_lib.pt). Each file will be displayed in a separate environment.",
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=["isaacgym", "isaaclab", "newton"],
    default="isaacgym",
    help="Simulator to use (isaacgym, isaaclab, newton)",
)
parser.add_argument(
    "--robot",
    type=str,
    choices=["g1", "rigv1", "h1_2", "smpl", "soma23", "kangaroo"],
    default="g1",
    help="Robot to load (g1, rigv1, h1_2, smpl, soma23, or kangaroo)",
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier (1.0 = normal speed)",
)
parser.add_argument(
    "--smoothness_threshold",
    type=float,
    default=6500.0,
    help="Smoothness threshold to highlight bodies (higher values = less smooth). FPS-invariant metric.",
)
parser.add_argument(
    "--metric",
    type=str,
    choices=["nj", "oi", "pj"],
    default="nj",
    help="Smoothness metric: 'nj' for normalized jerk, 'oi' for oscillation index, 'pj' for purposeful jerk",
)
parser.add_argument(
    "--use-data-vel",
    action="store_true",
    help="Use stored rigid_body_vel from motion data instead of computing velocities via finite differences (default: False, use finite differences)",
)
parser.add_argument(
    "--window_sec",
    type=float,
    default=0.4,
    help="Sliding window length in seconds for computing smoothness metrics",
)
parser.add_argument(
    "--origin_xy",
    type=float,
    nargs=2,
    default=[0.0, 0.0],
    help="Target x,y position to move all motions to (default: 0.0 0.0)",
)
parser.add_argument(
    "--hide-markers",
    action="store_true",
    help="Hide smoothness, contact, and selected-body marker spheres.",
)
parser.add_argument(
    "--in-place",
    action="store_true",
    help="Keep compatibility with retargeting commands; motion starts at origin.",
)
parser.add_argument(
    "--camera-view",
    choices=["default", "front", "side"],
    default="default",
    help="Set a reproducible initial camera view in IsaacLab.",
)
parser.add_argument(
    "--follow-camera",
    action="store_true",
    help="Move the selected fixed camera and source skeleton with the robot root.",
)
parser.add_argument(
    "--source-keypoints",
    type=Path,
    default=None,
    help="Draw a processed SOMA23 keypoint NPY beside the retargeted robot.",
)
parser.add_argument(
    "--source-offset-y",
    type=float,
    default=1.5,
    help="Sideways offset in meters for the source skeleton.",
)
parser.add_argument(
    "--source-yaw-deg",
    type=float,
    default=-90.0,
    help="Rotate the source skeleton around world Z to match robot forward.",
)
parser.add_argument(
    "--record-video",
    type=Path,
    default=None,
    help="Automatically record one clip and save it to this MP4 path.",
)
parser.add_argument(
    "--record-frames",
    type=int,
    default=0,
    help="Frames for automatic recording; zero uses one complete motion loop.",
)
parser.add_argument("--record-width", type=int, default=1280)
parser.add_argument("--record-height", type=int, default=720)
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import torch  # noqa: E402
import numpy as np  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

from protomotions.simulator.base_simulator.config import (  # noqa: E402
    VisualizationMarkerConfig,
    MarkerConfig,
    MarkerState,
)
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.robot_configs.base import ControlType  # noqa: E402
from protomotions.components.motion_lib import MotionLib  # noqa: E402
from protomotions.components.scene_lib import (  # noqa: E402
    SceneLib,
    MeshSceneObject,
    Scene,
    ObjectOptions,
    SceneLibConfig,
    ReplicationMethod,
    SubsetMethod,
)
import os  # noqa: E402


SOURCE_SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (3, 5), (5, 7),
    (2, 4), (4, 6), (6, 8),
    (0, 9), (0, 10), (9, 10),
    (9, 11), (11, 13),
    (10, 12), (12, 14),
]


@dataclass
class RobotSpec:
    """Robot specification with joint/body names for visualization"""

    # Body names to visualize (these are the rigid body names, not joint names)
    viz_bodies: List[str]


# Define robot specifications
ROBOT_SPECS = {
    "g1": RobotSpec(
        viz_bodies=[],
    ),
    "rigv1": RobotSpec(
        viz_bodies=[],
    ),
    "h1_2": RobotSpec(
        viz_bodies=[],
    ),
    "smpl": RobotSpec(
        viz_bodies=[],
    ),
    "soma23": RobotSpec(
        viz_bodies=[],
    ),
    "kangaroo": RobotSpec(
        viz_bodies=[],
    ),
}


# ----- Smoothness Metrics Implementation -----
def _diff(x, dt):
    """Compute finite difference with given time step"""
    return (x[1:] - x[:-1]) / dt


def normalized_jerk_from_vel(vel, dt, eps=0.1):
    """
    Compute normalized jerk from velocity trajectory.
    Args:
        vel: [T, N, 3] velocity trajectory
        dt: time step
    Returns:
        per_body_nj: [N] normalized jerk per body
        mean_nj: scalar mean normalized jerk

        --smoothness_threshold 6500.0 --window_sec 0.4 (using finite differences, which is default) seems to be good qualitative measures
        Uses T^5 for dimensionless, FPS-invariant normalization.
    """
    a = _diff(vel, dt)  # [T-1, N, 3]
    j = _diff(a, dt)  # [T-2, N, 3]

    speed = torch.linalg.norm(vel, dim=-1)  # [T, N]
    jnorm2 = torch.linalg.norm(j, dim=-1) ** 2  # [T-2, N]

    T_tot = vel.shape[0] * dt
    L = (speed * dt).sum(dim=0).clamp_min(eps)  # [N] - path length per body
    int_j2 = (jnorm2 * dt).sum(dim=0)  # [N] - integrated squared jerk
    # Using T^5 (not T^3) for dimensionless, FPS-invariant normalization
    nj = (T_tot**5 * int_j2) / (L**2 + eps)  # [N] - normalized jerk
    return nj, nj.mean()


def oscillation_index_from_vel(vel, dt, eps=0.001):
    """
    Compute oscillation index from velocity trajectory.
    Args:
        vel: [T, N, 3] velocity trajectory
        dt: time step
    Returns:
        per_body_oi: [N] oscillation index per body (0-2, higher = more oscillatory)
        mean_oi: scalar mean oscillation index

        threshold 1.2 (slightly larger than 1) seems meaningful
    """
    a = _diff(vel, dt)  # [T-1, N, 3]
    a1, a2 = a[:-1], a[1:]  # [T-2, N, 3]

    fps = 1.0 / dt
    a1 = a1 / fps
    a2 = a2 / fps

    num = (a1 * a2).sum(-1)  # [T-2, N]
    den = (torch.linalg.norm(a1, dim=-1) * torch.linalg.norm(a2, dim=-1)).clamp_min(eps)
    # print(torch.mean(den))
    cos = (num / den).clamp(-1, 1)  # [T-2, N]
    oi = (1 - cos).mean(dim=0)  # [N]
    return oi, oi.mean()


def purposeful_jerk_from_vel(vel, dt, eps=1e-8):
    """
    Compute purposeful jerk from velocity trajectory.
    High values indicate jerk that coincides with velocity direction changes.
    Args:
        vel: [T, N, 3] velocity trajectory
        dt: time step
    Returns:
        per_body_pj: [N] purposeful jerk per body
        mean_pj: scalar mean purposeful jerk
    """
    a = _diff(vel, dt)  # [T-1, N, 3]
    j = _diff(a, dt)  # [T-2, N, 3]
    v1, v2 = vel[:-1], vel[1:]  # [T-1, N, 3]

    num = (v1 * v2).sum(-1)  # [T-1, N]
    den = (torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)).clamp_min(eps)
    misalign = 1 - (num / den).clamp(-1, 1)  # [T-1, N], in [0,2]
    jn = torch.linalg.norm(j, dim=-1)  # [T-2, N]

    # Align shapes: use minimum length
    Tm = min(misalign.shape[0] - 1, jn.shape[0])
    pj = (jn[:Tm] * misalign[1 : 1 + Tm]).mean(dim=0)  # [N]
    return pj, pj.mean()


def create_checkerboard_ground(
    num_envs: int, device: torch.device, simulator_type: str = "isaacgym"
) -> SceneLib:
    """
    Create a visual checkerboard ground plane using a textured mesh.

    Args:
        num_envs: Number of environments
        device: Torch device
        simulator_type: Type of simulator (isaacgym, isaaclab, etc.)

    Returns:
        SceneLib with checkerboard ground for each environment
    """
    # Get path to the checkerboard asset (URDF for IsaacGym, USD for IsaacLab)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    checkerboard_dir = os.path.join(
        project_root, "protomotions/data/assets/checkerboard"
    )

    if simulator_type == "isaaclab":
        asset_path = os.path.join(checkerboard_dir, "checkerboard_ground.usda")
        asset_type = "USD"
    else:
        # IsaacGym, Newton, Genesis use URDF
        asset_path = os.path.join(checkerboard_dir, "checkerboard_ground.urdf")
        asset_type = "URDF"

    if not os.path.exists(asset_path):
        print(f"Warning: Checkerboard ground {asset_type} not found at {asset_path}")
        print(f"Assets should be in: {checkerboard_dir}")
        return None

    # Get texture path for IsaacGym (IsaacLab loads it from USD)
    texture_path = None
    if simulator_type != "isaaclab":
        texture_file = os.path.join(checkerboard_dir, "checkerboard_texture.png")
        if os.path.exists(texture_file):
            texture_path = texture_file

    # Create scenes for each environment
    # IMPORTANT: Each scene needs its own MeshSceneObject instance,
    # otherwise attributes get overwritten during _process_scene_objects()
    scenes = []
    for _ in range(num_envs):
        ground_mesh = MeshSceneObject(
            object_path=asset_path,
            translation=(0.0, 0.0, -0.005),  # Slightly below zero
            rotation=(0.0, 0.0, 0.0, 1.0),  # No rotation (x, y, z, w)
            options=ObjectOptions(
                fix_base_link=True,  # Static object
                vhacd_enabled=False,  # Disable convex decomposition for simple plane
                texture_path=texture_path,  # Texture for IsaacGym (None for IsaacLab)
            ),
        )
        scenes.append(Scene(objects=[ground_mesh], offset=(0.0, 0.0)))

    # Configure scene lib
    scene_lib_config = SceneLibConfig(
        scene_file=None,  # No file, using inline scene
        replicate_method=ReplicationMethod.SEQUENTIAL,
        subset_method=SubsetMethod.FIRST,
        pointcloud_samples_per_object=None,
    )

    # Return a SceneLib without terrain (avoids collision geometry in simulators)
    return SceneLib(
        config=scene_lib_config,
        num_envs=num_envs,
        scenes=scenes,
        device=device,
        terrain=None,  # No terrain to avoid unwanted collisions
    )


class MotionVisualizerSmoothness:
    def __init__(
        self,
        motion_files: List[str],
        robot_name: str = "g1",
        simulator_type: str = "isaacgym",
        headless: bool = False,
        cpu_only: bool = False,
        extra_simulator_params: dict = None,
        playback_speed: float = 1.0,
        metric: str = "nj",
        use_data_vel: bool = False,
        window_sec: float = 2.0,
    ):
        self.motion_files = [Path(f) for f in motion_files]
        self.robot_name = robot_name
        self.robot_spec = ROBOT_SPECS[robot_name]
        self.num_envs = len(motion_files)
        self.simulator_type = simulator_type
        self.headless = headless
        self.playback_speed = playback_speed
        self.device = torch.device("cuda:0" if not cpu_only else "cpu")
        self.smoothness_threshold = args.smoothness_threshold
        self.metric = metric
        self.use_data_vel = use_data_vel  # If False (default), use finite differences
        self.window_frames = max(4, int(round(window_sec * FPS)))

        # Load motion libraries (.pt files)
        from protomotions.components.motion_lib import MotionLibConfig

        self.motion_libs = [
            MotionLib(
                config=MotionLibConfig(motion_file=str(motion_file)), device=self.device
            )
            for motion_file in self.motion_files
        ]

        # Move all motions to the specified origin
        for i, motion_lib in enumerate(self.motion_libs):
            target_xy = torch.tensor(args.origin_xy, device=self.device)
            target_xy = target_xy + torch.tensor([1.0 * i, 0.0], device=self.device)
            print(f"Translating motion library {i} to origin {target_xy}")
            motion_lib.translate_all_motions_to_origin(target_xy)

        self.source_positions = None
        self.source_draw = None
        if args.source_keypoints is not None:
            if simulator_type != "isaaclab":
                raise ValueError("--source-keypoints currently requires --simulator isaaclab")
            source = np.load(args.source_keypoints, allow_pickle=True).item()
            positions = np.asarray(source["positions"], dtype=np.float32)
            if positions.ndim != 3 or positions.shape[1] < 15 or positions.shape[2] != 3:
                raise ValueError(
                    "Source keypoints must have shape [frames, >=15, 3], "
                    f"got {positions.shape}"
                )
            positions = positions[:, :15].copy()
            # Display the source motion in place beside the robot. Preserve
            # vertical motion while removing horizontal root translation.
            positions[:, :, :2] -= positions[:, :1, :2]
            yaw = np.deg2rad(args.source_yaw_deg)
            cosine, sine = np.cos(yaw), np.sin(yaw)
            source_xy = positions[:, :, :2].copy()
            positions[:, :, 0] = (
                cosine * source_xy[:, :, 0] - sine * source_xy[:, :, 1]
            )
            positions[:, :, 1] = (
                sine * source_xy[:, :, 0] + cosine * source_xy[:, :, 1]
            )
            positions[:, :, 1] += args.source_offset_y
            self.source_positions = positions

            import omni.kit.app

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            extension_manager.set_extension_enabled_immediate(
                "isaacsim.util.debug_draw", True
            )
            from isaacsim.util.debug_draw import _debug_draw

            self.source_draw = _debug_draw.acquire_debug_draw_interface()
            print(
                f"Loaded source skeleton: {args.source_keypoints} "
                f"({len(positions)} frames, offset_y={args.source_offset_y:.2f} m, "
                f"yaw={args.source_yaw_deg:.1f} deg)"
            )

        # Motion playback state
        self.current_motion_idx = 0
        self.current_frame = 0
        # Use the first motion lib to determine total motions and current motion length
        self.total_motions = self.motion_libs[0].num_motions()
        self.current_motion_length = (
            self.motion_libs[0]
            .get_motion_num_frames(None)[self.current_motion_idx]
            .item()
        )

        print(
            f"Loaded {len(self.motion_files)} motion files with {self.total_motions} motions each"
        )
        print(f"Motion files: {[str(f) for f in self.motion_files]}")
        print(
            f"Current motion {self.current_motion_idx} has {self.current_motion_length} frames"
        )

        # Load robot configuration using factory function
        self.robot_cfg = robot_config(robot_name)

        # Store kinematic info for later use
        self.kinematic_info = self.robot_cfg.kinematic_info

        # Create simulator configuration using factory function
        self.simulator_cfg = simulator_config(
            simulator_type,
            self.robot_cfg,
            headless=headless,
            num_envs=self.num_envs,
            experiment_name="motion_viz_smoothness",
        )

        # Override robot asset settings for motion visualization
        self.robot_cfg.asset.disable_gravity = True
        self.robot_cfg.asset.fix_base_link = False
        self.robot_cfg.asset.self_collisions = False

        # Use torque control (zero torque) to maintain poses
        self.robot_cfg.control.control_type = ControlType.TORQUE

        # Create visualization markers
        self.viz_markers = (
            {} if args.hide_markers else self._create_visualization_markers()
        )

        # Initialize body markers after kinematic info is loaded
        if not args.hide_markers:
            self._initialize_body_markers()

        # Create custom key handlers for speed and threshold control
        custom_key_handlers = {
            "1": self.increase_speed,  # Key 1: Increase playback speed
            "2": self.decrease_speed,  # Key 2: Decrease playback speed
            "3": self.increase_smoothness_threshold,  # Key 3: Increase smoothness threshold
            "4": self.decrease_smoothness_threshold,  # Key 4: Decrease smoothness threshold
        }

        # Create checkerboard ground for visualization
        print("Creating checkerboard ground plane...")
        scene_lib = create_checkerboard_ground(
            self.num_envs, self.device, self.simulator_type
        )
        print("Checkerboard ground loaded successfully")
        terrain = None

        # Get simulator class and instantiate
        SimulatorClass = get_class(self.simulator_cfg._target_)

        extra_params = extra_simulator_params or {}
        self.simulator = SimulatorClass(
            config=self.simulator_cfg,
            robot_config=self.robot_cfg,
            terrain=terrain,
            device=self.device,
            scene_lib=scene_lib,
            custom_key_handlers=custom_key_handlers,
            **extra_params,
        )
        # Initialize the simulator with visualization markers
        self.simulator._initialize_with_markers(self.viz_markers)

        if args.record_video is not None:
            if simulator_type not in {"isaaclab", "newton"}:
                raise ValueError(
                    "--record-video is currently supported by IsaacLab and Newton"
                )
            if headless:
                raise ValueError("--record-video requires a visible simulator viewport")
            if simulator_type == "isaaclab":
                try:
                    from omni.kit.viewport.utility import get_active_viewport

                    get_active_viewport().set_texture_resolution(
                        (args.record_width, args.record_height)
                    )
                    print(
                        f"Recording resolution: {args.record_width}x{args.record_height}"
                    )
                except Exception as error:
                    print(f"Warning: could not set recording resolution: {error}")

        if simulator_type == "isaaclab" and args.camera_view != "default":
            camera_views = {
                "front": ([3.5, 0.0, 1.35], [0.0, 0.0, 0.9]),
                "side": ([0.0, 3.5, 1.35], [0.0, 0.0, 0.9]),
            }
            eye, target = camera_views[args.camera_view]
            if args.source_keypoints is not None and args.camera_view == "front":
                center_y = 0.5 * args.source_offset_y
                eye = [4.5, center_y, 1.5]
                target = [0.0, center_y, 0.9]
            self.simulator._fixed_camera_view = (eye, target)
            print(f"IsaacLab fixed camera view: {args.camera_view}")
        elif simulator_type == "newton" and args.camera_view != "default":
            camera_offsets = {
                # Kangaroo's forward axis points along +X. Place the camera on
                # +X and look back toward the torso for a true front view.
                "front": ([4.0, 0.0, 1.0], [0.0, 0.0, 0.2]),
                "side": ([0.0, -5.0, 1.0], [0.0, 0.0, 0.2]),
            }
            self.simulator._fixed_camera_offsets = camera_offsets[args.camera_view]
            print(f"Newton fixed camera view: {args.camera_view}")

        print(f"Loaded {robot_name} robot using {simulator_type}")
        print(f"Visualizing bodies: {self.robot_spec.viz_bodies}")
        vel_source = "data_vel" if self.use_data_vel else "finite_diff"
        print(
            f"Smoothness metric: {self.metric} | velocity source: {vel_source} | window: {self.window_frames} frames"
        )
        print(f"Smoothness threshold: {self.smoothness_threshold}")
        if args.hide_markers:
            print("Visualization markers: hidden")
        else:
            print("Visualization:")
            print("  Red spheres - Specified body markers")
            print("  Yellow spheres - Bodies exceeding smoothness threshold")
            print("  Purple spheres - Bodies in contact with ground")
        print("Controls:")
        print("  'R' - Switch to next motion")
        print("  '1' - Increase playback speed by 150% (NumPad 1 for IsaacLab)")
        print("  '2' - Decrease playback speed by 150% (NumPad 2 for IsaacLab)")
        print("  '3' - Increase smoothness threshold by 1.5x (NumPad 3 for IsaacLab)")
        print("  '4' - Decrease smoothness threshold by 1.5x (NumPad 4 for IsaacLab)")
        print("  'L' - Start/stop viewport video recording")
        print("  ';' - Cancel and delete the current recording")
        print("Motion will play automatically and loop")

        self.simulator.user_requested_reset = True

        # Speed control state
        self.speed_change_factor = 1.5  # 150% speed change
        self.min_speed = 0.01  # Minimum playback speed
        self.max_speed = 10.0  # Maximum playback speed

        # Pre-computed smoothness metrics for current motion
        # Shape: [num_frames, num_envs, num_bodies] - stores smoothness score per body per frame
        self.precomputed_smoothness = None

        # Pre-compute smoothness for the initial motion
        print("Pre-computing smoothness metrics for initial motion...")
        self._precompute_motion_smoothness()

    def _draw_source_skeleton(self, frame: int, root_position=None) -> None:
        """Draw the synchronized SOMA23 source skeleton beside the robot."""
        if self.source_positions is None or self.source_draw is None:
            return
        points = self.source_positions[frame % len(self.source_positions)].copy()
        if args.follow_camera and root_position is not None:
            points[:, :2] += np.asarray(root_position[:2], dtype=np.float32)
        starts = [tuple(points[start]) for start, _ in SOURCE_SKELETON_EDGES]
        ends = [tuple(points[end]) for _, end in SOURCE_SKELETON_EDGES]
        self.source_draw.clear_points()
        self.source_draw.clear_lines()
        self.source_draw.draw_points(
            [tuple(point) for point in points],
            [(0.1, 0.45, 1.0, 1.0)] * len(points),
            [12.0] * len(points),
        )
        self.source_draw.draw_lines(
            starts,
            ends,
            [(0.05, 0.05, 0.05, 1.0)] * len(starts),
            [3.0] * len(starts),
        )

    def _create_visualization_markers(self) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for specified body locations"""
        # Create one marker config for each body we want to visualize
        marker_configs = [
            MarkerConfig(size="regular") for _ in self.robot_spec.viz_bodies
        ]

        # Yellow joint markers for ALL bodies (get count from kinematic_info)
        # Note: kinematic_info will be set after _create_simulator_config is called
        self.joint_marker_name = "joint_highlight_markers"
        # We'll create these markers in the simulator initialization

        # Purple contact markers for ALL bodies
        self.contact_marker_name = "contact_markers"
        # We'll create these markers in the simulator initialization

        # Create visualization marker groups (initially empty, will be populated after config loading)
        markers = {
            "body_markers": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=marker_configs
            ),
        }

        return markers

    def _initialize_body_markers(self):
        """Initialize body markers after kinematic info is loaded"""
        if self.kinematic_info is None:
            return

        num_bodies = self.kinematic_info.num_bodies
        joint_marker_configs = [MarkerConfig(size="regular") for _ in range(num_bodies)]

        contact_marker_configs = [
            MarkerConfig(size="regular")  # Smaller size for contact markers
            for _ in range(num_bodies)
        ]

        # Add the body markers to the existing visualization markers
        self.viz_markers[self.joint_marker_name] = VisualizationMarkerConfig(
            type="sphere",
            color=(1.0, 1.0, 0.0),  # yellow
            markers=joint_marker_configs,
        )

        self.viz_markers[self.contact_marker_name] = VisualizationMarkerConfig(
            type="sphere",
            color=(0.8, 0.0, 0.8),  # purple
            markers=contact_marker_configs,
        )

    def _switch_to_next_motion(self):
        """Switch to the next motion in the dataset"""
        self.current_motion_idx = (self.current_motion_idx + 1) % self.total_motions
        self.current_frame = 0
        self.current_motion_length = (
            self.motion_libs[0]
            .get_motion_num_frames(None)[self.current_motion_idx]
            .item()
        )

        print(
            f"Switched to motion {self.current_motion_idx}/{self.total_motions-1} "
            f"(length: {self.current_motion_length} frames)"
        )
        print(
            f"Current motion: {self.motion_libs[0].motion_files[self.current_motion_idx]}"
        )

        # Pre-compute smoothness for new motion
        print("Pre-computing smoothness metrics for new motion...")
        self._precompute_motion_smoothness()

    def _precompute_motion_smoothness(self):
        """Pre-compute smoothness metrics for the entire current motion"""
        motion_idx = torch.tensor(
            [self.current_motion_idx], device=self.device, dtype=torch.long
        )
        dt = 1.0 / FPS

        # Load all frames for all environments
        all_positions = []
        all_velocities = []

        for frame_idx in range(self.current_motion_length):
            frame_tensor = torch.tensor([frame_idx], device=self.device)

            # Get state for all environments
            pos_list = []
            vel_list = []
            for motion_lib in self.motion_libs:
                state = motion_lib.get_motion_state_exact_frame(
                    motion_idx, frame_tensor
                )
                pos_list.append(state.rigid_body_pos[0])  # [num_bodies, 3]
                if state.rigid_body_vel is not None:
                    vel_list.append(state.rigid_body_vel[0])
                else:
                    vel_list.append(torch.zeros_like(state.rigid_body_pos[0]))

            # Stack: [num_envs, num_bodies, 3]
            all_positions.append(torch.stack(pos_list, dim=0))
            all_velocities.append(torch.stack(vel_list, dim=0))

        # Stack to [num_frames, num_envs, num_bodies, 3]
        positions_tensor = torch.stack(all_positions, dim=0)
        velocities_tensor = torch.stack(all_velocities, dim=0)

        T, E, B, _ = positions_tensor.shape

        # Compute smoothness using sliding window
        # Result shape: [num_frames, num_envs, num_bodies]
        smoothness_scores = torch.zeros(T, E, B, device=self.device)

        for frame_idx in range(T):
            # Get window around this frame
            window_start = max(0, frame_idx - self.window_frames // 2)
            window_end = min(T, frame_idx + self.window_frames // 2 + 1)

            if window_end - window_start < 4:  # Need at least 4 frames for jerk
                continue

            # Get windowed data
            pos_window = positions_tensor[window_start:window_end]  # [W, E, B, 3]
            vel_window = velocities_tensor[window_start:window_end]  # [W, E, B, 3]

            W = pos_window.shape[0]
            N = E * B

            # Reshape to [W, N, 3]
            pos_reshaped = pos_window.view(W, N, 3)
            vel_reshaped = vel_window.view(W, N, 3)

            # Use finite differences if configured
            if not self.use_data_vel:
                vel_reshaped = _diff(pos_reshaped, dt)
                # Pad velocity
                if vel_reshaped.shape[0] >= 2:
                    v_extrapolated = 2 * vel_reshaped[:1] - vel_reshaped[1:2]
                else:
                    v_extrapolated = torch.zeros_like(vel_reshaped[:1])
                vel_reshaped = torch.cat([v_extrapolated, vel_reshaped], dim=0)

            # Compute smoothness metric
            if self.metric == "nj":
                per_body_scores, _ = normalized_jerk_from_vel(vel_reshaped, dt)
            elif self.metric == "oi":
                per_body_scores, _ = oscillation_index_from_vel(vel_reshaped, dt)
            else:  # pj
                per_body_scores, _ = purposeful_jerk_from_vel(vel_reshaped, dt)

            # Reshape back to [E, B]
            per_body_scores = per_body_scores.view(E, B)
            smoothness_scores[frame_idx] = per_body_scores

        # Store pre-computed scores
        self.precomputed_smoothness = smoothness_scores
        print(f"Smoothness pre-computed for {T} frames")

    def _get_current_pose(self):
        """Get the current pose for the selected motion and frame using MotionLib API for all environments"""
        motion_idx = torch.tensor(
            [self.current_motion_idx], device=self.device, dtype=torch.long
        )
        clamped_frame = min(self.current_frame, self.current_motion_length - 1)

        # Get poses from all motion libraries
        dof_pos_list = []
        rigid_body_pos_list = []
        rigid_body_rot_list = []
        rigid_body_vel_list = []

        for motion_lib in self.motion_libs:
            state = motion_lib.get_motion_state_exact_frame(
                motion_idx, torch.tensor([clamped_frame], device=self.device)
            )
            dof_pos_list.append(state.dof_pos[0])
            rigid_body_pos_list.append(state.rigid_body_pos[0])
            rigid_body_rot_list.append(state.rigid_body_rot[0])
            # Handle case where rigid_body_vel might be None
            if state.rigid_body_vel is not None:
                rigid_body_vel_list.append(state.rigid_body_vel[0])
            else:
                rigid_body_vel_list.append(torch.zeros_like(state.rigid_body_pos[0]))

        # Stack to create batch dimension for environments
        dof_pos = torch.stack(dof_pos_list, dim=0)  # [num_envs, num_dofs]
        rigid_body_pos = torch.stack(
            rigid_body_pos_list, dim=0
        )  # [num_envs, num_bodies, 3]
        rigid_body_rot = torch.stack(
            rigid_body_rot_list, dim=0
        )  # [num_envs, num_bodies, 4]
        rigid_body_vel = torch.stack(
            rigid_body_vel_list, dim=0
        )  # [num_envs, num_bodies, 3]

        return dof_pos, rigid_body_pos, rigid_body_rot, rigid_body_vel

    def _update_contact_markers(self) -> Dict[str, MarkerState]:
        """Update contact markers to show which bodies are in contact with the ground."""
        # Get contact data for current frame from the first motion library
        motion_idx = torch.tensor(
            [self.current_motion_idx], device=self.device, dtype=torch.long
        )
        clamped_frame = min(self.current_frame, self.current_motion_length - 1)

        # Get contact state from motion library
        contact_states = []
        for motion_lib in self.motion_libs:
            state = motion_lib.get_motion_state_exact_frame(
                motion_idx, torch.tensor([clamped_frame], device=self.device)
            )
            if state.rigid_body_contacts is not None:
                contact_states.append(state.rigid_body_contacts[0])  # [num_bodies]
            else:
                # Fallback if no contact data
                contact_states.append(
                    torch.zeros(
                        self.kinematic_info.num_bodies,
                        dtype=torch.bool,
                        device=self.device,
                    )
                )

        # Stack contact states for all environments
        contact_mask = torch.stack(contact_states, dim=0)  # [num_envs, num_bodies]

        # Get positions/orientations for ALL bodies
        all_body_state = self.simulator.get_bodies_state()
        all_translations = (
            all_body_state.rigid_body_pos.detach().clone()
        )  # [num_envs, all_bodies, 3]
        all_orientations = (
            all_body_state.rigid_body_rot.detach().clone()
        )  # [num_envs, all_bodies, 4]

        # Only show contact markers for bodies that are in contact
        # Hide non-contact markers below ground
        mask = contact_mask.unsqueeze(-1)  # [num_envs, all_bodies, 1]
        hidden_pos = torch.tensor([0.0, 0.0, -100.0], device=self.device).view(1, 1, 3)
        contact_translations = torch.where(mask, all_translations, hidden_pos)

        # # Offset contact markers slightly below the body center for visibility
        # contact_offset = torch.tensor([0.0, 0.0, -0.05], device=self.device).view(1, 1, 3)
        # contact_translations = torch.where(mask, contact_translations + contact_offset, hidden_pos)

        return {
            self.contact_marker_name: MarkerState(
                translation=contact_translations, orientation=all_orientations
            )
        }

    def _update_joint_highlights(self) -> Dict[str, MarkerState]:
        """Get which joints to highlight based on pre-computed smoothness metrics and return marker states."""

        # Look up pre-computed smoothness for current frame
        clamped_frame = min(self.current_frame, self.current_motion_length - 1)

        if (
            self.precomputed_smoothness is None
            or clamped_frame >= self.precomputed_smoothness.shape[0]
        ):
            # No pre-computed data available, no highlighting
            self.highlight_mask = torch.zeros(
                self.num_envs,
                self.kinematic_info.num_bodies,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            # Get pre-computed scores for this frame: [num_envs, num_bodies]
            per_body_scores = self.precomputed_smoothness[clamped_frame]

            # Determine which bodies exceed threshold
            highlight = (
                per_body_scores > self.smoothness_threshold
            )  # [num_envs, num_bodies]
            self.highlight_mask = highlight

        # Get positions/orientations for ALL bodies
        all_body_state = self.simulator.get_bodies_state()
        all_translations = (
            all_body_state.rigid_body_pos.detach().clone()
        )  # [num_envs, all_bodies, 3]
        all_orientations = (
            all_body_state.rigid_body_rot.detach().clone()
        )  # [num_envs, all_bodies, 4]

        # Only show for highlighted bodies by hiding non-highlighted markers below ground
        mask = self.highlight_mask.unsqueeze(-1)  # [num_envs, all_bodies, 1]
        hidden_pos = torch.tensor([0.0, 0.0, -100.0], device=self.device).view(1, 1, 3)
        translations = torch.where(mask, all_translations, hidden_pos)

        return {
            self.joint_marker_name: MarkerState(
                translation=translations, orientation=all_orientations
            )
        }

    def _set_robot_pose(self, dof_pos, rigid_body_pos=None, rigid_body_rot=None):
        """Set the robot to the specified pose"""
        # Kinematic playback already provides root + DOF state. Avoid reading
        # the complete robot state back through Fabric before every frame;
        # that readback can stall when Fabric GPU properties are not updated.
        root_pos = rigid_body_pos[:, 0, :].detach()
        root_rot = rigid_body_rot[:, 0, :].detach()
        zero_root = torch.zeros(self.num_envs, 3, device=self.device)
        reset_state = ResetState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=zero_root,
            root_ang_vel=zero_root.clone(),
            dof_pos=dof_pos.detach(),
            dof_vel=torch.zeros_like(dof_pos),
            state_conversion=StateConversion.COMMON,
        )

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.reset_envs(reset_state, env_ids=env_ids)

    def _get_updated_marker_positions(self):
        """Update marker positions to follow the specified bodies"""
        if not self.viz_markers:
            return

        # this will convert to sim common ordering, which is the MJCF ordering
        current_state = self.simulator.get_bodies_state()

        idx_in_common = [
            self.simulator._body_names.index(body_name)
            for body_name in self.robot_spec.viz_bodies
        ]

        all_positions = (
            current_state.rigid_body_pos[:, idx_in_common, :].detach().clone()
        )
        all_orientations = (
            current_state.rigid_body_rot[:, idx_in_common, :].detach().clone()
        )

        marker_states = {}

        marker_states["body_markers"] = MarkerState(
            translation=all_positions, orientation=all_orientations
        )

        # Add/update joint highlight markers
        joint_marker_states = self._update_joint_highlights()
        marker_states.update(joint_marker_states)

        # Add/update contact markers
        contact_marker_states = self._update_contact_markers()
        marker_states.update(contact_marker_states)

        return marker_states

    def increase_speed(self):
        """Increase playback speed by the speed change factor"""
        new_speed = min(self.playback_speed * self.speed_change_factor, self.max_speed)
        if new_speed != self.playback_speed:
            self.playback_speed = new_speed
            print(f"Playback speed increased to {self.playback_speed:.3f}x")
            return True
        return False

    def decrease_speed(self):
        """Decrease playback speed by the speed change factor"""
        new_speed = max(self.playback_speed / self.speed_change_factor, self.min_speed)
        if new_speed != self.playback_speed:
            self.playback_speed = new_speed
            print(f"Playback speed decreased to {self.playback_speed:.3f}x")
            return True
        return False

    def increase_smoothness_threshold(self):
        """Increase smoothness threshold by 1.5x"""
        self.smoothness_threshold *= 1.5
        print(f"Smoothness threshold increased to {self.smoothness_threshold:.3f}")

    def decrease_smoothness_threshold(self):
        """Decrease smoothness threshold by 1.5x"""
        new_threshold = max(
            self.smoothness_threshold / 1.5, 0.001
        )  # Minimum threshold of 0.001
        if new_threshold != self.smoothness_threshold:
            self.smoothness_threshold = new_threshold
            print(f"Smoothness threshold decreased to {self.smoothness_threshold:.3f}")
        else:
            print(f"Smoothness threshold at minimum: {self.smoothness_threshold:.3f}")

    def run(self):
        """Main simulation loop"""
        step_count = 0
        marker_states = None
        target_dt = 1.0 / FPS  # wall-clock time per motion frame
        first_playback_step = True
        automatic_recording = args.record_video is not None
        recorded_frames = 0
        recording_limit = (
            args.record_frames
            if args.record_frames > 0
            else self.current_motion_length
        )

        if automatic_recording:
            args.record_video.parent.mkdir(parents=True, exist_ok=True)
            self.playback_speed = 1.0
            self.current_frame = 0
            self.simulator._toggle_video_record()
            print(
                f"Automatic {self.simulator_type} recording: "
                f"{recording_limit} frames -> "
                f"{args.record_video}"
            )

        print(
            "Playback loop running in the simulator window; "
            "the terminal remains quiet during normal playback."
        )

        while True:
            frame_start = time.perf_counter()

            # Check for reset request (R key press triggers this in simulator)
            if self.simulator.user_requested_reset:
                self._switch_to_next_motion()
                self.simulator.user_requested_reset = False

            # Calculate playback parameters based on speed
            # For speed < 1.0: slow down by updating motion less frequently (frames_per_step > 1)
            # For speed >= 1.0: speed up by skipping motion frames (frame_skip > 1)
            if self.playback_speed < 1.0:
                frames_per_step = max(1, int(1.0 / self.playback_speed))
                frame_skip = 1  # Don't skip frames when slowing down
            else:
                frames_per_step = 1  # Update every step when speeding up
                frame_skip = max(
                    1, int(self.playback_speed)
                )  # Skip frames for fast playback

            # Update motion frame based on playback speed
            if step_count % frames_per_step == 0:
                playback_frame = self.current_frame
                # Get current pose for display
                dof_pos, rigid_body_pos, rigid_body_rot, _ = self._get_current_pose()

                if first_playback_step:
                    print(
                        f"First motion pose loaded: frame={self.current_frame}, "
                        f"dofs={dof_pos.shape[-1]}"
                    )

                # Set robot pose
                self._set_robot_pose(dof_pos, rigid_body_pos, rigid_body_rot)
                root_position = rigid_body_pos[0, 0].detach().cpu().numpy()
                if args.follow_camera and args.camera_view != "default":
                    if self.simulator_type == "isaaclab":
                        center_y = (
                            0.5 * args.source_offset_y
                            if args.source_keypoints is not None
                            else 0.0
                        )
                        if args.camera_view == "front":
                            eye_offset = np.asarray([4.5, center_y, 1.5])
                            target_offset = np.asarray([0.0, center_y, 0.9])
                        else:
                            eye_offset = np.asarray([0.0, 3.5, 1.35])
                            target_offset = np.asarray([0.0, 0.0, 0.9])
                        follow_offset = np.asarray(
                            [root_position[0], root_position[1], 0.0]
                        )
                        self.simulator._fixed_camera_view = (
                            (eye_offset + follow_offset).tolist(),
                            (target_offset + follow_offset).tolist(),
                        )
                self._draw_source_skeleton(playback_frame, root_position)
                if first_playback_step:
                    print(f"First motion pose written to {self.simulator_type}")

                # Advance frame with skip for fast playback
                self.current_frame += frame_skip

                # Loop motion when finished
                if self.current_frame >= self.current_motion_length:
                    self.current_frame = 0

            # Zero torque control to maintain pose
            # Kangaroo observes 32 joint states but controls only 28 actuated
            # joints. Use the simulator's actual action tensor shape instead
            # of assuming one action per observed DOF.
            _common_actions = torch.zeros_like(
                self.simulator.get_current_actions()
            )

            if marker_states is None or step_count % frames_per_step == 0:
                marker_states = self._get_updated_marker_positions()

            self.simulator.step(_common_actions, markers_callback=lambda: marker_states)
            if automatic_recording:
                recorded_frames += 1
                if recorded_frames >= recording_limit:
                    self.simulator._toggle_video_record()
                    # One render processes the stop transition and performs
                    # synchronous H.264 encoding in RecordingMixin.
                    self.simulator.render()
                    generated_video = Path(
                        f"{self.simulator._curr_user_recording_name}.mp4"
                    )
                    if not generated_video.is_file():
                        raise RuntimeError(
                            f"IsaacLab recording was not created: {generated_video}"
                        )
                    shutil.move(str(generated_video), str(args.record_video))
                    print(
                        f"Automatic {self.simulator_type} video saved: "
                        f"{args.record_video}"
                    )
                    return
            if first_playback_step:
                print(f"First {self.simulator_type} playback step completed")
                first_playback_step = False

            step_count += 1

            # Throttle to real-time (adjusted by playback speed)
            elapsed = time.perf_counter() - frame_start
            sleep_time = target_dt / max(self.playback_speed, 0.01) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    # Use the global args that were parsed early
    global args, AppLauncher

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Extra simulator parameters for IsaacLab
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {
            "headless": args.headless,
            "device": str(device),
            # # Performance settings for faster-than-realtime rendering
            # "rendering_mode": "performance",  # Options: "performance", "balanced", "quality"
        }
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    visualizer = MotionVisualizerSmoothness(
        motion_files=args.motion_files,
        robot_name=args.robot,
        simulator_type=args.simulator,
        headless=args.headless,
        cpu_only=args.cpu_only,
        extra_simulator_params=extra_simulator_params,
        playback_speed=args.playback_speed,
        metric=args.metric,
        use_data_vel=args.use_data_vel,
        window_sec=args.window_sec,
    )

    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.simulator.close()


if __name__ == "__main__":
    main()
