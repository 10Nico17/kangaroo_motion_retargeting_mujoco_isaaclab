"""Load the Kangaroo MJCF directly in MuJoCo without retargeting overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading
import time

import glfw
import mujoco
import mujoco.viewer


DEFAULT_MODEL = Path(
    "protomotions/data/assets/Kangaroo/kangaroo_grippers_ias.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--keyframe",
        default="init_state",
        help="MJCF keyframe to display; use an empty string for qpos0.",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Advance the original MJCF physics instead of holding the pose.",
    )
    return parser.parse_args()


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"


def print_leg_hierarchy(model: mujoco.MjModel) -> None:
    print("\nKangaroo leg body hierarchy from the loaded XML")
    print("-" * 72)
    for body_id in range(1, model.nbody):
        name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not (name.startswith("leg_left") or name.startswith("leg_right")):
            continue
        parent_id = int(model.body_parentid[body_id])
        parent_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        side = "LEFT " if name.startswith("leg_left") else "RIGHT"
        print(f"{side}  {parent_name:28s} -> {name}")


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {model_path}")

    # This is a direct load: the XML is not copied or modified.
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.keyframe:
        key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
        )
        if key_id < 0:
            available = [
                object_name(model, mujoco.mjtObj.mjOBJ_KEY, index)
                for index in range(model.nkey)
            ]
            raise ValueError(
                f"Keyframe {args.keyframe!r} not found. Available: {available}"
            )
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"Loaded XML directly: {model_path}")
    print(f"Pose: {args.keyframe or 'qpos0'}")
    print(f"Bodies: {model.nbody}, joints: {model.njnt}, equalities: {model.neq}")
    print(f"Physics: {'running' if args.physics else 'disabled (static pose)'}")
    print_leg_hierarchy(model)
    print("\nQ or Esc: close viewer cleanly")

    stop_requested = threading.Event()

    def key_callback(keycode: int) -> None:
        if keycode in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            stop_requested.set()

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def request_shutdown(_signum, _frame) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_shutdown)
    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.85)
            viewer.cam.distance = 2.7
            viewer.cam.azimuth = 180.0
            viewer.cam.elevation = -10.0

            previous_time = time.monotonic()
            while viewer.is_running() and not stop_requested.is_set():
                if args.physics:
                    now = time.monotonic()
                    elapsed = now - previous_time
                    previous_time = now
                    steps = max(1, int(elapsed / model.opt.timestep))
                    for _ in range(min(steps, 20)):
                        mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep * 0.5))
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        print("MuJoCo XML viewer closed.")


if __name__ == "__main__":
    main()
