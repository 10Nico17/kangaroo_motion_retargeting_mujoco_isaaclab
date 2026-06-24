"""Visualize and test the configured kangaroo_grippers_ias USD in Isaac Lab."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics", action="store_true", help="Advance PhysX.")
parser.add_argument(
    "--show-collisions",
    action="store_true",
    help="Render collision geometry. Hidden by default; physics stays enabled.",
)
parser.add_argument(
    "--diagnose-geometry",
    action="store_true",
    help="Print every rendered Kangaroo geometry and exit after one frame.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
import carb  # noqa: E402
import omni.usd  # noqa: E402
import omni.physx.bindings._physx as physx_bindings  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


USD_PATH = (
    Path(__file__).resolve().parents[1]
    / "protomotions/data/assets/Kangaroo/usd/kangaroo_grippers_ias"
    / "kangaroo_grippers_ias_configured.usd"
)


def loop_errors_mm() -> dict[str, float]:
    stage = omni.usd.get_context().get_stage()
    result = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.GetParent().GetName() != "loop_joints":
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        world0 = UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world1 = UsdGeom.Xformable(body1).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        point0 = world0.Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        point1 = world1.Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        result[prim.GetName()] = (point0 - point1).GetLength() * 1000.0
    return result


def print_summary() -> None:
    stage = omni.usd.get_context().get_stage()
    prims = list(stage.Traverse())
    joints = [p for p in prims if "Joint" in p.GetTypeName()]
    drives = [
        p
        for p in joints
        if any(schema.startswith("PhysicsDriveAPI:") for schema in p.GetAppliedSchemas())
    ]
    states = [
        p
        for p in joints
        if any(schema.startswith("PhysicsJointStateAPI:") for schema in p.GetAppliedSchemas())
    ]
    loops = [p for p in joints if p.GetParent().GetName() == "loop_joints"]
    print(f"Loaded Kangaroo IAS USD: {USD_PATH}")
    print(f"Physics joints: {len(joints)}")
    print(f"Configured drives: {len(drives)}")
    print(f"Initial joint states: {len(states)}")
    print(f"Spherical loop joints: {len(loops)}")


def hide_collision_geometry() -> None:
    """Hide imported collision meshes without disabling collision physics."""
    if args.show_collisions:
        return

    stage = omni.usd.get_context().get_stage()

    # Collision helpers live inside instance proxies. Authoring visibility on
    # their nearest instance root can hide an entire robot link, including its
    # real mesh. De-instance only the composed viewer stage first, so each
    # individual helper becomes editable without modifying the source USD.
    deinstanced = 0
    for _ in range(8):
        instance_roots = [
            prim
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Kangaroo")
            and prim.IsInstance()
        ]
        if not instance_roots:
            break
        for prim in instance_roots:
            prim.SetInstanceable(False)
            deinstanced += 1

    hidden_paths = set()
    orange_material_matches = 0
    for prim in stage.Traverse():
        path_parts = prim.GetPath().pathString.split("/")
        is_collision_scope = "collisions" in path_parts
        # MJCF imports may put collision helpers below a ``visuals`` scope,
        # e.g. ``pelvis_2_collision``. Their name still identifies them as
        # collision render geometry.
        is_named_collision = any("collision" in part.lower() for part in path_parts)
        is_collision_prim = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_visual_prim = "visuals" in path_parts
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        material_name = material.GetPrim().GetName().lower() if material else ""
        # Collision geoms use this MJCF material even when the importer gives
        # them anonymous names such as ``_geom_47`` below ``visuals``.
        has_collision_material = "bright_orange" in material_name
        if has_collision_material:
            orange_material_matches += 1
        if not (
            is_collision_scope
            or is_named_collision
            or has_collision_material
            or (is_collision_prim and not is_visual_prim)
        ):
            continue

        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        imageable.MakeInvisible()
        hidden_paths.add(prim.GetPath().pathString)
    print(
        f"Hidden collision visuals: {len(hidden_paths)} prims "
        f"({orange_material_matches} bright_orange material matches; "
        f"de-instanced {deinstanced}; physics remains active)"
    )


def diagnose_render_geometry() -> None:
    """Report the authored geometry that remains visible after filtering."""
    stage = omni.usd.get_context().get_stage()
    visible = []
    invisible = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.GetPath().pathString.startswith("/World/Kangaroo"):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        entry = {
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
            "material": material.GetPath().pathString if material else "<none>",
            "collision_api": prim.HasAPI(UsdPhysics.CollisionAPI),
            "visibility": str(imageable.ComputeVisibility()),
        }
        if entry["visibility"] == "invisible":
            invisible.append(entry)
        else:
            visible.append(entry)

    print("\n=== Kangaroo render-geometry diagnosis ===")
    print(f"Visible Gprims: {len(visible)} | Invisible Gprims: {len(invisible)}")
    for entry in visible:
        print(
            "VISIBLE "
            f"type={entry['type']:<10} collision_api={str(entry['collision_api']):<5} "
            f"material={entry['material']} path={entry['path']}"
        )
    print("=== End geometry diagnosis ===\n")


def disable_physx_collision_debug() -> None:
    """Disable persistent wireframe and solid collider debug overlays."""
    settings = carb.settings.get_settings()
    settings.set_int(physx_bindings.SETTING_DISPLAY_COLLIDERS, 0)
    settings.set_bool(
        physx_bindings.SETTING_VISUALIZATION_COLLISION_MESH, False
    )
    # Same setting used by Isaac Sim's Physics viewport menu.
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 0)
    print("PhysX collider debug visualization: disabled")


def main() -> None:
    disable_physx_collision_debug()
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim.set_camera_view(eye=(3.5, 3.5, 2.2), target=(0.0, 0.0, 0.9))

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    robot = sim_utils.UsdFileCfg(usd_path=str(USD_PATH))
    robot.func("/World/Kangaroo", robot)
    hide_collision_geometry()

    sim.reset()
    disable_physx_collision_debug()
    print_summary()
    if args.diagnose_geometry:
        diagnose_render_geometry()
        sim.render()
        return
    print("PhysX running." if args.physics else "Static viewer; add --physics to simulate.")

    step = 0
    while simulation_app.is_running():
        if args.physics:
            sim.step()
            step += 1
            if step % 500 == 0:
                print(
                    "Loop drift [mm]: "
                    + ", ".join(f"{name}={value:.4f}" for name, value in loop_errors_mm().items())
                )
        else:
            sim.render()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
