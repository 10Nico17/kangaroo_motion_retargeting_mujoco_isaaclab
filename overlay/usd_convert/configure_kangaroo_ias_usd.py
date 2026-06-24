"""Restore IAS actuators, initial state, and loop constraints in imported USD."""

from math import degrees, pi
from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import mujoco  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "protomotions/data/assets/Kangaroo/kangaroo_grippers_ias.xml"
IMPORTED_USD = (
    ROOT
    / "protomotions/data/assets/Kangaroo/usd/kangaroo_grippers_ias"
    / "kangaroo_grippers_ias.usd"
)
OUTPUT_USD = IMPORTED_USD.with_name("kangaroo_grippers_ias_configured.usd")
KEYFRAME = "init_state"
LOOP_NAMES = {
    "leg_left_4_link_leg_left_knee_link",
    "leg_right_4_link_leg_right_knee_link",
}
FIXED_HELPER_BODIES = {"leg_left_ankle_link", "leg_right_ankle_link"}


def joint_prims_by_name(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    result = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            result[prim.GetName()] = prim
    return result


def remove_drive(prim: Usd.Prim, axis: str) -> None:
    prim.RemoveAPI(UsdPhysics.DriveAPI, axis)
    for prop in list(prim.GetProperties()):
        if prop.GetName().startswith(f"drive:{axis}:"):
            prim.RemoveProperty(prop.GetName())


def restore_initial_state_and_drives(stage: Usd.Stage) -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
    if key_id < 0:
        raise RuntimeError(f"Missing MJCF keyframe: {KEYFRAME}")

    prims = joint_prims_by_name(stage)
    key_qpos = model.key_qpos[key_id]
    key_ctrl = model.key_ctrl[key_id]

    # Author the complete scalar-joint state, including passive femur/knee joints.
    state_count = 0
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        prim = prims.get(name)
        if prim is None:
            raise RuntimeError(f"USD is missing MJCF joint: {name}")
        qpos = float(key_qpos[model.jnt_qposadr[joint_id]])
        axis = "angular" if joint_type == mujoco.mjtJoint.mjJNT_HINGE else "linear"
        value = degrees(qpos) if axis == "angular" else qpos
        state = PhysxSchema.JointStateAPI.Apply(prim, axis)
        state.CreatePositionAttr(value)
        state.CreateVelocityAttr(0.0)
        state_count += 1

    active_names = set()
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_type = model.jnt_type[joint_id]
        axis = "angular" if joint_type == mujoco.mjtJoint.mjJNT_HINGE else "linear"
        prim = prims.get(name)
        if prim is None:
            raise RuntimeError(f"USD is missing actuator joint: {name}")

        kp = float(model.actuator_gainprm[actuator_id, 0])
        kd = -float(model.actuator_biasprm[actuator_id, 2])
        target = float(key_ctrl[actuator_id])
        max_force = max(abs(float(x)) for x in model.actuator_forcerange[actuator_id])

        # USD angular drives operate in degrees, while MJCF gains use radians.
        if axis == "angular":
            kp *= pi / 180.0
            kd *= pi / 180.0
            target = degrees(target)

        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(kp)
        drive.CreateDampingAttr(kd)
        drive.CreateMaxForceAttr(max_force)
        drive.CreateTargetPositionAttr(target)
        drive.CreateTargetVelocityAttr(0.0)
        active_names.add(name)

    # The importer adds zero-force drives to passive joints; remove them entirely.
    for name, prim in prims.items():
        if name in active_names or name in LOOP_NAMES:
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            remove_drive(prim, "angular")
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            remove_drive(prim, "linear")

    print(f"Authored {state_count} initial joint positions from '{KEYFRAME}'")
    print(f"Configured {len(active_names)} MJCF actuators")


def replace_loop_joints(stage: Usd.Stage) -> None:
    for name in LOOP_NAMES:
        old = stage.GetPrimAtPath(f"{stage.GetDefaultPrim().GetPath()}/loop_joints/{name}")
        if not old.IsValid():
            raise RuntimeError(f"Missing imported loop joint: {name}")
        old_joint = UsdPhysics.Joint(old)
        body0 = old_joint.GetBody0Rel().GetTargets()
        body1 = old_joint.GetBody1Rel().GetTargets()
        pos0 = old_joint.GetLocalPos0Attr().Get()
        pos1 = old_joint.GetLocalPos1Attr().Get()
        path = old.GetPath()
        stage.RemovePrim(path)

        joint = UsdPhysics.SphericalJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets(body0)
        joint.CreateBody1Rel().SetTargets(body1)
        joint.CreateLocalPos0Attr(pos0)
        joint.CreateLocalPos1Attr(pos1)
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        print(f"Replaced loop with PhysicsSphericalJoint: {path}")


def configure_fixed_helpers(stage: Usd.Stage) -> None:
    """Keep fixed ankle bodies attached and give PhysX valid tiny inertias."""
    for name in FIXED_HELPER_BODIES:
        body = next(
            (p for p in stage.Traverse() if p.GetName() == name and not p.IsA(UsdPhysics.Joint)),
            None,
        )
        joint = next(
            (p for p in stage.Traverse() if p.GetName() == name and p.IsA(UsdPhysics.FixedJoint)),
            None,
        )
        if body is None or joint is None:
            raise RuntimeError(f"Could not find fixed helper body and joint: {name}")
        mass = UsdPhysics.MassAPI.Apply(body)
        # These bodies only group collision helpers and are welded to the foot.
        # A negligible positive mass avoids PhysX's invalid-inertia fallback
        # without materially changing the robot dynamics.
        mass.CreateMassAttr(0.001)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-7, 1.0e-7, 1.0e-7))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        print(f"Configured fixed ankle helper mass/inertia: {name}")


def configure_single_articulation_root(stage: Usd.Stage) -> None:
    """Keep only the physical base as the articulation root.

    The MJCF importer also marks its synthetic ``worldBody`` scope as an
    articulation root.  Isaac Lab resolves both roots below the spawned asset
    and rejects the articulation as ambiguous.
    """
    base_body = next(
        (
            prim
            for prim in stage.Traverse()
            if prim.GetName() == "base_link"
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        None,
    )
    if base_body is None:
        raise RuntimeError("Could not find the Kangaroo base rigid body")

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        if prim == base_body:
            continue
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print(f"Removed extra articulation root: {prim.GetPath()}")

    UsdPhysics.ArticulationRootAPI.Apply(base_body)
    print(f"Configured articulation root: {base_body.GetPath()}")


def hide_collision_geometry(stage: Usd.Stage) -> None:
    """Hide collision meshes visually without disabling their physics APIs."""
    # The MJCF importer authors many visual/collision scopes as instances.
    # Visibility cannot be authored on descendants of instance proxies. Make
    # those roots editable in the output asset first; physics APIs remain
    # untouched.
    deinstanced = 0
    for _ in range(16):
        instance_roots = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instance_roots:
            break
        for prim in instance_roots:
            prim.SetInstanceable(False)
            deinstanced += 1

    hidden = 0
    for prim in stage.Traverse():
        path_parts = prim.GetPath().pathString.split("/")
        is_collision_scope = "collisions" in path_parts
        is_named_collision = any(
            "collision" in part.lower() for part in path_parts
        )
        is_collision_prim = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_visual_prim = "visuals" in path_parts
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        material_name = material.GetPrim().GetName().lower() if material else ""
        if not (
            is_collision_scope
            or is_named_collision
            or "bright_orange" in material_name
            or (is_collision_prim and not is_visual_prim)
        ):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
            hidden += 1
    print(
        f"Hid {hidden} collision-geometry scopes "
        f"after de-instancing {deinstanced} prims"
    )


def main() -> None:
    imported = Usd.Stage.Open(str(IMPORTED_USD))
    if imported is None:
        raise RuntimeError(f"Cannot open imported USD: {IMPORTED_USD}")
    flattened = imported.Flatten()
    if not flattened.Export(str(OUTPUT_USD)):
        raise RuntimeError(f"Cannot export: {OUTPUT_USD}")

    stage = Usd.Stage.Open(str(OUTPUT_USD))
    restore_initial_state_and_drives(stage)
    replace_loop_joints(stage)
    configure_fixed_helpers(stage)
    configure_single_articulation_root(stage)
    hide_collision_geometry(stage)
    stage.GetRootLayer().Save()
    print(f"Wrote configured Kangaroo IAS USD: {OUTPUT_USD}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
