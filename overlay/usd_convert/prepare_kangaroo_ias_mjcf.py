"""Create a relocatable copy of the current Kangaroo IAS MJCF."""

from pathlib import Path
import os
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "protomotions/data/assets/Kangaroo"
SOURCE = ASSET_ROOT / "kangaroo_grippers_ias.xml"
OUTPUT = ASSET_ROOT / "generated/kangaroo_grippers_ias_assets_resolved.xml"
MESH_ROOT = ROOT / "data/assets"


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("MJCF has no <compiler> element")

    # Keep the new left/right-specific meshes and only adjust their root path
    # for the generated XML's location.
    meshdir = Path(os.path.relpath(MESH_ROOT, OUTPUT.parent))
    compiler.set("meshdir", meshdir.as_posix())

    mesh_count = 0
    for mesh in root.findall("./asset/mesh"):
        relative_path = mesh.get("file")
        if not relative_path:
            raise ValueError(f"Mesh without file attribute: {mesh.attrib}")
        resolved = MESH_ROOT / relative_path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mesh_count += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="unicode")
    print(f"Validated {mesh_count} meshes from {MESH_ROOT}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
