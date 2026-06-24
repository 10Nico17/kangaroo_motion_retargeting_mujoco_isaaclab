"""Convert the Kangaroo IAS MJCF with the importer explicitly enabled."""

from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import omni.kit.app  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "protomotions/data/assets/Kangaroo/kangaroo_grippers_ias.xml"
)
OUTPUT_DIR = ROOT / "protomotions/data/assets/Kangaroo/usd/kangaroo_grippers_ias"
OUTPUT_NAME = "kangaroo_grippers_ias.usd"


def main() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
    for _ in range(3):
        omni.kit.app.get_app().update()

    # Import only after enabling the extension so its Kit commands are registered.
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = MjcfConverterCfg(
        asset_path=str(INPUT),
        usd_dir=str(OUTPUT_DIR),
        usd_file_name=OUTPUT_NAME,
        fix_base=False,
        import_sites=True,
        force_usd_conversion=True,
        make_instanceable=False,
    )
    converter = MjcfConverter(cfg)
    print(f"Generated USD: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
