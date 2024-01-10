import tyro
from gs_toolkit.data.process_data.colmap_converter_to_gstk_dataset import (
    ColmapConverterToGstkDataset,
)
from gs_toolkit.utils.rich_utils import CONSOLE


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    try:
        tyro.cli(ColmapConverterToGstkDataset).main()
    except Exception as e:
        CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()
