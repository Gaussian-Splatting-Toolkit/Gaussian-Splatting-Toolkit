"""Processes a video or image sequence to a gs_toolkit compatible dataset."""

from dataclasses import dataclass
from typing import Union
import tyro
from typing_extensions import Annotated

from gs_toolkit.process_data.images_to_gstk_dataset import ImagesToGSToolkitDataset
from gs_toolkit.process_data.video_to_gstk_dataset import VideoToGSToolkitDataset
from gs_toolkit.utils.rich_utils import CONSOLE

Commands = Union[
    Annotated[ImagesToGSToolkitDataset, tyro.conf.subcommand(name="images")],
    Annotated[VideoToGSToolkitDataset, tyro.conf.subcommand(name="video")],
]


@dataclass
class NotInstalled:
    def main(self) -> None:
        ...


# Add aria subcommand if projectaria_tools is installed.
try:
    import projectaria_tools
except ImportError:
    projectaria_tools = None

if projectaria_tools is not None:
    from gs_toolkit.scripts.datasets.process_project_aria import ProcessProjectAria

    # Note that Union[A, Union[B, C]] == Union[A, B, C].
    Commands = Union[
        Commands, Annotated[ProcessProjectAria, tyro.conf.subcommand(name="aria")]
    ]
else:
    Commands = Union[
        Commands,
        Annotated[
            NotInstalled,
            tyro.conf.subcommand(
                name="aria",
                description="**Not installed.** Processing Project Aria data requires `pip install projectaria_tools'[all]'`.",
            ),
        ],
    ]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # type: ignore
