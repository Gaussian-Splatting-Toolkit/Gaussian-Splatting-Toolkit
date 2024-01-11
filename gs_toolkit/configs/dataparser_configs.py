"""
Aggregate all the dataparser configs in one location.
"""

from typing import TYPE_CHECKING

import tyro

from gs_toolkit.data.dataparsers.base_dataparser import DataParserConfig
from gs_toolkit.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from gs_toolkit.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from gs_toolkit.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig

dataparsers = {
    "blender-data": BlenderDataParserConfig(),
    "scannet-data": ScanNetDataParserConfig(),
    "colmap": ColmapDataParserConfig(),
}

if TYPE_CHECKING:
    # For static analysis (tab completion, type checking, etc), just use the base
    # dataparser config.
    DataParserUnion = DataParserConfig
else:
    # At runtime, populate a Union type dynamically. This is used by `tyro` to generate
    # subcommands in the CLI.
    DataParserUnion = tyro.extras.subcommand_type_from_defaults(
        dataparsers,
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[
    DataParserUnion
]  # Omit prefixes of flags in subcommands.
"""Union over possible dataparser types, annotated with metadata for tyro. This is
the same as the vanilla union, but results in shorter subcommand names."""
