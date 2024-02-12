"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch.nn import Parameter
import tyro
from typing_extensions import Annotated

from gs_toolkit.exporter.exporter_utils import collect_camera_poses
from gs_toolkit.models.gaussian_splatting import GaussianSplattingModel
from gs_toolkit.pipelines.base_pipeline import VanillaPipeline
from gs_toolkit.utils.eval_utils import eval_setup
from gs_toolkit.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [
            ("transforms_train.json", train_frames),
            ("transforms_eval.json", eval_frames),
        ]:
            if len(frames) == 0:
                CONSOLE.print(
                    f"[bold yellow]No frames found for {file_name}. Skipping."
                )
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(
                f"[bold green]:white_check_mark: Saved poses to {output_file_path}"
            )


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, GaussianSplattingModel)

        model: GaussianSplattingModel = pipeline.model

        filename = self.output_dir / "point_cloud.ply"

        with torch.no_grad():
            param_group = model.get_gaussian_param_groups()
            xyz = param_group["xyz"][0].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = param_group["features_dc"][0].detach().cpu().numpy()
            f_rest = (
                param_group["features_rest"][0]
                .detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            opacities = param_group["opacity"][0].detach().cpu().numpy()
            scale = param_group["scaling"][0].detach().cpu().numpy()
            rotation = param_group["rotation"][0].detach().cpu().numpy()

            dtype_full = [
                (attribute, "f4")
                for attribute in self.construct_list_of_attributes(param_group)
            ]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(filename)

    def construct_list_of_attributes(self, param: dict[str, list[Parameter]]):
        l = ["x", "y", "z", "nx", "ny", "nz"]  # noqa: E741
        # All channels except the 3 DC
        for i in range(param["features_dc"][0].shape[1]):
            l.append("f_dc_{}".format(i))
        for i in range(
            param["features_rest"][0].shape[1] * param["features_rest"][0].shape[2]
        ):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(param["scaling"][0].shape[1]):
            l.append("scale_{}".format(i))
        for i in range(param["rotation"][0].shape[1]):
            l.append("rot_{}".format(i))
        return l


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="camera-poses")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
