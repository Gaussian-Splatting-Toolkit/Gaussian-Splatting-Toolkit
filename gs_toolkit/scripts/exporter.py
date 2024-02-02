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
import open3d as o3d
import torch
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

        map_to_tensors = {}

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            n = positions.shape[0]
            map_to_tensors["positions"] = positions
            map_to_tensors["normals"] = np.zeros_like(positions, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(
                f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}"
            )
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select, :]

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        o3d.t.io.write_point_cloud(str(filename), pcd)


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
