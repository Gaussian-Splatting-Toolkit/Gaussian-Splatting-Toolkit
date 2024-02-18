"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import open3d as o3d

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch.nn import Parameter
import tyro
from typing_extensions import Annotated

from gs_toolkit.exporter.exporter_utils import (
    collect_camera_poses,
    generate_point_cloud,
)
from gs_toolkit.models.gaussian_splatting import GaussianSplattingModel
from gs_toolkit.pipelines.base_pipeline import VanillaPipeline
from gs_toolkit.data.datamanagers.full_images_datamanager import FullImageDatamanager
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

        filename = self.output_dir / "gaussians.ply"

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


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "open3d"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        rgb_dir = self.output_dir / "rgb"
        if not rgb_dir.exists():
            rgb_dir.mkdir(parents=True)

        depth_dir = self.output_dir / "depth"
        if not depth_dir.exists():
            depth_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(pipeline.datamanager, (FullImageDatamanager))

        # Whether the normals should be estimated based on the point cloud.
        pcd = generate_point_cloud(
            pipeline=pipeline,
            output_dir=self.output_dir,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(
                points.shape[0], axis=0
            )[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="camera-poses")],
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="point-cloud")],
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
