from pathlib import Path
import sys
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np

from sklearn.pipeline import Pipeline
from gs_toolkit.pipelines.base_pipeline import VanillaPipeline
from gs_toolkit.data.datasets.base_dataset import InputDataset
import torch
import open3d as o3d

from gs_toolkit.utils.rich_utils import (
    CONSOLE,
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


def collect_camera_poses_for_dataset(
    dataset: Optional[InputDataset],
) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        transform = cameras.camera_to_worlds[idx].tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames


def collect_camera_poses(
    pipeline: VanillaPipeline,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collects camera poses for train and eval datasets.

    Args:
        pipeline: Pipeline to evaluate with.

    Returns:
        List of train camera poses, list of eval camera poses.
    """

    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    train_frames = collect_camera_poses_for_dataset(train_dataset)
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames


def generate_point_cloud(
    pipeline: Pipeline,
    output_dir: Path,
    voxel_length: float = 2.0 / 512.0,
    sdf_trunc: float = 0.04,
    color_type: o3d.pipelines.integration.TSDFVolumeColorType = o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a gaussian splatting.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        voxel_length: Voxel length for the volume.
        sdf_trunc: SDF truncation for the volume.
        color_type: Color type for the volume.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.

    Returns:
        Point cloud.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_type,
    )
    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    idx = 0
    with progress as progress_bar:
        task = progress_bar.add_task(
            "Generating Point Cloud", total=len(pipeline.datamanager.train_dataset)
        )
        while not progress_bar.finished:
            with torch.no_grad():
                cameras, _ = pipeline.datamanager.next_train(0)
                outputs = pipeline.model.get_outputs_for_camera(cameras)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"Could not find {rgb_output_name} in the model outputs",
                    justify="center",
                )
                CONSOLE.print(
                    f"Please set --rgb_output_name to one of: {outputs.keys()}",
                    justify="center",
                )
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"Could not find {depth_output_name} in the model outputs",
                    justify="center",
                )
                CONSOLE.print(
                    f"Please set --depth_output_name to one of: {outputs.keys()}",
                    justify="center",
                )
                sys.exit(1)
            # Save rgb image
            rgb = outputs[rgb_output_name].cpu().numpy()
            rgb_path = output_dir / "rgb" / f"frame_{idx+1:05d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR))
            # Save depth image, convert depth unit from m to mm
            depth = outputs[depth_output_name].cpu().numpy()
            depth_path = output_dir / "depth" / f"depth_{idx+1:05d}.png"
            depth = Image.fromarray((1000 * depth[:, :, 0]).astype(np.uint32))
            depth.save(str(depth_path))

            rgb = o3d.io.read_image(str(rgb_path))
            depth = o3d.io.read_image(str(depth_path))
            pose = cameras.camera_to_worlds[0].cpu().numpy()
            pose = np.vstack([pose, np.array([0, 0, 0, 1])]).T
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=cameras.width,
                height=cameras.height,
                fx=cameras.fx,
                fy=cameras.fy,
                cx=cameras.cx,
                cy=cameras.cy,
            )
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            progress_bar.update(task, advance=1)
            idx += 1

    return volume.extract_point_cloud()
