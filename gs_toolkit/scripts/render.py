from dataclasses import dataclass
import shutil
from typing import Union
from typing_extensions import Annotated
import cv2
from PIL import Image

import numpy as np
from gs_toolkit.render.render import Renderer
from gs_toolkit.utils.eval_utils import eval_setup
import json
import tyro
from pathlib import Path
from rich.progress import track

from gs_toolkit.utils.rich_utils import (
    CONSOLE,
)


@dataclass
class RenderFromTrajectory:
    trajectory_path: Path
    config_file: Path
    num_frames_target: int = 100

    def __post_init__(self):
        self._validate()
        self.renderer = Renderer(self.config_file)
        self.output_dir = self.config_file.parent / Path("render")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            (self.output_dir / "rgb").mkdir()
            (self.output_dir / "depth").mkdir()

    def _validate(self):
        assert self.trajectory_path is not None
        assert self.trajectory_path.suffix == ".json"
        assert self.config_file is not None

    def main(self):
        # Read trajectory from file
        with open(self.trajectory_path, "r") as f:
            meta = json.load(f)
        tranjectory = meta["camera_path"]
        num = len(tranjectory)
        CONSOLE.print(f"Total number of frames: {num}")
        interval = num // self.num_frames_target
        assert interval > 0
        poses = []
        # Render
        idx = 0
        for i in track(range(0, num, interval), description="Rendering"):
            pose = tranjectory[i]["camera_to_world"]
            # Reshape pose from (16) to (4, 4)
            pose = np.reshape(np.array(pose), (4, 4))
            self.renderer.get_output_from_pose(pose)
            # Save rgb image
            rgb = self.renderer.rgb
            rgb_path = self.output_dir / "rgb" / f"frame_{idx:05d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR))
            # Save depth image, convert depth unit from m to mm
            depth = self.renderer.depth
            depth_path = self.output_dir / "depth" / f"depth_{idx:05d}.png"
            depth = Image.fromarray((1000 * depth[:, :, 0]).astype(np.uint32))
            depth.save(str(depth_path))
            # pose[:3, 3] = pose[:3, 3] * 1000
            poses.append(pose)
            idx += 1
        # Save poses in json
        poses = np.array(poses)
        poses = poses.tolist()
        poses = {"camera_path": poses}
        poses_path = self.output_dir / "poses.json"
        with open(poses_path, "w") as f:
            json.dump(poses, f)


@dataclass
class RenderFromCameraPoses:
    config_file: Path
    output_dir: Path
    width: int = 1280
    height: int = 720

    def __post_init__(self):
        self._validate()
        self.renderer = Renderer(self.config_file)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        rgb_path = self.output_dir / "rgb"
        if not rgb_path.exists():
            (self.output_dir / "rgb").mkdir()

        depth_path = self.output_dir / "depth"
        if not depth_path.exists():
            (self.output_dir / "depth").mkdir()

        self.gt_rgb_path = self.output_dir / "gt" / "rgb"
        if not self.gt_rgb_path.exists():
            (self.output_dir / "gt" / "rgb").mkdir(parents=True)

        self.gt_depth_path = self.output_dir / "gt" / "depth"
        if not self.gt_depth_path.exists():
            (self.output_dir / "gt" / "depth").mkdir(parents=True)

    def _validate(self):
        assert self.config_file is not None

    def main(self):
        # Render
        poses = []
        _, pipeline, _, _ = eval_setup(self.config_file)
        cameras = pipeline.datamanager.train_dataset.cameras
        image_filenames = pipeline.datamanager.train_dataset.image_filenames

        for i in track(range(len(cameras)), description="Rendering training set"):
            # camera = camera.squeeze(0)
            camera = cameras[i]
            # Image file name is the last part of the path
            # image_filename = str(image_filenames[i]).split("/")[-1].split(".")[0]
            pose = camera.camera_to_worlds.cpu().numpy()
            pose = np.vstack([pose, np.array([0, 0, 0, 1])])
            self.renderer.get_output_from_pose(
                pose, width=self.width, height=self.height
            )
            # Save rgb image
            rgb = self.renderer.rgb
            rgb_path = self.output_dir / "rgb" / f"frame_{i:05}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR))
            # Save depth image, convert depth unit from m to mm
            depth = self.renderer.depth
            depth_path = self.output_dir / "depth" / f"depth_{i:05}.png"
            depth = Image.fromarray((1000 * depth[:, :, 0]).astype(np.uint32))
            depth.save(str(depth_path))
            pose = camera.camera_to_worlds.cpu().numpy()
            pose = np.vstack([pose, np.array([0, 0, 0, 1])])
            poses.append(pose)

            # Copy the ground truth rgb and depth images to the output directory
            rgb_gt_path = image_filenames[i]
            depth_name = (
                str(image_filenames[i])
                .split("/")[-1]
                .replace("frame", "depth")
                .replace("jpg", "png")
            )
            depth_gt_path = image_filenames[i].parent.parent / "depth" / depth_name
            if depth_gt_path.exists():
                # Copy the file
                shutil.copy(str(depth_gt_path), str(self.gt_depth_path))
                # Change the name into depth_{i:05}.png in the output directory
                new_name = self.gt_depth_path / f"depth_{i:05}.png"
                shutil.move(str(self.gt_depth_path / depth_name), str(new_name))

            # Copy the file
            shutil.copy(str(rgb_gt_path), str(self.gt_rgb_path))
            # Change the name into frame_{i:05}.png in the output directory
            new_name = self.gt_rgb_path / f"frame_{i:05}.jpg"
            shutil.move(str(self.gt_rgb_path / rgb_gt_path.name), str(new_name))

        # Write the poses to a file
        poses = np.array(poses)
        poses = poses.tolist()
        poses = {"camera_path": poses}
        poses_path = self.output_dir / "poses.json"
        with open(poses_path, "w") as f:
            json.dump(poses, f)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderFromTrajectory, tyro.conf.subcommand(name="trajectory")],
        Annotated[RenderFromCameraPoses, tyro.conf.subcommand(name="pose")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
