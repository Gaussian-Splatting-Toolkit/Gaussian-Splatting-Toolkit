from dataclasses import dataclass
from typing import Union
from typing_extensions import Annotated
import cv2
from PIL import Image

import numpy as np
import torch
from gs_toolkit.render.render import Renderer
from gs_toolkit.utils.eval_utils import eval_setup
from gs_toolkit.utils.rich_utils import CONSOLE
import json
import tyro
from pathlib import Path
from rich.progress import track

from gs_toolkit.utils.rich_utils import (
    CONSOLE,
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
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
            rgb_path = self.output_dir / "rgb" / f"frame_{idx+1:05d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR))
            # Save depth image, convert depth unit from m to mm
            depth = self.renderer.depth
            depth_path = self.output_dir / "depth" / f"depth_{idx+1:05d}.png"
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

    def _validate(self):
        assert self.config_file is not None

    def main(self):
        # Render
        idx = 0
        poses = []
        _, pipeline, _, _ = eval_setup(self.config_file)
        progress = Progress(
            TextColumn("Rendering rgb and depth images"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
            console=CONSOLE,
        )
        with progress as progress_bar:
            task = progress_bar.add_task(
                "Generating Point Cloud", total=len(pipeline.datamanager.train_dataset)
            )
            while not progress_bar.finished:
                with torch.no_grad():
                    cameras, _ = pipeline.datamanager.next_train(0)
                    outputs = pipeline.model.get_outputs_for_camera(cameras)
                # Save rgb image
                rgb = outputs["rgb"].cpu().numpy()
                rgb_path = self.output_dir / "rgb" / f"frame_{idx+1:05d}.png"
                cv2.imwrite(str(rgb_path), cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR))
                # Save depth image, convert depth unit from m to mm
                depth = outputs["depth"].cpu().numpy()
                depth_path = self.output_dir / "depth" / f"depth_{idx+1:05d}.png"
                depth = Image.fromarray((1000 * depth[:, :, 0]).astype(np.uint32))
                depth.save(str(depth_path))
                pose = cameras.camera_to_worlds[0].cpu().numpy()
                pose = np.vstack([pose, np.array([0, 0, 0, 1])]).T
                poses.append(pose)

                progress_bar.update(task, advance=1)
                idx += 1

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
