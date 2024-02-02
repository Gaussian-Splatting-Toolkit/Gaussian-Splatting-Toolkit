from dataclasses import dataclass
import cv2
from PIL import Image

import numpy as np
from gs_toolkit.render.render import Renderer
from gs_toolkit.utils.rich_utils import CONSOLE
import json
import tyro
from pathlib import Path
from rich.progress import track


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


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RenderFromTrajectory]).main()


if __name__ == "__main__":
    entrypoint()
