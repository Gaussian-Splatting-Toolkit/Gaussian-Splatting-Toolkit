from dataclasses import dataclass
import cv2

import numpy as np
from gs_toolkit.render.render import Renderer
import json
from pathlib import Path


@dataclass
class RenderFromTrajectory:
    trajectory_path: Path
    config_file: Path
    output_dir: Path
    num_frames_target: int = 200

    def __post_init__(self):
        self._validate()
        self.renderer = Renderer(self.config_file)

    def _validate(self):
        assert self.trajectory_path is not None
        assert self.trajectory_path.suffix == ".json"
        assert self.config_file is not None
        assert self.output_dir is not None

    def render(self):
        # Read trajectory from file
        with open(self.trajectory_path, "r") as f:
            meta = json.load(f)
        tranjectory = meta["keyframes"]
        num = len(tranjectory)
        interval = num // self.num_frames_target
        assert interval > 0
        # Render
        for i in range(0, num, interval):
            pose = tranjectory[i]["matrix"]
            # Reshape pose from (16) to (4, 4)
            pose = np.array(pose).reshape((4, 4))
            outputs = self.renderer.get_output_from_pose(pose)
            # Save rgb image
            rgb = outputs["rgb"]
            rgb_path = self.output_dir / "rgb" / f"frame_{i+1:05d}.png"
            cv2.imwrite(rgb_path, rgb)
            # Save depth image
            depth = outputs["depth"]
            depth_path = self.output_dir / "depth" / f"depth_{i+1:05d}.png"
            cv2.imwrite(depth_path, depth)
