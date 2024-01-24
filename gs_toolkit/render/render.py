import json
import cv2
import torch
import yaml
import numpy as np
from gs_toolkit.cameras.cameras import Cameras, CameraType
from gs_toolkit.utils.eval_utils import eval_setup
from pathlib import Path


class Renderer:
    """
    Renderer class for rendering images from a given pose
    """

    fx: float
    """camera focal length in x direction"""
    fy: float
    """camera focal length in y direction"""
    cx: float
    """camera center in x direction""" ""
    cy: float
    """camera center in y direction""" ""
    load_ckpt: Path
    """path to checkpoint to load"""
    outputs: dict[str, torch.Tensor] | None = None
    """outputs from pipeline, contains rgb and depth images"""
    camera_type = CameraType.PERSPECTIVE
    """camera type, currently only perspective is supported"""

    def __init__(self, load_ckpt, camera_type=CameraType.PERSPECTIVE):
        self.load_ckpt: Path = load_ckpt
        if not self.load_ckpt.exists():
            raise Exception(f"Checkpoint {self.load_ckpt} does not exist")
        self.camera_type = camera_type
        _, self.pipeline, _, _ = eval_setup(
            self.load_ckpt,
            test_mode="inference",
        )
        ckpt_config = yaml.load(self.load_ckpt.read_text(), Loader=yaml.Loader)
        data_config = ckpt_config.data / "transforms.json"
        # From json read the fx, fy, cx, cy
        with open(data_config, "r") as f:
            meta = json.load(f)
        self.fx = meta["fl_x"]
        self.fy = meta["fl_y"]
        self.cx = meta["cx"]
        self.cy = meta["cy"]

    def get_output_from_pose(self, pose):
        poses = [pose]
        poses = np.array(poses).astype(np.float32)
        camera_to_world = torch.from_numpy(poses[:, :3])

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        return self._get_output(cameras)

    def _get_output(self, cameras) -> dict[str, torch.Tensor]:
        self.outputs = self.pipeline.model.get_outputs_for_camera(cameras)
        return self.outputs

    def show(self):
        if self.outputs is None:
            raise Exception("No outputs to show")
        cv2.imshow("rgb", self.outputs["rgb"].cpu().numpy())
        cv2.imshow("depth", self.outputs["depth"].cpu().numpy())
        cv2.waitKey(0)

    @property
    def rgb(self) -> np.ndarray:
        return self.outputs["rgb"].cpu().numpy()

    @property
    def depth(self) -> np.ndarray:
        return self.outputs["depth"].cpu().numpy()
