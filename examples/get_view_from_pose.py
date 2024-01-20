from gs_toolkit.cameras.cameras import Cameras, CameraType
from gs_toolkit.utils.eval_utils import eval_setup
import numpy as np
from pathlib import Path
import torch
import cv2

# Parameters
fx = 386.31666583967507
fy = 394.51799422016785
cx = 234.9184435902503
cy = 424.12873736061147

poses = []
pose = np.array(
    [
        [
            -0.9070786306434533,
            0.29808580235020926,
            -0.29724268244522,
            -0.04332952589694477,
        ],
        [
            -0.42096123079209796,
            -0.6423091763137213,
            0.6404924388259328,
            0.09336557417193989,
        ],
        [
            1.942890293094024e-16,
            0.7061046497937973,
            0.7081074943393686,
            0.10322192546978119,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

poses.append(pose)
poses = np.array(poses).astype(np.float32)
camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

cameras = Cameras(
    camera_to_worlds=camera_to_world,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    camera_type=CameraType.PERSPECTIVE,
)

load_config = Path(
    "/workspace/outputs/robot_studio/gaussian-splatting/2024-01-20_123509/config.yml"
)
_, pipeline, _, _ = eval_setup(
    load_config,
    test_mode="inference",
)

outputs = pipeline.model.get_outputs_for_camera(cameras)

cv2.imshow("rgb", outputs["rgb"].cpu().numpy())
cv2.imshow("depth", outputs["depth"].cpu().numpy())

cv2.waitKey(0)
