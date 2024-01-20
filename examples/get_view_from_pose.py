from gs_toolkit.render.render import Renderer
from pathlib import Path
import numpy as np

from gs_toolkit.utils.rich_utils import CONSOLE

# Parameters
fx = 386.31666583967507
fy = 394.51799422016785
cx = 234.9184435902503
cy = 424.12873736061147

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

load_config = Path(
    "/workspace/outputs/robot_studio/gaussian-splatting/2024-01-20_123509/config.yml"
)


def main():
    renderer = Renderer(fx, fy, cx, cy, load_config)
    renderer.get_output_from_pose(pose)
    renderer.show()
    # Get rgb image
    rgb = renderer.rgb
    CONSOLE.print(f"rgb shape: {rgb.shape}")
    # Get depth image
    depth = renderer.depth
    CONSOLE.print(f"depth shape: {depth.shape}")


if __name__ == "__main__":
    main()
