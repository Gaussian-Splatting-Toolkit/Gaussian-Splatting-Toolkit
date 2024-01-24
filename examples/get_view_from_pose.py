from gs_toolkit.render.render import Renderer
from pathlib import Path
import numpy as np

from gs_toolkit.utils.rich_utils import CONSOLE

# Parameters
pose = np.array(
    [
        [
            -0.21889258379730414,
            -0.7638227123074239,
            0.6071745226225083,
            0.8005772638878474,
        ],
        [
            0.9757489619561686,
            -0.17135055591024806,
            0.13620921492582197,
            0.10887314377266714,
        ],
        [
            5.551115123125783e-17,
            0.62226509716726,
            0.7828065845708124,
            0.24449519885676949,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

load_config = Path(
    "/workspace/outputs/student_lounge_object/gaussian-splatting/2024-01-24_104615/config.yml"
)


def main():
    renderer = Renderer(load_config)
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
