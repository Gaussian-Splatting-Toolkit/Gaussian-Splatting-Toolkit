from PIL import Image
import cv2
from gs_toolkit.render.render import Renderer
from pathlib import Path
import numpy as np

from gs_toolkit.utils.rich_utils import CONSOLE

# Parameters
pose = np.array(
    [
        [
            -0.2331026274491228,
            -0.5020248573103451,
            0.8328470494147426,
            0.377221441268921,
        ],
        [
            0.93255135093636,
            -0.3582112311713894,
            0.04508538265831796,
            0.1409502506256097,
        ],
        [
            0.27570118415248984,
            0.7871821622122828,
            0.551663938056232,
            -0.05554025769233704,
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
    cv2.imwrite(
        "/data/gs-recon/student_lounge_object/rgb.png",
        cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR),
    )
    # Get depth image
    depth = renderer.depth
    CONSOLE.print(f"depth shape: {depth.shape}")
    depth = (depth * 1000).astype(np.uint32)
    im = Image.fromarray(depth[:, :, 0])
    im.save("/data/gs-recon/student_lounge_object/depth.png")


if __name__ == "__main__":
    main()
