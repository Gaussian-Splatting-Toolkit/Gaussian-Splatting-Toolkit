from gs_toolkit.render.render import Renderer
from pathlib import Path
import numpy as np

from gs_toolkit.utils.rich_utils import CONSOLE

# Parameters
pose = np.array(
    [
        [
            0.20990344882011414,
            0.03703904151916504,
            0.977020263671875,
            0.014316114597022533,
        ],
        [
            0.9774146676063538,
            0.01710675284266472,
            -0.2106366902589798,
            0.08852361142635345,
        ],
        [
            -0.02451542392373085,
            0.9991674423217773,
            -0.03261175751686096,
            -0.07301409542560577,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

load_config = Path(
    "/mnt/d/Projects/Gaussian-Splatting-Toolkit/outputs/microwave_fine/gaussian-splatting/2024-02-14_173020/config.yml"
)


def main():
    renderer = Renderer(load_config)
    renderer.get_output_from_pose(pose, width=640, height=480)
    renderer.show()
    # Get rgb image
    rgb = renderer.rgb
    CONSOLE.print(f"rgb shape: {rgb.shape}")
    # cv2.imwrite(
    #     "exports/student_lounge_object/rgb.png",
    #     cv2.cvtColor(255 * rgb, cv2.COLOR_RGB2BGR),
    # )
    # Get depth image
    depth = renderer.depth
    CONSOLE.print(f"depth shape: {depth.shape}")
    depth = (depth * 1000).astype(np.uint32)
    # im = Image.fromarray(depth[:, :, 0])
    # im.save("exports/student_lounge_object/depth.png")


if __name__ == "__main__":
    main()
