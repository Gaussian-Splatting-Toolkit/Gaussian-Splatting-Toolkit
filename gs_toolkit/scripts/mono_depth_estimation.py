import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image

from gs_toolkit.process_data.depth_estimation.dpt import DepthAnything
from gs_toolkit.process_data.depth_estimation.utils.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--outdir", type=str, default="./vis_depth")
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"]
    )

    args = parser.parse_args()

    margin_width = 50
    caption_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    depth_anything = (
        DepthAnything.from_pretrained(
            "LiheYoung/depth_anything_{}14".format(args.encoder)
        )
        .to(DEVICE)
        .eval()
    )

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [
            os.path.join(args.img_path, filename)
            for filename in filenames
            if not filename.startswith(".")
        ]
        filenames.sort()

    os.makedirs(args.outdir, exist_ok=True)

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (1 - depth) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        depth_img = Image.fromarray(depth)

        filename = os.path.basename(filename)

        depth_img.save(
            os.path.join(args.outdir, filename[: filename.rfind(".")] + "_depth.png")
        )
