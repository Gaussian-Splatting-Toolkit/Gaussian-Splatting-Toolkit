import numpy as np
import open3d as o3d


def extract_pcd_from_rgbd(
    rgb_file: str,
    depth_file: str,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    mask_file: str | None = None,
) -> o3d.geometry.PointCloud:
    color = o3d.io.read_image(rgb_file)
    depth = o3d.io.read_image(depth_file)
    if mask_file is not None:
        mask = o3d.io.read_image(mask_file)
        mask_np = np.asarray(mask)
        mask_np[mask_np > 0] = 1
        depth_np = np.asarray(depth)
        depth_np = depth_np * mask_np
        depth = o3d.geometry.Image(depth_np)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=3.0, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic,
    )
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd


if __name__ == "__main__":
    rgb_file = "/mnt/d/Projects/reconstruction/data/small_obj/rope_N/rgb/00000.jpg"
    depth_file = "/mnt/d/Projects/reconstruction/data/small_obj/rope_N/depth/00000.png"
    mask_file = "/mnt/d/Projects/reconstruction/data/small_obj/rope_N/mask/00000.jpg"
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1280,
        height=720,
        fx=617.141220756315,
        fy=619.3666113012686,
        cx=641.9824480711256,
        cy=360.94842617544964,
    )
    pcd = extract_pcd_from_rgbd(rgb_file, depth_file, intrinsic, mask_file)
