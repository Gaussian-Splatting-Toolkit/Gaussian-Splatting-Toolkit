"""
Tools supporting the execution of COLMAP and preparation of COLMAP-based datasets for gs_toolkit training.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from rich.progress import track

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from gs_toolkit.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)
from gs_toolkit.process_data.process_data_utils import CameraModel
from gs_toolkit.utils import colormaps
from gs_toolkit.utils.rich_utils import CONSOLE


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "SIMPLE_RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out


def colmap_to_json(
    scale_factor: float,
    recon_dir: Path,
    output_dir: Path,
    image_id_to_mask_path: Optional[Dict[int, Path]] = None,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    image_rename_map: Optional[Dict[str, str]] = None,
    scales: Optional[Dict[int, float]] = None,
    shifts: Optional[Dict[int, float]] = None,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        scales: Scales for each depth image
        shifts: Shifts for each depth image

    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = im_data.name
        if image_rename_map is not None:
            name = image_rename_map[name]
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if scales is not None:
            frame["scale"] = scales[im_id]
        if shifts is not None:
            frame["shift"] = shifts[im_id]
        if image_id_to_mask_path is not None:
            mask_path = image_id_to_mask_path[im_id]
            frame["mask_path"] = str(mask_path.relative_to(mask_path.parent.parent))
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_path"] = str(depth_path.relative_to(depth_path.parent.parent))
        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    out = parse_colmap_camera_params(cam_id_to_camera[1])
    out["applied_scale"] = scale_factor
    if os.path.exists(output_dir / "colmap" / "point_cloud.ply"):
        out["ply_file_path"] = "colmap/point_cloud.ply"
    out["frames"] = frames

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


def create_sfm_depth(
    recon_dir: Path,
    output_dir: Path,
    verbose: bool = True,
    depth_scale_to_integer_factor: float = 1000.0,
    min_depth: float = 0.001,
    max_depth: float = 10000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 2,
    include_depth_debug: bool = False,
    input_images_dir: Optional[Path] = None,
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth map images encoded as
    16-bit "millimeter depth" PNGs.

    Notes:
     * This facility does NOT use COLMAP dense reconstruction; it creates depth
        maps from sparse SfM points here.
     * COLMAP does *not* reconstruct metric depth unless you give it calibrated
        (metric) intrinsics as input. Therefore, "depth" in this function has
        potentially ambiguous units.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        verbose: If True, logs progress of depth image creation.
        depth_scale_to_integer_factor: Use this parameter to tune the conversion of
          raw depth measurements to integer depth values.  This value should
          be equal to 1. / `depth_unit_scale_factor`, where
          `depth_unit_scale_factor` is the value you provide at training time.
          E.g. for millimeter depth, leave `depth_unit_scale_factor` at 1e-3
          and depth_scale_to_integer_factor at 1000.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
        include_depth_debug: Also include debug images showing depth overlaid
          upon RGB.
    Returns:
        Depth file paths indexed by COLMAP image id
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # ptid_to_info = recon.points3D
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    if verbose:
        iter_images = track(
            im_id_to_image.items(),
            total=len(im_id_to_image.items()),
            description="Creating depth maps ...",
        )
    else:
        iter_images = iter(im_id_to_image.items())

    image_id_to_depth_path = {}
    for im_id, im_data in iter_images:
        # TODO(1480) BEGIN delete when abandoning colmap_parsing_utils
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        rotation = qvec2rotmat(im_data.qvec)
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )
        # TODO(1480) END delete when abandoning colmap_parsing_utils

        # TODO(1480) BEGIN use pycolmap API

        # # Get only keypoints that have corresponding triangulated 3D points
        # p2ds = im_data.get_valid_points2D()

        # xyz_world = np.array([ptid_to_info[p2d.point3D_id].xyz for p2d in p2ds])

        # # COLMAP OpenCV convention: z is always positive
        # z = (im_data.rotation_matrix() @ xyz_world.T)[-1] + im_data.tvec[-1]

        # # Mean reprojection error in image space
        # errors = np.array([ptid_to_info[p2d.point3D_id].error for p2d in p2ds])

        # # Number of frames in which each frame is visible
        # n_visible = np.array([ptid_to_info[p2d.point3D_id].track.length() for p2d in p2ds])

        # Note: these are *unrectified* pixel coordinates that should match the original input
        # no matter the camera model
        # uv = np.array([p2d.xy for p2d in p2ds])

        # TODO(1480) END use pycolmap API

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z

        # E.g. if `depth` is metric and in units of meters, and `depth_scale_to_integer_factor`
        # is 1000, then `depth_img` will be integer millimeters.
        depth_img = (depth_scale_to_integer_factor * depth).astype(np.uint16)

        out_name = str(im_data.name)
        depth_path = output_dir / out_name
        if depth_path.suffix == ".jpg":
            depth_path = depth_path.with_suffix(".png")
        cv2.imwrite(str(depth_path), depth_img)  # type: ignore

        image_id_to_depth_path[im_id] = depth_path

        if include_depth_debug:
            assert (
                input_images_dir is not None
            ), "Need explicit input_images_dir for debug images"
            assert input_images_dir.exists(), input_images_dir

            depth_flat = depth.flatten()[:, None]
            overlay = (
                255.0
                * colormaps.apply_depth_colormap(torch.from_numpy(depth_flat)).numpy()
            )
            overlay = overlay.reshape([H, W, 3])
            input_image_path = input_images_dir / im_data.name
            input_image = cv2.imread(str(input_image_path))  # type: ignore
            debug = 0.3 * input_image + 0.7 + overlay

            out_name = out_name + ".debug.jpg"
            output_path = output_dir / "debug_depth" / out_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug.astype(np.uint8))  # type: ignore

    return image_id_to_depth_path


def align_depth(
    recon_dir: Path,
    depth_dir: Path,
    verbose: bool = True,
    min_depth: float = 0.001,
    max_depth: float = 10000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 2,
) -> Union[Dict[int, Path], float]:
    """Aligns COLMAP's sparse depth maps with the ground truth depth maps.

    Notes:
        * This function assumes that the COLMAP depth maps are in the same order as the ground truth depth maps.
        * This function assumes that the ground truth depth maps are in the same order as the COLMAP images.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        depth_dir: Path to the directory containing the ground truth depth maps.
        verbose: If True, logs progress of depth image alignment.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
    Returns:
        Depth file paths indexed by COLMAP image id, and the mean scale factor.
    """
    if not depth_dir.exists():
        raise RuntimeError(f"You are required to provide depth maps in {depth_dir}")
    ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    if verbose:
        iter_images = track(
            im_id_to_image.items(),
            total=len(im_id_to_image.items()),
            description="Calculating depth maps ...",
        )
    else:
        iter_images = iter(im_id_to_image.items())

    total_scale = []
    total_variances = []
    image_id_to_depth_path = {}
    # cov = []
    for im_id, im_data in iter_images:
        # Replace jpg with png in im_data.name as depth name
        depth_name = im_data.name.replace(".jpg", ".png")
        # Replace prefix from "frame_" to "depth_"
        depth_name = depth_name.replace("frame_", "depth_")
        depth_path = depth_dir / depth_name
        # Read depth image
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)  # type: ignore
        image_id_to_depth_path[im_id] = depth_path
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        rotation = qvec2rotmat(im_data.qvec)
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )
        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth_measure = depth_img[vv, uu]
        # Choose the idx that depth measure is not 0
        idx = np.where((depth_measure > 30) & (depth_measure < 1_000))
        z = z[idx]
        depth_measure = depth_measure[idx] / 1000
        if len(z) != 0:
            total_scale.append(np.mean(depth_measure / z))
            total_variances.append(np.var(depth_measure / z))
        # cov.append(np.cov(z, depth_measure)[0, 1])
    if np.mean(total_variances) / np.mean(total_scale) > 0.1:
        CONSOLE.log(
            f"[bold yellow]Warning: The variance ({np.mean(total_variances)}) over mean ({np.mean(total_scale)}) of scale is high, scaling may not be accurate."
        )
    return image_id_to_depth_path, np.mean(total_scale)


def align_mono_depth(
    recon_dir: Path,
    depth_dir: Path,
    verbose: bool = True,
    min_depth: float = 0.001,
    max_depth: float = 10000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 2,
) -> Tuple[Dict[int, Path], Dict[int, float], Dict[int, float]]:
    if not depth_dir.exists():
        raise RuntimeError(f"You are required to provide depth maps in {depth_dir}")
    ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    if verbose:
        iter_images = track(
            im_id_to_image.items(),
            total=len(im_id_to_image.items()),
            description="Calculating depth maps ...",
        )
    else:
        iter_images = iter(im_id_to_image.items())

    scales = {}
    shifts = {}
    image_id_to_depth_path = {}
    # cov = []
    for im_id, im_data in iter_images:
        # Replace jpg with png in im_data.name as depth name
        depth_name = im_data.name.replace(".jpg", ".png")
        # Replace prefix from "frame_" to "depth_"
        depth_name = depth_name.replace("frame_", "depth_")
        depth_path = depth_dir / depth_name
        # Read depth image
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)  # type: ignore
        image_id_to_depth_path[im_id] = depth_path
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        rotation = qvec2rotmat(im_data.qvec)
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )
        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth_est = depth_img[vv, uu] / 255.0
        if len(z) != 0:
            # fit a linear model
            A = np.vstack([z, np.ones(len(z))]).T
            m, c = np.linalg.lstsq(A, depth_est, rcond=None)[0]
            scales[im_id] = m
            shifts[im_id] = c

    return image_id_to_depth_path, scales, shifts


def get_mask_files(
    recon_dir: Path,
    mask_dir: Path,
    verbose: bool = True,
) -> Dict[int, Path]:
    """
    Get the mask files for each image in the reconstruction.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        mask_dir: Path to the directory containing the mask images.
        verbose: If True, logs progress of mask image creation.
    Returns:
        Mask file paths indexed by COLMAP image id
    """
    if not mask_dir.exists():
        raise RuntimeError(f"You are required to provide masks in {mask_dir}")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    if verbose:
        iter_images = track(
            im_id_to_image.items(),
            total=len(im_id_to_image.items()),
            description="Getting the mask ...",
        )
    else:
        iter_images = iter(im_id_to_image.items())

    image_id_to_depth_path = {}
    # cov = []
    for im_id, im_data in iter_images:
        # Replace jpg with png in im_data.name as depth name
        mask_name = im_data.name.replace(".jpg", ".png")
        mask_path = mask_dir / mask_name
        image_id_to_depth_path[im_id] = mask_path

    return image_id_to_depth_path


def get_matching_summary(num_initial_frames: int, num_matched_frames: int) -> str:
    """Returns a summary of the matching results.

    Args:
        num_initial_frames: The number of initial frames.
        num_matched_frames: The number of matched frames.

    Returns:
        A summary of the matching results.
    """
    match_ratio = num_matched_frames / num_initial_frames
    if match_ratio == 1:
        return "[bold green]COLMAP found poses for all images, CONGRATS!"
    if match_ratio < 0.4:
        result = f"[bold red]COLMAP only found poses for {num_matched_frames / num_initial_frames * 100:.2f}%"
        result += (
            " of the images. This is low.\nThis can be caused by a variety of reasons,"
        )
        result += " such poor scene coverage, blurry images, or large exposure changes."
        return result
    if match_ratio < 0.8:
        result = f"[bold yellow]COLMAP only found poses for {num_matched_frames / num_initial_frames * 100:.2f}%"
        result += " of the images.\nThis isn't great, but may be ok."
        result += "\nMissing poses can be caused by a variety of reasons, such poor scene coverage, blurry images,"
        result += " or large exposure changes."
        return result
    return f"[bold green]COLMAP found poses for {num_matched_frames / num_initial_frames * 100:.2f}% of the images."
