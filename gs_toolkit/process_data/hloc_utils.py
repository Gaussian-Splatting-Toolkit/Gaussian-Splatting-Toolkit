from pathlib import Path
from typing import Literal

from gs_toolkit.process_data.process_data_utils import CameraModel

import pycolmap
from hloc import (  # type: ignore
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)


def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    feature_type: Literal[
        "sift",
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sosnet",
        "disk",
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
        "disk+lightglue",
        "superpoint+lightglue",
    ] = "superglue",
    num_matched: int = 50,
) -> None:
    """Runs hloc on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Method to use for matching images.
        feature_type: Type of visual features to use.
        matcher_type: Type of feature matcher to use.
        num_matched: Number of image pairs for loc.
    """
    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sparse" / "0"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    references = [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)  # type: ignore

    retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
    if num_matched >= len(references):
        num_matched = len(references)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # type: ignore
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore
    reconstruction.main(  # type: ignore
        sfm_dir,
        image_dir,
        sfm_pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.SINGLE,  # type: ignore
        image_options=image_options,
        verbose=verbose,
    )

    model = pycolmap.Reconstruction(str(sfm_dir))
    model.export_PLY(colmap_dir / "point_cloud.ply")
