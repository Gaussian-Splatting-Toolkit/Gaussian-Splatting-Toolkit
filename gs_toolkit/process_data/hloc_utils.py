import sys
from pathlib import Path
from typing import Literal

from gs_toolkit.process_data.process_data_utils import CameraModel
from gs_toolkit.utils.rich_utils import CONSOLE

try:
    # TODO(1480) un-hide pycolmap import
    import pycolmap
    from hloc import (  # type: ignore
        extract_features,
        match_features,
        pairs_from_exhaustive,
        pairs_from_retrieval,
        reconstruction,
    )
except ImportError:
    _HAS_HLOC = False

else:
    _HAS_HLOC = True

try:
    from pixsfm.refine_hloc import PixSfM  # type: ignore
except ImportError:
    _HAS_PIXSFM = False
else:
    _HAS_PIXSFM = True


def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
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
    refine_pixsfm: bool = False,
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
        refine_pixsfm: If True, refine the reconstruction using pixel-perfect-sfm.
    """
    if not _HAS_HLOC:
        CONSOLE.print(
            f"[bold red]Error: To use this set of parameters ({feature_type}/{matcher_type}/hloc), "
            "you must install hloc toolbox!!"
        )
        sys.exit(1)

    if refine_pixsfm and not _HAS_PIXSFM:
        CONSOLE.print(
            "[bold red]Error: use refine_pixsfm, you must install pixel-perfect-sfm toolbox!!"
        )
        sys.exit(1)

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
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)  # type: ignore
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
        if num_matched >= len(references):
            num_matched = len(references)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # type: ignore
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore
    if refine_pixsfm:
        sfm = PixSfM(  # type: ignore
            conf={
                "dense_features": {"use_cache": True},
                "KA": {
                    "dense_features": {"use_cache": True},
                    "max_kps_per_problem": 1000,
                },
                "BA": {"strategy": "costmaps"},
            }
        )
        refined, _ = sfm.reconstruction(
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            image_list=references,
            camera_mode=pycolmap.CameraMode.SINGLE,  # type: ignore
            image_options=image_options,
            verbose=verbose,
        )
        print("Refined", refined.summary())

    else:
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
