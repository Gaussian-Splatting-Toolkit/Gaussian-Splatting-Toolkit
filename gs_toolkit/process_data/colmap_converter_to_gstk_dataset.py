"""Base class to processes a image sequence to a gs_toolkit compatible dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from gs_toolkit.process_data import colmap_utils, hloc_utils, process_data_utils
from gs_toolkit.process_data.base_converter_to_gstk_dataset import (
    BaseConverterToGSToolkitDataset,
)
from gs_toolkit.process_data.process_data_utils import CAMERA_MODELS
from gs_toolkit.utils import install_checks
from gs_toolkit.utils.rich_utils import CONSOLE


@dataclass
class ColmapConverterToGSToolkitDataset(BaseConverterToGSToolkitDataset):
    """Base class to process images into a gs_toolkit dataset using colmap"""

    camera_type: Literal["perspective", "fisheye", "equirectangular"] = "perspective"
    """Camera model to use."""
    refine_intrinsics: bool = True
    """If True, do bundle adjustment to refine intrinsics.
    Only works with colmap sfm_tool"""
    feature_type: Literal[
        "sift",
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sosnet",
        "disk",
    ] = "superpoint_aachen"
    """Type of feature to use."""
    matcher_type: Literal[
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
        "disk+lightglue",
        "superpoint+lightglue",
    ] = "superpoint+lightglue"
    """Matching algorithm."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    colmap_model_path: Path = Path("colmap/sparse/0")
    """Optionally sets the path of the colmap model. Used only when --skip-colmap is set to True. The path is relative
       to the output directory.
    """
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
    images_per_equirect: Literal[8, 14] = 8
    """Number of samples per image to take from each equirectangular image.
       Used only when camera-type is equirectangular.
    """
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    crop_bottom: float = 0.0
    """Portion of the image to crop from the bottom.
       Can be used instead of `crop-factor 0.0 [num] 0.0 0.0` Should be in [0,1].
    """
    gpu: bool = True
    """If True, use GPU."""
    use_sfm_depth: bool = False
    """If True, export and use depth maps induced from SfM points."""
    include_depth_debug: bool = False
    """If --use-sfm-depth and this flag is True, also export debug images showing Sf overlaid upon input images."""
    same_dimensions: bool = True
    """Whether to assume all images are same dimensions and so to use fast downscaling with no autorotation."""
    depth_data: Optional[Path] = None
    """Path to depth data. If set, will use this depth data instead of running COLMAP to generate depth data."""
    mask_data: Optional[Path] = None
    """Path to mask data. If set, will use this mask data instead of running COLMAP to generate mask data."""
    using_est_depth: bool = False
    """If True, using estimated depth data."""

    @staticmethod
    def default_colmap_path() -> Path:
        return Path("colmap/sparse/0")

    @property
    def absolute_colmap_model_path(self) -> Path:
        return self.output_dir / self.colmap_model_path

    @property
    def absolute_colmap_path(self) -> Path:
        return self.output_dir / "colmap"

    def _save_transforms(
        self,
        num_frames: int,
        scale_factor: float = 1.0,
        image_id_to_depth_path: Optional[Dict[int, Path]] = None,
        image_id_to_mask_path: Optional[Dict[int, Path]] = None,
        image_rename_map: Optional[Dict[str, str]] = None,
        scales: Optional[Dict[int, float]] = None,
        shifts: Optional[Dict[int, float]] = None,
    ) -> List[str]:
        """Save colmap transforms into the output folder

        Args:
            image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
            image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        """
        summary_log = []
        if (self.absolute_colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status(
                "[bold yellow]Saving results to transforms.json", spinner="balloon"
            ):
                num_matched_frames = colmap_utils.colmap_to_json(
                    scale_factor=scale_factor,
                    recon_dir=self.absolute_colmap_model_path,
                    output_dir=self.output_dir,
                    image_id_to_depth_path=image_id_to_depth_path,
                    image_id_to_mask_path=image_id_to_mask_path,
                    image_rename_map=image_rename_map,
                    scales=scales,
                    shifts=shifts,
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(
                colmap_utils.get_matching_summary(num_frames, num_matched_frames)
            )

        else:
            CONSOLE.log(
                "[bold yellow]Warning: Could not find existing COLMAP results. "
                "Not generating transforms.json"
            )
        return summary_log

    def _export_depth(self) -> Tuple[Optional[Dict[int, Path]], List[str]]:
        """If SFM is used for creating depth image, this method will create the depth images from image in
        `self.image_dir`.

        Returns:
            Depth file paths indexed by COLMAP image id, logs
        """
        summary_log = []
        if self.use_sfm_depth:
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            image_id_to_depth_path = colmap_utils.create_sfm_depth(
                recon_dir=self.output_dir / self.default_colmap_path(),
                output_dir=depth_dir,
                include_depth_debug=self.include_depth_debug,
                input_images_dir=self.image_dir,
                verbose=self.verbose,
            )
            summary_log.append(
                process_data_utils.downscale_images(
                    depth_dir,
                    self.num_downscales,
                    folder_name="depths",
                    nearest_neighbor=True,
                    verbose=self.verbose,
                )
            )
            return image_id_to_depth_path, summary_log
        return None, summary_log

    def _align_depth(self) -> Tuple[Optional[Dict[int, Path]], List[str]]:
        scale_factor = 1.0
        if self.depth_data is not None and not self.using_est_depth:
            image_id_to_depth_path, scale_factor = colmap_utils.align_depth(
                recon_dir=self.output_dir / self.default_colmap_path(),
                depth_dir=self.depth_image_dir,
                verbose=self.verbose,
            )
            return image_id_to_depth_path, scale_factor
        CONSOLE.print(f"Scale factor: {scale_factor}")
        return None, scale_factor

    def _align_mono_depth(
        self,
    ) -> Tuple[Optional[Dict[int, Path]], Dict[int, float], Dict[int, float]]:
        image_id_to_depth_path, scales, shifts = colmap_utils.align_mono_depth(
            recon_dir=self.output_dir / self.default_colmap_path(),
            depth_dir=self.depth_image_dir,
            verbose=self.verbose,
        )
        return image_id_to_depth_path, scales, shifts

    def _export_mask(self) -> Optional[Dict[int, Path]]:
        if self.mask_data is not None:
            mask_dir = self.output_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            image_id_to_mask_path = colmap_utils.get_mask_files(
                recon_dir=self.output_dir / self.default_colmap_path(),
                mask_dir=mask_dir,
                verbose=self.verbose,
            )
            return image_id_to_mask_path

    def _run_colmap(self, mask_path: Optional[Path] = None):
        """
        Args:
            mask_path: Path to the camera mask. Defaults to None.
        """
        self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)

        # set the image_dir if didn't copy
        if self.skip_image_processing:
            image_dir = self.data
        else:
            image_dir = self.image_dir

        assert self.feature_type is not None
        assert self.matcher_type is not None
        hloc_utils.run_hloc(
            image_dir=image_dir,
            colmap_dir=self.absolute_colmap_path,
            camera_model=CAMERA_MODELS[self.camera_type],
            verbose=self.verbose,
            feature_type=self.feature_type,
            matcher_type=self.matcher_type,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        if self.crop_bottom < 0.0 or self.crop_bottom > 1:
            raise RuntimeError("crop_bottom must be set between 0 and 1.")

        if self.crop_bottom > 0.0:
            self.crop_factor = (0.0, self.crop_bottom, 0.0, 0.0)
