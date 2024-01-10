import os
from gs_toolkit.utils.rich_utils import CONSOLE, status
from gs_toolkit.scripts.scripts import run_command
from dataclasses import dataclass
import shutil
from gs_toolkit.data.process_data.base_converter_to_gstk_dataset import (
    BaseConverterToGstkDataset,
)


@dataclass
class ColmapConverterToGstkDataset(BaseConverterToGstkDataset):
    """Process data using Colmap.

    This script does the following:

    1. Select keyframes from the input video.
    2. Extract the pose and sparse point cloud using COLMAP.
    """

    fps: int = 3
    """Number of frames per second to extract from the video."""
    colmap_command: str = "colmap"
    """Path to the colmap executable."""
    magick_command: str = "magick"
    """Path to the magick executable."""
    camera: str = "OPENCV"
    """Camera model to use for colmap."""
    no_gpu: bool = False
    """If True, disable GPU usage."""
    skip_matching: bool = False
    """If True, skip feature matching"""
    resize: bool = False
    """If True, resize images to 50%, 25% and 12.5% of the original size."""

    def main(self) -> None:
        """Main function."""
        summary_log = []
        num_frames = 0

        # If the data is a video, extract the keyframes.
        if self.input.suffix in [".mp4", ".avi", ".mov", ".MP4", ".MOV"]:
            num_frames = self.extract_keyframes(self.fps)
            
        assert num_frames > 0, "No frames extracted. Exiting."
        # TODO: Down the vocabtree if the number of frames is too large.

        use_gpu = 1 if not self.no_gpu else 0

        data_dir_str = str(self.data_dir)

        if not self.skip_matching:
            os.makedirs(data_dir_str + "/distorted/sparse", exist_ok=True)

            ## Feature extraction
            feat_extracton_cmd = (
                self.colmap_command + " feature_extractor "
                "--database_path "
                + data_dir_str
                + "/distorted/database.db \
                --image_path "
                + data_dir_str
                + "/input \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model "
                + self.camera
                + " \
                --SiftExtraction.use_gpu "
                + str(use_gpu)
            )
            with status("Extracting features...", spinner="bouncingBall", verbose=self.verbose):
                run_command(feat_extracton_cmd, verbose=self.verbose)

            ## Feature matching
            feat_matching_cmd = (
                self.colmap_command
                + " exhaustive_matcher \
                --database_path "
                + data_dir_str
                + "/distorted/database.db \
                --SiftMatching.use_gpu "
                + str(use_gpu)
            )
            with status("Matching features...", spinner="bouncingBall", verbose=self.verbose):
                run_command(feat_matching_cmd, verbose=self.verbose)

            ### Bundle adjustment
            # The default Mapper tolerance is unnecessarily large,
            # decreasing it speeds up bundle adjustment steps.
            mapper_cmd = (
                self.colmap_command
                + " mapper \
                --database_path "
                + data_dir_str
                + "/distorted/database.db \
                --image_path "
                + data_dir_str
                + "/input \
                --output_path "
                + data_dir_str
                + "/distorted/sparse \
                --Mapper.ba_global_function_tolerance=0.000001"
            )
            with status("Running bundle adjustment...", spinner="bouncingBall", verbose=self.verbose):
                run_command(mapper_cmd, verbose=self.verbose)

        ### Image undistortion
        ## We need to undistort our images into ideal pinhole intrinsics.
        img_undist_cmd = (
            self.colmap_command
            + " image_undistorter \
            --image_path "
            + data_dir_str
            + "/input \
            --input_path "
            + data_dir_str
            + "/distorted/sparse/0 \
            --output_path "
            + data_dir_str
            + "\
            --output_type COLMAP"
        )
        with status("Undistorting images...", spinner="bouncingBall", verbose=self.verbose):
            run_command(img_undist_cmd, verbose=self.verbose)

        files = os.listdir(data_dir_str + "/sparse")
        os.makedirs(data_dir_str + "/sparse/0", exist_ok=True)
        # Copy each file from the source directory to the destination directory
        for file in files:
            if file == "0":
                continue
            source_file = os.path.join(data_dir_str, "sparse", file)
            destination_file = os.path.join(data_dir_str, "sparse", "0", file)
            shutil.move(source_file, destination_file)

        if self.resize:
            print("Copying and resizing...")

            # Resize images.
            os.makedirs(data_dir_str + "/images_2", exist_ok=True)
            os.makedirs(data_dir_str + "/images_4", exist_ok=True)
            os.makedirs(data_dir_str + "/images_8", exist_ok=True)
            # Get the list of files in the source directory
            files = os.listdir(data_dir_str + "/images")
            # Copy each file from the source directory to the destination directory
            for file in files:
                source_file = os.path.join(data_dir_str, "images", file)

                destination_file = os.path.join(data_dir_str, "images_2", file)
                shutil.copy2(source_file, destination_file)
                exit_code = os.system(
                    self.magick_command + " mogrify -resize 50% " + destination_file
                )
                if exit_code != 0:
                    CONSOLE.log(f"50% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)

                destination_file = os.path.join(data_dir_str, "images_4", file)
                shutil.copy2(source_file, destination_file)
                exit_code = os.system(
                    self.magick_command + " mogrify -resize 25% " + destination_file
                )
                if exit_code != 0:
                    CONSOLE.log(f"25% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)

                destination_file = os.path.join(data_dir_str, "images_8", file)
                shutil.copy2(source_file, destination_file)
                exit_code = os.system(
                    self.magick_command + " mogrify -resize 12.5% " + destination_file
                )
                if exit_code != 0:
                    CONSOLE.log(f"12.5% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)

        summary_log.append("Colmap processing complete.")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")
        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()
