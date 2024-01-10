import os
from gs_toolkit.utils.rich_utils import CONSOLE
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConverterToGstkDataset:
    """Base class to process images or video into a gs_toolkit dataset."""

    input: Path
    """Path to the data, either a video or a folder with images."""
    data_dir: Path
    """Path to the output directory."""
    eval_data: Path | None = None
    """Path to the evaluation data, either a video or a folder with images. If set to None, the first will be used for both training and evaluation."""
    verbose: bool = False
    """If True, print more information."""

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def extract_keyframes(self, fps: int) -> None:
        """Extract keyframes from the input video.

        Args:
            fps: The number of frames per second to extract.
        """
        # Process data with ffmpeg and save the results into input dir.
        with CONSOLE.status("Extracting keyframes...", spinner="bouncingBall"):
            os.system(f"ffmpeg -i {self.input} -vf fps={fps} {self.input_dir}/%06d.png")

    @property
    def input_dir(self) -> Path:
        """Path to the directory containing the images."""
        return self.data_dir / "input"

    @abstractmethod
    def main(self) -> None:
        """This method implements the conversion logic for each type of data"""
        raise NotImplementedError("The main method must be implemented.")
