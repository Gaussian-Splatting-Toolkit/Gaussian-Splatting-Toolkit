"""
Base class to process images or video into a gs_toolkit dataset
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BaseConverterToGSToolkitDataset(ABC):
    """Base class to process images or video into a gs_toolkit dataset."""

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    eval_data: Optional[Path] = None
    """Path the eval data, either a video file or a directory of images. If set to None, the first will be used both for training and eval"""
    verbose: bool = False
    """If True, print extra logging."""

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    @property
    def image_dir(self) -> Path:
        return self.output_dir / "images"

    @abstractmethod
    def main(self) -> None:
        """This method implements the conversion logic for each type of data"""
        raise NotImplementedError(
            "the main method for conversion needs to be implemented"
        )
