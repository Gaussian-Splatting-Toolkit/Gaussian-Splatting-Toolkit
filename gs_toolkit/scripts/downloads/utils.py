# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
from dataclasses import dataclass
from pathlib import Path

from gs_toolkit.configs.base_config import PrintableConfig


@dataclass
class DatasetDownload(PrintableConfig):
    """Download a dataset"""

    capture_name = None

    save_dir: Path = Path("data/")
    """The directory to save the dataset to"""

    def download(self, save_dir: Path) -> None:
        """Download the dataset"""
        raise NotImplementedError
