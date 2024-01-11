"""Helpers for checking if programs are installed"""

import shutil
import sys

from gs_toolkit.utils.rich_utils import CONSOLE


def check_ffmpeg_installed():
    """Checks if ffmpeg is installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        CONSOLE.print("[bold red]Could not find ffmpeg. Please install ffmpeg.")
        print("See https://ffmpeg.org/download.html for installation instructions.")
        sys.exit(1)


def check_colmap_installed():
    """Checks if colmap is installed."""
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        CONSOLE.print("[bold red]Could not find COLMAP. Please install COLMAP.")
        print(
            "See https://colmap.github.io/install.html for installation instructions."
        )
        sys.exit(1)


def check_curl_installed():
    """Checks if curl is installed."""
    curl_path = shutil.which("curl")
    if curl_path is None:
        CONSOLE.print(
            "[bold red]Could not find [yellow]curl[red], Please install [yellow]curl"
        )
        sys.exit(1)
