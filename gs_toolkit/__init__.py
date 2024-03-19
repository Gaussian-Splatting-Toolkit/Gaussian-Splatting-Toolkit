# If code is running on Windows, set pathlib.PosixPath = pathlib.WindowsPath
import sys
import warnings

if sys.platform == "win32":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath

# Ignore all warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="deva")
warnings.filterwarnings("ignore", category=UserWarning, module="segment-anything")
warnings.filterwarnings("ignore", category=UserWarning, module="GroundingDINO")
