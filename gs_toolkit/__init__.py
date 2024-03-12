# If code is running on Windows, set pathlib.PosixPath = pathlib.WindowsPath
import sys

if sys.platform == "win32":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath
