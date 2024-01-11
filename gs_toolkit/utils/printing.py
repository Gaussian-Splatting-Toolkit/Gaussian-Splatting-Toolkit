"""A collection of common strings and print statements used throughout the codebase."""

from math import floor, log

from gs_toolkit.utils.rich_utils import CONSOLE


def print_tcnn_speed_warning(module_name: str):
    """Prints a warning about the speed of the TCNN."""
    CONSOLE.line()
    CONSOLE.print(
        f"[bold yellow]WARNING: Using a slow implementation for the {module_name} module. "
    )
    CONSOLE.print(
        "[bold yellow]:person_running: :person_running: "
        + "Install tcnn for speedups :person_running: :person_running:"
    )
    CONSOLE.print(
        "[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    CONSOLE.line()


def human_format(num):
    """Format a number in a more human readable way

    Args:
        num: number to format
    """
    units = ["", "K", "M", "B", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(num, k)))
    return f"{(num / k**magnitude):.2f} {units[magnitude]}"
