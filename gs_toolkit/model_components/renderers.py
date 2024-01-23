"""
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import contextlib

# import math
from typing import Literal, Optional, Union, Generator

# from typing import Tuple

# import nerfacc
# import torch
from jaxtyping import Float

# from jaxtyping import Int
from torch import Tensor

# from torch import nn

# from gs_toolkit.cameras.rays import RaySamples
# from gs_toolkit.utils import colors
# from gs_toolkit.utils.math import components_from_spherical_harmonics, safe_normalize

BackgroundColor = Union[
    Literal["random", "last_sample", "black", "white"],
    Float[Tensor, "3"],
    Float[Tensor, "*bs 3"],
]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


@contextlib.contextmanager
def background_color_override_context(
    mode: Float[Tensor, "3"]
) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color
