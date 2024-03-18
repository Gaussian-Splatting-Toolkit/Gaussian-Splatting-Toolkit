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

from typing import Literal, Optional, Union, Generator
from jaxtyping import Float
from torch import Tensor

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
