""" Pixel sampling utils such as eroding of valid masks that we sample from. """

import torch
from torch import Tensor
from jaxtyping import Float


def dilate(
    tensor: Float[Tensor, "bs 1 H W"], kernel_size=3
) -> Float[Tensor, "bs 1 H W"]:
    """Dilate a tensor with 0s and 1s. 0s will be be expanded based on the kernel size.

    Args:
        kernel_size: Size of the pooling region. Dilates/contracts 1 pixel if kernel_size is 3.
    """

    unique_vals = torch.unique(tensor)
    if any(val not in (0, 1) for val in unique_vals) or tensor.dtype != torch.float32:
        raise ValueError(
            "Input tensor should contain only values 0 and 1, and should have dtype torch.float32."
        )

    return torch.nn.functional.max_pool2d(
        tensor, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
    )


def erode(
    tensor: Float[Tensor, "bs 1 H W"], kernel_size=3
) -> Float[Tensor, "bs 1 H W"]:
    """Erode a tensor with 0s and 1s. 1s will be expanded based on the kernel size.

    Args:
        kernel_size: Size of the pooling region. Erodes/expands 1 pixel if kernel_size is 3.
    """

    unique_vals = torch.unique(tensor)
    if any(val not in (0, 1) for val in unique_vals) or tensor.dtype != torch.float32:
        raise ValueError(
            "Input tensor should contain only values 0 and 1, and should have dtype torch.float32."
        )

    x = 1 - dilate(1 - tensor, kernel_size=kernel_size)
    # set edge pixels to 0
    p = (kernel_size - 1) // 2
    x[:, :, :p, :] *= 0
    x[:, :, :, :p] *= 0
    x[:, :, -p:, :] *= 0
    x[:, :, :, -p:] *= 0
    return x


def erode_mask(tensor: Float[Tensor, "bs 1 H W"], pixel_radius: int = 1):
    """Erode a mask. Expands 1 values to nearby pixels with a max pooling operation.
    A pixel radius of 1 will also extend the 1s along the diagonal.

    Args:
        pixel_radius: The number of pixels away from valid pixels (1s) that we may sample.
    """
    kernel_size = 1 + 2 * pixel_radius
    return erode(tensor, kernel_size=kernel_size)
