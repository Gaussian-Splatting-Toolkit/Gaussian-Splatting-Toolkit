"""functionality to handle multiprocessing syncing and communicating"""
import math
from typing import Union
import torch
import torch.distributed as dist

LOCAL_PROCESS_GROUP = None


def is_dist_avail_and_initialized() -> bool:
    """Returns True if distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get total number of available gpus"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of current thread"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """The rank of the current process within the local (per-machine) process group."""
    if not is_dist_avail_and_initialized():
        return 0
    assert (
        LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please use launch() to spawn processes!"
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    The size of the per-machine process group,
    i.e. the number of processes per machine.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size(group=LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """check to see if you are currently on the main process"""
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if dist.get_world_size() == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[get_local_rank()])
    else:
        dist.barrier()


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(
    znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"
):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r  # noqa: E741
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
