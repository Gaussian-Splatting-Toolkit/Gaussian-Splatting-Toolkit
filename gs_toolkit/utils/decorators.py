"""
Decorator definitions
"""
from typing import Callable, List

from gs_toolkit.utils import comms


def decorate_all(decorators: List[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class

    Args:
        decorators: list of decorators to add to all functions in the class
    """

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != "__init__":
                for decorator in decorators:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def check_profiler_enabled(func: Callable) -> Callable:
    """Decorator: check if profiler is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.profiler != "none":
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_viewer_enabled(func: Callable) -> Callable:
    """Decorator: check if the viewer or legacy viewer is enabled and only run on main process"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if (
            self.config.is_viewer_enabled() or self.config.is_viewer_legacy_enabled()
        ) and comms.is_main_process():
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_eval_enabled(func: Callable) -> Callable:
    """Decorator: check if evaluation step is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if (
            self.config.is_wandb_enabled()
            or self.config.is_tensorboard_enabled()
            or self.config.is_comet_enabled()
        ):
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if comms.is_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper
