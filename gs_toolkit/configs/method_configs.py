"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import tyro

from gs_toolkit.configs.base_config import ViewerConfig
from gs_toolkit.engine.optimizers import AdamOptimizerConfig
from gs_toolkit.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from gs_toolkit.engine.trainer import TrainerConfig
from gs_toolkit.models.gaussian_splatting import GaussianSplattingModelConfig
from gs_toolkit.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from gs_toolkit.pipelines.base_pipeline import VanillaPipelineConfig
from gs_toolkit.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "gaussian-splatting": "Gaussian Splatting model",
}

method_configs["gaussian-splatting"] = TrainerConfig(
    method_name="gaussian-splatting",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=ColmapDataParserConfig(load_3D_points=True),
        ),
        model=GaussianSplattingModelConfig(),
    ),
    optimizers={
        "xyz": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-5, max_steps=30000
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


def merge_methods(
    methods, method_descriptions, new_methods, new_descriptions, overwrite=True
):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(
        sorted(method_descriptions.items(), key=lambda x: x[0])
    )
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions

AnnotatedBaseConfigUnion = (
    tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(
                defaults=all_methods, descriptions=all_descriptions
            )
        ]
    ]
)
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""