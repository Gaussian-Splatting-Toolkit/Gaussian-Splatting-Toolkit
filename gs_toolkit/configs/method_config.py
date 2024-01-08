from __future__ import annotations

import tyro
from collections import OrderedDict
from gs_toolkit.engine.trainer import TrainerConfig

method_configs: dict[str, TrainerConfig] = {}
descriptions = {
    "gaussian-splatting": "Gaussian Splatting model",
}


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
