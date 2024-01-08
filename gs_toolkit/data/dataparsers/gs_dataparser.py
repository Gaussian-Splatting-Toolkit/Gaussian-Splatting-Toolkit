from argparse import ArgumentParser
from gs_toolkit.data.dataparsers.base_dataparser import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
)


def gs_parser() -> [ArgumentParser, ModelParams, OptimizationParams, PipelineParams]:
    """Returns a parser for the Gaussian Splatting."""
    parser = ArgumentParser(description="parser for gaussian splatting")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        help="number of iterations to test",
        default=[7_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        help="number of iterations to save",
        default=[7_000, 30_000],
    )
    parser.add_argument("--start_entropy_iterations", type=int, default=28_000)
    parser.add_argument("--entropy_iterations_num", type=int, default=2_000)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    return parser, lp, op, pp
