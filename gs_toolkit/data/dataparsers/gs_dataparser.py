from argparse import ArgumentParser
from dataparsers.base_dataparser import ModelParams, OptimizationParams, PipelineParams


def gs_parser() -> ArgumentParser:
    """Returns a parser for the Gaussian Splatting."""
    parser = ArgumentParser(description="parser for gaussian splatting")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
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
    parser.add_argument("--start_checkpoint", type=str, default=None)
    return parser, lp, op, pp
