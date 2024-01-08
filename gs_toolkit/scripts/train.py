import tyro
from gs_toolkit.engine.trainer import Trainer
from gs_toolkit.configs.method_config import AnnotatedBaseConfigUnion
from gs_toolkit.data.dataparsers.gs_dataparser import gs_parser
from gs_toolkit.configs.config_utils import convert_markup_to_ansi


def main() -> None:
    parser, lp, op, pp = gs_parser()
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    trainer = Trainer()
    trainer.training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.start_entropy_iterations,
        args.entropy_iterations_num,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
