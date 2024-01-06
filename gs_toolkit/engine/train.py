from data.dataparsers.gs_dataparser import gs_parser
from data.dataparsers.base_dataparser import GroupParams
from model_components.scene import Scene


class Trainer:
    def __init__(self) -> None:
        pass

    def training(
        self,
        dataset: GroupParams,
        opt: GroupParams,
        pipe: GroupParams,
        testing_iterations: list[int],
        saving_iterations: list[int],
        checkpoint_iterations: int,
        checkpoint: str,
        debug_from: int,
        start_entropy_iterations: int,
        entropy_iterations_num: int,
    ):
        pass

    def training_report(
        self,
        tb_writer,
        iteration,
        Ll1,
        loss,
        l1_loss,
        ealpsed,
        testing_iterations,
        scene: Scene,
        renderFunc,
        renderArgs,
    ):
        pass


if __name__ == "__main__":
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
