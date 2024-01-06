from data.dataparsers import gs_parser


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    start_entropy_iterations,
    entropy_iterations_num,
):
    pass


if __name__ == "__main__":
    parser, lp, op, pp = gs_parser()
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
