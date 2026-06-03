import numpy as np
import yaml

from calibr8.util.driver_support import (
    evaluate_objective_and_gradient,
    evaluate_objective_or_gradient
)
from calibr8.util.input_file_io import (
    cleanup_files,
    setup_opt_parameters,
    setup_text_parameters,
    standard_parser
)


def main():
    parser = standard_parser()
    parser.add_argument(
        "--out", type=str, default="gradient_fd.csv",
        help="CSV‐style output file (space separated)"
    )
    args = parser.parse_args()

    # read common args
    input_files = args.input_files
    num_procs   = args.num_procs
    use_srun    = args.use_srun
    rng_seed    = getattr(args, "seed", 44)
    out_file    = args.out

    # text‐param setup
    tp_init, tp_scales, tp_opt_name = (
        args.text_parameters_initial_values_file,
        args.text_parameters_scales_file,
        args.text_parameters_opt_values_filename
    )
    text_params_data = setup_text_parameters(tp_init, tp_scales, tp_opt_name)
    num_text_params = len(text_params_data[0])

    # load YAML inputs
    input_yamls = [yaml.safe_load(open(f, "r")) for f in input_files]

    # parameter setup
    (
        opt_names, opt_scales, opt_block_inds,
        opt_init_params, opt_bounds
    ) = setup_opt_parameters(input_yamls[0], text_params_data)

    # build RNG direction
    rng = np.random.default_rng(rng_seed)
    num_params = len(opt_init_params)
    dir_vec = rng.uniform(-1.0, 1.0, num_params)
    norm_dir = np.linalg.norm(dir_vec)
    if norm_dir > 0:
        dir_vec /= norm_dir

    # FD step sizes
    perturbations = np.logspace(-2, -9, 8)

    # pack args for driver
    obj_args = (
        opt_scales, opt_names, opt_block_inds,
        input_yamls, num_procs, use_srun,
        num_text_params, tp_opt_name
    )

    # reference f and ∇f
    J0, grad0, success = evaluate_objective_and_gradient(
        opt_init_params, *obj_args
    )
    if not success:
        raise RuntimeError("Reference evaluation failed")

    ref_dir_deriv = float(np.dot(dir_vec, grad0))

    # function for f-only calls
    f_only = lambda x: evaluate_objective_or_gradient(
        x, *(obj_args + (False,))
    )

    # write CSV header + data
    with open(out_file, "w") as fp:
        # header
        fp.write("step_size ref_dir_deriv fd_approx abs_error\n")
        # data rows
        for h in perturbations:
            p_plus  = opt_init_params +  h * dir_vec
            p_minus = opt_init_params -  h * dir_vec

            Jp = float(f_only(p_plus))
            Jm = float(f_only(p_minus))

            fd_dir = (Jp - Jm) / (2.0 * h)
            err    = abs(ref_dir_deriv - fd_dir)

            fp.write(f"{h:.11e} {ref_dir_deriv:.11e} "
                     f"{fd_dir:.11e} {err:.11e}\n")

    cleanup_files()


if __name__ == "__main__":
    main()
