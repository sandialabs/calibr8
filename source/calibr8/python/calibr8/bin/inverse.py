import numpy as np
import pickle

import argparse
import yaml

from scipy.optimize import minimize

from calibr8.util.driver_support import (
    evaluate_objective_or_gradient,
    OptimizationIterator
)

from calibr8.util.input_file_io import (
    cleanup_files,
    get_opt_options,
    setup_opt_parameters,
    setup_text_parameters,
    standard_parser,
    write_output_file
)


def main():
    parser = standard_parser()
    parser.add_argument("-o", "--output_file", type=str,
        default="calibrated_params.txt",
        help="output file that contains the calibrated parameters")
    parser.add_argument("--run_objective_only", action="store_true",
        help="flag to indicate whether to run objective only (no optimization)")

    args = parser.parse_args()

    input_files = args.input_files

    # optional arguments that have defaults
    num_procs = args.num_procs
    output_file = args.output_file
    use_srun = args.use_srun
    run_objective_only = args.run_objective_only

    # optional arguments that do not have defaults
    text_parameters_initial_values_file = \
        args.text_parameters_initial_values_file
    text_parameters_scales_file = args.text_parameters_scales_file
    text_parameters_opt_values_filename = \
        args.text_parameters_opt_values_filename

    text_params_data = setup_text_parameters(
        text_parameters_initial_values_file,
        text_parameters_scales_file,
        text_parameters_opt_values_filename
    )
    num_text_params = len(text_params_data[0])

    input_yamls = [
        yaml.safe_load(open(input_file, "r")) for input_file in input_files
    ]

    opt_param_names, opt_param_scales, opt_param_block_indices, \
        opt_init_params, opt_bounds = \
        setup_opt_parameters(input_yamls[0], text_params_data)


    num_iters, gradient_tol, max_ls_evals = get_opt_options(input_yamls[0])
    l_bfgs_b_opts = {
        "maxiter": num_iters,
        "gtol": gradient_tol,
        "maxls": max_ls_evals,
        "ftol": 10. * np.finfo(float).eps
    }

    objective_args = (
        opt_param_scales, opt_param_names, opt_param_block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_parameters_opt_values_filename
    )

    if run_objective_only:
        evaluate_gradient = False
        evaluate_objective_or_gradient(
            opt_init_params, *objective_args, evaluate_gradient
        )
        cleanup_files(evaluate_gradient)
    else:
        evaluate_gradient = True
        opt_iterator = OptimizationIterator(objective_args)
        res = minimize(
            fun=opt_iterator.objective_fun_and_grad,
            x0=opt_init_params,
            method="L-BFGS-B",
            jac=True,
            bounds=opt_bounds,
            options=l_bfgs_b_opts,
            callback=opt_iterator.callback
        )

        with open("minimize_results.pkl", "wb") as file:
            pickle.dump(res, file)

        write_output_file(res.x, opt_param_scales, opt_param_names,
            output_file)

        cleanup_files(evaluate_gradient)
