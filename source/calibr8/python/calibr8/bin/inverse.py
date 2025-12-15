import numpy as np
import pickle

import argparse
import yaml

from functools import partial
from scipy.optimize import fmin_l_bfgs_b

from calibr8.util.driver_support import (
    get_run_command,
    objective_and_gradient
)

from calibr8.util.input_file_io import (
    cleanup_files,
    get_opt_options,
    setup_opt_parameters,
    setup_text_parameters,
    write_output_file
)


def main():
    parser = \
        argparse.ArgumentParser(description="calibrate with a python optimizer")
    parser.add_argument("input_file", type=str, help="inverse input yaml file")
    parser.add_argument("-n", "--num_procs", type=int, default=1,
        help="number of MPI ranks")
    parser.add_argument("-o", "--output_file", type=str,
        default="calibrated_params.txt",
        help="output file that contains the calibrated parameters")
    # will be read in
    parser.add_argument("-pi", "--text_parameters_initial_values_file",
        type=str,
        help="text file that contains additional parameters initial values")
    parser.add_argument("-ps", "--text_parameters_scales_file", type=str,
        help="text file that contains additional parameters scales")
    # will be written
    parser.add_argument("-po", "--text_parameters_opt_values_filename",
        type=str,
        help="name for text file that contains text parameters values")

    args = parser.parse_args()

    input_file = args.input_file

    # optional arguments that have defaults
    num_procs = args.num_procs
    output_file = args.output_file

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

    with open(input_file, "r") as file:
        input_yaml = yaml.safe_load(file)

    opt_param_names, opt_param_scales, opt_param_block_indices, \
        opt_init_params, opt_bounds = \
        setup_opt_parameters(input_yaml, text_params_data)

    num_iters, gradient_tol, max_ls_evals = get_opt_options(input_yaml)

    run_command = get_run_command(num_procs)

    pt_objective_and_grad = partial(objective_and_gradient,
        scales=opt_param_scales, param_names=opt_param_names,
        block_indices=opt_param_block_indices,
        input_yaml=input_yaml, run_command=run_command,
        num_text_params=num_text_params,
        text_params_filename=text_parameters_opt_values_filename
    )

    opt_params, fun_vals, cvg_dict = fmin_l_bfgs_b(
        pt_objective_and_grad, opt_init_params,
        bounds=opt_bounds,
        maxiter=num_iters, pgtol=gradient_tol, maxls=max_ls_evals,
        iprint=1, factr=10
    )
    with open("cvg_dict.pkl", "wb") as file:
        pickle.dump(cvg_dict, file)

    write_output_file(opt_params, opt_param_scales, opt_param_names,
        output_file)

    cleanup_files()
