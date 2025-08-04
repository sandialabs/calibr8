import numpy as np

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


def print_header(ref_dir_deriv):
    headers = ["step size", "grad'*dir", "FD approx", "abs error"]
    if ref_dir_deriv >= 0.:
        print(f"{headers[0]:>19}  {headers[1]:>17}  "
              f"{headers[2]:>17}  {headers[3]:>17}")
        print(f"{'-' * 9:>19}  {'-' * 9:>17}  {'-' * 9:>17}  {'-' * 9:>17}")
    else:
        print(f"{headers[0]:>19}  {headers[1]:>18}  "
              f"{headers[2]:>18}  {headers[3]:>17}")
        print(f"{'-' * 9:>19}  {'-' * 9:>18}  {'-' * 9:>18}  {'-' * 9:>17}")


def main():
    parser = \
        argparse.ArgumentParser(description="calibrate with a python optimizer")
    parser.add_argument("input_file", type=str, help="inverse input yaml file")
    parser.add_argument("-n", "--num_procs", type=int, default=1,
        help="number of MPI ranks")
    parser.add_argument("-s", "--seed", type=int, default=22,
        help="seed for random direction RNG")
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
    rng_seed = args.seed

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

    # make input parameter
    rng = np.random.default_rng(rng_seed)
    num_params = len(opt_init_params)
    random_direction = rng.uniform(-1., 1., num_params)
    perturbations = np.logspace(0, -12, 13)

    obj_and_grad_run_command = get_run_command(num_procs)
    obj_only_run_command = get_run_command(num_procs, False)

    J_ref, grad_ref = objective_and_gradient(opt_init_params,
        scales=opt_param_scales, param_names=opt_param_names,
        block_indices=opt_param_block_indices,
        input_yaml=input_yaml, run_command=obj_and_grad_run_command,
        num_text_params=num_text_params,
        text_params_filename=text_parameters_opt_values_filename
    )
    ref_dir_deriv = random_direction @ grad_ref

    objective_value = partial(objective_and_gradient,
        scales=opt_param_scales, param_names=opt_param_names,
        block_indices=opt_param_block_indices,
        input_yaml=input_yaml, run_command=obj_only_run_command,
        num_text_params=num_text_params,
        text_params_filename=text_parameters_opt_values_filename,
        evaluate_objective_only=True
    )

    print_header(ref_dir_deriv)

    for idx, perturbation in enumerate(perturbations):
        perturbed_dir = perturbation * random_direction
        params_plus = opt_init_params + perturbed_dir
        params_minus = opt_init_params - perturbed_dir

        J_plus = objective_value(params_plus)
        J_minus = objective_value(params_minus)

        fd_dir_deriv = (J_plus - J_minus) / (2. * perturbation)
        abs_diff_dir_deriv = np.abs(ref_dir_deriv - fd_dir_deriv)

        print(f"{perturbation:>19.11e}  {ref_dir_deriv:>15.11e}  "
              f"{fd_dir_deriv:>15.11e}  {abs_diff_dir_deriv:>15.11e}")

    cleanup_files()
