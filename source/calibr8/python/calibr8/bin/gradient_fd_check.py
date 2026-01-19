import numpy as np

import argparse
import yaml

from functools import partial
from scipy.optimize import fmin_l_bfgs_b

from calibr8.util.driver_support import (
    get_run_command,
    evaluate_objective_and_gradient,
    evaluate_objective_or_gradient
)

from calibr8.util.input_file_io import (
    cleanup_files,
    get_opt_options,
    setup_opt_parameters,
    setup_text_parameters,
    standard_parser,
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
    parser = standard_parser()

    args = parser.parse_args()

    input_files = args.input_files

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

    input_yamls = [
        yaml.safe_load(open(input_file, "r")) for input_file in input_files
    ]

    opt_param_names, opt_param_scales, opt_param_block_indices, \
        opt_init_params, opt_bounds = \
        setup_opt_parameters(input_yamls[0], text_params_data)

    # make input parameter
    rng = np.random.default_rng(rng_seed)
    num_params = len(opt_init_params)
    random_direction = rng.uniform(-1., 1., num_params)
    perturbations = np.logspace(-2, -9, 8)

    objective_args = (
        opt_param_scales, opt_param_names, opt_param_block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_parameters_opt_values_filename
    )

    J_ref, grad_ref = evaluate_objective_and_gradient(
        opt_init_params, *objective_args
    )
    ref_dir_deriv = random_direction @ grad_ref

    objective_value = (
        lambda x: evaluate_objective_or_gradient(
            x, *(objective_args + (False,))
        )
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

        print(f"{perturbation:>19.11e}  {ref_dir_deriv:>15.11e}  " \
              f"{fd_dir_deriv:>15.11e}  {abs_diff_dir_deriv:>15.11e}")

    cleanup_files()
