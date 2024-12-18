import numpy as np

import argparse
import subprocess
import yaml

from functools import partial
from scipy.optimize import fmin_l_bfgs_b

from calibr8.util.parameter_transforms import (
    get_opt_bounds,
    transform_parameters
)


class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def get_yaml_input_file_contents_by_section(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    section_keys = list(entire_yaml_input_file[top_key])

    return top_key, section_keys


def get_local_residual_materials_block(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    yaml_input_file = entire_yaml_input_file[top_key]
    local_residual_materials_block = \
        yaml_input_file["residuals"]["local residual"]["materials"]

    # only single block calibration problems are supported
    local_residual_elem_set_names = list(local_residual_materials_block.keys())
    assert len(local_residual_elem_set_names) == 1
    elem_set_name = local_residual_elem_set_names[0]

    local_residual_params_block = local_residual_materials_block[elem_set_name]

    return local_residual_params_block


def get_materials_and_inverse_blocks(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    yaml_input_file = entire_yaml_input_file[top_key]
    local_residual_materials_block = \
        yaml_input_file["residuals"]["local residual"]["materials"]
    inverse_materials_block = \
        yaml_input_file["inverse"]["materials"]

    # only single block calibration problems are supported
    local_residual_elem_set_names = list(local_residual_materials_block.keys())
    inverse_materials_elem_set_names = list(local_residual_materials_block.keys())
    assert local_residual_elem_set_names == inverse_materials_elem_set_names
    assert len(local_residual_elem_set_names) == 1
    elem_set_name = local_residual_elem_set_names[0]

    local_residual_params_block = local_residual_materials_block[elem_set_name]
    inverse_params_block = inverse_materials_block[elem_set_name]

    return local_residual_params_block, inverse_params_block


def get_opt_param_info(inverse_block):
    opt_param_names = list(inverse_block.keys())
    opt_param_scales = list(inverse_block.values())

    return opt_param_names, opt_param_scales


def get_initial_opt_params(local_residual_params_block,
        opt_param_names):
    initial_opt_params = np.array(
        [local_residual_params_block[name] for name in opt_param_names]
    )

    return initial_opt_params


def get_opt_options(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    yaml_input_file = entire_yaml_input_file[top_key]
    inverse_block = yaml_input_file["inverse"]

    obj_type = inverse_block["objective type"]
    if obj_type == "adjoint":
        obj_exe = "objective"
    elif obj_type == "vfm":
        obj_exe = "vfm_objective"
    else:
        raise ValueError("objective type not supported")

    num_iterations = inverse_block["iteration limit"]
    gradient_tol = float(inverse_block["gradient tolerance"])
    max_ls_evals = inverse_block["max line search evals"]

    return obj_exe, num_iterations, gradient_tol, max_ls_evals


def convert_none_or_float(string):
    if string == "None":
        return None
    else:
        return float(string)


def convert_str_scale(str_scale):
    split_str_scale = str_scale.split()
    scale = [convert_none_or_float(string) for string in split_str_scale]
    if len(scale) == 1:
        return scale[0]
    else:
        return scale


def setup_text_parameters(init_values_file, scales_file, opt_filename):
    # init_values_file -> contains a num_params file with initial values
    # scales_file -> empty or num_params length file with scaling factors
    #                viable scaling factors include:
    #                1. a single float -> log scaling; no bounds
    #                2. two floats -> linear scaling; scales = bounds
    #                3. None -> no scaling; no bounds
    # opt_filename -> name for a file that will be written at each
    #                 optimization iteration for use by Calibr8
    #                 (contains unscaled parameter values)

    if init_values_file is None:
        return np.empty(0), []

    assert opt_filename is not None

    init_values = np.loadtxt(init_values_file)
    num_params = len(init_values)

    if scales_file is None:
        scales = np.ones(num_params)
    else:
        with open(f"{scales_file}", "r") as file:
            str_scales = [line.strip() for line in file]
        scales = [convert_str_scale(str_scale) for str_scale in str_scales]

    return init_values, scales


def setup_opt_parameters(input_yaml, text_params_data):
    local_residual_params_block, inverse_params_block = \
        get_materials_and_inverse_blocks(input_yaml)

    text_params_initial_values, text_param_scales = text_params_data
    num_text_params = len(text_params_initial_values)
    text_param_names = [f"p_{ii}" for ii in range(num_text_params)]

    opt_param_names, opt_param_scales = get_opt_param_info(inverse_params_block)
    initial_param_values = np.r_[
        get_initial_opt_params(local_residual_params_block, opt_param_names),
        text_params_initial_values
    ]

    opt_param_names += text_param_names
    opt_param_scales += text_param_scales

    initial_opt_params = transform_parameters(
        initial_param_values, opt_param_scales, False
    )

    opt_bounds = get_opt_bounds(opt_param_scales)

    return opt_param_names, opt_param_scales, initial_opt_params, opt_bounds


def update_yaml_input_file_parameters(input_yaml, param_names, param_values):

    local_residual_params_block = \
        get_local_residual_materials_block(input_yaml)

    for param_name, param_value in zip(param_names, param_values):
        local_residual_params_block[param_name] = float(param_value)


def write_output_file(opt_params, opt_param_scales, param_names,
        output_file):

    unscaled_opt_params = \
        transform_parameters(opt_params, opt_param_scales, True)
    with open(f"{output_file}", "w") as file:
        for name, value in zip(param_names, unscaled_opt_params):
            file.write(f"{name}: {value:.12e}\n")


def cleanup_files():
    files = ["run.yaml", "objective_value.txt", "objective_gradient.txt"]
    subprocess.run(["rm"] + files)


def objective_and_gradient(params, scales, param_names,
        input_yaml, run_command,
        num_text_params, text_params_filename):

    unscaled_params = transform_parameters(params, scales,
        transform_from_canonical=True)

    num_params = len(params)
    num_input_file_params = num_params - num_text_params

    if num_input_file_params > 0:
        update_yaml_input_file_parameters(input_yaml,
            param_names[:num_input_file_params],
            unscaled_params[:num_input_file_params]
        )

    with open("run.yaml", "w") as file:
        yaml.dump(input_yaml, file, default_flow_style=False, sort_keys=False,
            Dumper=IndentDumper)

    if text_params_filename is not None:
        np.savetxt(text_params_filename, unscaled_params[-num_text_params:])

    subprocess.run(["bash", "-c", run_command])

    J = np.loadtxt("objective_value.txt")
    #grad = grad_transform(np.loadtxt("objective_gradient.txt"),
    #    unscaled_params, scales)

    # cheese so that text parameters can be debugged
    grad = grad_transform(np.loadtxt("objective_gradient.txt"),
        unscaled_params[:num_input_file_params],
        scales[:num_input_file_params])

    return J, np.r_[grad, np.zeros(num_text_params)]


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
    parser.add_argument("-pi", "--text_parameters_initial_values_file", type=str,
        help="text file that contains additional parameters initial values")
    parser.add_argument("-ps", "--text_parameters_scales_file", type=str,
        help="text file that contains additional parameters scales")
    # will be written
    parser.add_argument("-po", "--text_parameters_opt_values_filename", type=str,
        help="name for text file that contains additional"
             " parameters optimization values")

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

    opt_param_names, opt_param_scales, opt_init_params, opt_bounds = \
        setup_opt_params(input_yaml, text_params_data)

    obj_exe, num_iters, gradient_tol, max_ls_evals = get_opt_options(input_yaml)

    run_command = get_run_command(num_procs, obj_exe)

    pt_objective_and_grad = partial(objective_and_gradient,
        scales=opt_param_scales, param_names=opt_param_names,
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

    write_output_file(opt_params, opt_param_scales, opt_param_names,
        output_file)

    cleanup_files()
