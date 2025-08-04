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


def convert_to_floats(data):
    if isinstance(data, list):
        return [convert_to_floats(item) for item in data]
    elif isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            return data
    else:
        return data


def flatten_list_of_lists(list_of_lists):
    flattened_list = [
        item for sublist in list_of_lists for item in sublist
    ]

    return flattened_list


def get_yaml_input_file_contents_by_section(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    section_keys = list(entire_yaml_input_file[top_key])

    return top_key, section_keys


def get_local_residual_materials_blocks(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    yaml_input_file = entire_yaml_input_file[top_key]
    local_residual_materials_block = \
        yaml_input_file["residuals"]["local residual"]["materials"]

    local_residual_elem_set_names = list(local_residual_materials_block.keys())
    local_residual_params_blocks = [
        local_residual_materials_block[elem_set_name]
        for elem_set_name in local_residual_elem_set_names
    ]

    return local_residual_params_blocks


def get_materials_and_inverse_blocks(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    yaml_input_file = entire_yaml_input_file[top_key]
    local_residual_materials_block = \
        yaml_input_file["residuals"]["local residual"]["materials"]
    inverse_materials_block = \
        yaml_input_file["inverse"]["materials"]

    local_residual_elem_set_names = list(local_residual_materials_block.keys())
    inverse_materials_elem_set_names = list(local_residual_materials_block.keys())
    assert local_residual_elem_set_names == inverse_materials_elem_set_names
    local_residual_params_blocks = [
        local_residual_materials_block[elem_set_name]
        for elem_set_name in local_residual_elem_set_names
    ]
    inverse_params_blocks = [
        inverse_materials_block[elem_set_name]
        for elem_set_name in inverse_materials_elem_set_names
    ]

    return local_residual_params_blocks, inverse_params_blocks


def get_opt_param_info(inverse_blocks):
    param_names = []
    param_scales = []
    param_block_indices = []
    for block_idx, inverse_block in enumerate(inverse_blocks):
        block_param_names = list(inverse_block.keys())
        param_names.append(block_param_names)
        param_scales.append(convert_to_floats(list(inverse_block.values())))
        param_block_indices.append([block_idx] * len(block_param_names))

    flat_param_names = flatten_list_of_lists(param_names)
    flat_param_scales = flatten_list_of_lists(param_scales)
    flat_param_block_indices = flatten_list_of_lists(param_block_indices)

    return flat_param_names, flat_param_scales, flat_param_block_indices


def get_initial_opt_params(local_residual_params_blocks,
        inverse_blocks):
    initial_opt_params_list = []
    params_blocks = zip(local_residual_params_blocks, inverse_blocks)
    for local_residual_params_block, inverse_block in params_blocks:
        initial_opt_params_list.append(convert_to_floats([
            local_residual_params_block[name]
            for name in list(inverse_block.keys())
        ]))

    initial_opt_params = flatten_list_of_lists(initial_opt_params_list)

    return initial_opt_params


def get_opt_options(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    yaml_input_file = entire_yaml_input_file[top_key]
    inverse_block = yaml_input_file["inverse"]

    num_iterations = inverse_block["iteration limit"]
    gradient_tol = float(inverse_block["gradient tolerance"])
    max_ls_evals = inverse_block["max line search evals"]

    return num_iterations, gradient_tol, max_ls_evals


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
    local_residual_params_blocks, inverse_params_blocks = \
        get_materials_and_inverse_blocks(input_yaml)

    text_params_initial_values, text_param_scales = text_params_data
    num_text_params = len(text_params_initial_values)
    text_param_names = [f"p_{ii}" for ii in range(num_text_params)]

    opt_param_names, opt_param_scales, opt_param_block_indices = \
        get_opt_param_info(inverse_params_blocks)
    initial_param_values = np.r_[
        get_initial_opt_params(local_residual_params_blocks,
        inverse_params_blocks),
        text_params_initial_values
    ]

    opt_param_names += text_param_names
    opt_param_scales += text_param_scales

    initial_opt_params = transform_parameters(
        initial_param_values, opt_param_scales, False
    )

    opt_bounds = get_opt_bounds(opt_param_scales)

    return (opt_param_names, opt_param_scales, opt_param_block_indices,
        initial_opt_params, opt_bounds)


def update_yaml_input_file_parameters(input_yaml,
        param_names, param_values, param_block_indices):

    local_residual_params_blocks = \
        get_local_residual_materials_blocks(input_yaml)

    param_info = zip(param_names, param_values, param_block_indices)
    for param_name, param_value, param_block_idx in param_info:
        local_residual_params_blocks[param_block_idx][param_name] = \
            float(param_value)


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
