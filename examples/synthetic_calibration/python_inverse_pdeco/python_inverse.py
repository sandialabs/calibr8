import numpy as np

import subprocess

import yaml

from functools import partial
from scipy.optimize import fmin_l_bfgs_b


class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def get_run_command(num_proc):
    command = f"mpiexec -n {num_proc} objective run.yaml true"

    return command


def transform_parameters(values, scales, transform_from_canonical):
    transformed_params = np.array([
        value_transform(value, scale, transform_from_canonical)
        for value, scale in zip(values, scales)
    ])

    return transformed_params


def value_transform(value, scale, transform_from_canonical):
    if isinstance(scale, float):
        return log_transform(value, scale, transform_from_canonical)
    else:
        return bounds_transform(value, scale, transform_from_canonical)


def log_transform(value, ref_value, transform_from_canonical):
    if transform_from_canonical:
        transformed_value = ref_value * np.exp(value)
    else:
        transformed_value = np.log(value / ref_value)

    return transformed_value


def bounds_transform(value, bounds, transform_from_canonical):
    span = 0.5 * (bounds[1] - bounds[0])
    mean = 0.5 * (bounds[0] + bounds[1])

    if transform_from_canonical:
        transformed_value = span * value + mean
    else:
        transformed_value = (value - mean) / span

    return transformed_value


def first_deriv_transform(value, scale):
    if isinstance(scale, float):
        return value
    else:
        return 0.5 * (scale[1] - scale[0])


def get_opt_bounds(scales):
    return [get_opt_bounds_by_type(scale) for scale in scales]


def get_opt_bounds_by_type(scale):
    if isinstance(scale, float):
        return [None, None]
    else:
        return [-1., 1.]


def grad_transform(grad, values, scales):
    transformed_grad = np.array([
        grad_component * first_deriv_transform(value, scale)
        for grad_component, value, scale in zip(grad, values, scales)
    ])

    return transformed_grad


def get_yaml_input_file_contents_by_section(entire_yaml_input_file):
    top_key = list(entire_yaml_input_file.keys())[0]
    section_keys = list(entire_yaml_input_file[top_key])

    return top_key, section_keys


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

    num_iterations = inverse_block["iteration limit"]
    gradient_tol = float(inverse_block["gradient tolerance"])
    max_ls_evals = inverse_block["max line search evals"]

    return num_iterations, gradient_tol, max_ls_evals


def update_yaml_input_file_parameters(input_yaml, param_names, param_values):

    local_residual_params_block, _ = \
        get_materials_and_inverse_blocks(input_yaml)

    for param_name, param_value in zip(param_names, param_values):
        local_residual_params_block[param_name] = float(param_value)


def setup_opt_params(input_yaml):
    local_residual_params_block, inverse_params_block = \
        get_materials_and_inverse_blocks(input_yaml)

    opt_param_names, opt_param_scales = get_opt_param_info(inverse_params_block)

    initial_opt_params = transform_parameters(
        get_initial_opt_params(local_residual_params_block, opt_param_names),
        opt_param_scales, False)

    opt_bounds = get_opt_bounds(opt_param_scales)

    return opt_param_names, opt_param_scales, initial_opt_params, opt_bounds


def cleanup_files():
    files = ["run.yaml", "objective_value.txt", "objective_gradient.txt"]
    subprocess.run(["rm"] + files)


def objective_and_gradient(params, scales, param_names,
        input_yaml, run_command):

    unscaled_params = transform_parameters(params, scales,
        transform_from_canonical=True)

    update_yaml_input_file_parameters(input_yaml,
        param_names, unscaled_params)

    with open("run.yaml", "w") as file:
        yaml.dump(input_yaml, file, default_flow_style=False, sort_keys=False,
            Dumper=IndentDumper)

    subprocess.run(["bash", "-c", run_command])

    J = np.loadtxt("objective_value.txt")
    grad = grad_transform(np.loadtxt("objective_gradient.txt"),
        unscaled_params, scales)

    return J, grad

# TODO:
# x. get optimizer options from inverse block
# x. parameter scaling and unscaling
# x. put optimizer in
# x. cleanup function (remove run.yaml, objective*.txt files)
# 5. make script callable + input parsing


# these will be input arguments:
input_file = "pdeco_notch2D_small_J2_plane_stress.yaml"
num_proc = 1

with open(input_file, "r") as file:
    input_yaml = yaml.safe_load(file)

opt_param_names, opt_param_scales, initial_opt_params, opt_bounds = \
    setup_opt_params(input_yaml)

run_command = get_run_command(num_proc)

pt_objective_and_grad = partial(objective_and_gradient,
    scales=opt_param_scales, param_names=opt_param_names,
    input_yaml=input_yaml, run_command=run_command)

num_iters, gradient_tol, max_ls_evals = get_opt_options(input_yaml)

opt_params, fun_vals, cvg_dict = fmin_l_bfgs_b(
    pt_objective_and_grad, initial_opt_params,
    bounds=opt_bounds,
    maxiter=num_iters, pgtol=gradient_tol, maxls=max_ls_evals,
    iprint=1, factr=10
)

unscaled_opt_params = transform_parameters(opt_params, opt_param_scales, True)
np.savetxt("calibrated_params.txt", unscaled_opt_params)
print(f"opt params = {unscaled_opt_params}")

cleanup_files()
