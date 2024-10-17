import numpy as np

import subprocess

import yaml


class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def build_run_command(calibr8_root, flavor, num_proc):
    command = f"cd {calibr8_root} && " \
              f"source env/{flavor}.sh && " \
              f"source capp-setup.sh && " \
              "capp load && " \
              "cd - > /dev/null &&" \
              "mpiexec -n {num_proc} objective run.yaml"

    return command


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


def update_yaml_input_file_parameters(local_residual_params_block,
        param_names, param_values):
    for param_name, param_value in zip(param_names, param_values):
        local_residual_params_block[param_name] = float(param_value)

# TODO:
# 1. get optimizer options from inverse block
# 2. parameter scaling and unscaling
# 3. put optimizer in
# 4. cleanup function (remove run.yaml, objective*.txt files)
# 5. make script callable + input parsing


# these will be input arguments:
input_file = "pdeco_notch2D_small_J2_plane_stress.yaml"
calibr8_root = "/Users/dtseidl/char/calibr8/github-calibr8"
flavor = "osx-shared"
num_proc = 1

with open(input_file, "r") as file:
    input_yaml = yaml.safe_load(file)

local_residual_params_block, inverse_params_block = \
    get_materials_and_inverse_blocks(input_yaml)

opt_param_names, opt_param_scales = get_opt_param_info(inverse_params_block)

initial_opt_params = get_initial_opt_params(local_residual_params_block,
    opt_param_names)

new_values = 1.1 * initial_opt_params

update_yaml_input_file_parameters(local_residual_params_block,
        opt_param_names, new_values)

with open("run.yaml", "w") as file:
    yaml.dump(input_yaml, file, default_flow_style=False, sort_keys=False,
        Dumper=IndentDumper)

run_command = build_run_command(calibr8_root, flavor, num_proc)
eval_obj_and_grad_str = "true"
run_command += f" {eval_obj_and_grad_str}"
subprocess.run(["bash", "-c", run_command])
