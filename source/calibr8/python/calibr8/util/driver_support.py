import numpy as np

import subprocess
import yaml

from calibr8.util.input_file_io import (
    IndentDumper,
    update_yaml_input_file_parameters
)
from calibr8.util.parameter_transforms import (
    grad_transform,
    transform_parameters
)


def get_run_command(num_proc, evaluate_gradient=True):
    if evaluate_gradient:
        return f"mpiexec -n {num_proc} objective run.yaml true"
    else:
        return f"mpiexec -n {num_proc} objective run.yaml false"


def objective_and_gradient(params, scales, param_names,
        input_yaml, run_command,
        num_text_params, text_params_filename,
        evaluate_objective_only=False):

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
    if evaluate_objective_only:
        return J

    grad = grad_transform(np.loadtxt("objective_gradient.txt"),
        unscaled_params, scales)

    return J, grad
