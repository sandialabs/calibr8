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


def get_run_command(num_procs, evaluate_gradient=True):
    if evaluate_gradient:
        return f"mpiexec -n {num_procs} objective run.yaml true"
    else:
        return f"mpiexec -n {num_procs} objective run.yaml false"


def run_objective_binary(params,
    scales, param_names, block_indices,
    input_yaml, num_procs,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    unscaled_params = transform_parameters(params, scales,
        transform_from_canonical=True)

    num_params = len(params)
    num_input_file_params = num_params - num_text_params

    if num_input_file_params > 0:
        update_yaml_input_file_parameters(input_yaml,
            param_names[:num_input_file_params],
            unscaled_params[:num_input_file_params],
            block_indices[:num_input_file_params]
        )

    with open("run.yaml", "w") as file:
        yaml.dump(input_yaml, file, default_flow_style=False, sort_keys=False,
            Dumper=IndentDumper)

    if text_params_filename is not None:
        np.savetxt(text_params_filename, unscaled_params[-num_text_params:])

    subprocess.run(["bash", "-c", get_run_command(num_procs, evaluate_gradient)])



class OptimizationIterator():
    def __init__(self, objective_args):
        self.objective_fun_and_grad = (
            lambda x: self.evaluate_objective_and_gradient(
                x, *objective_args
            )
        )

        self._num_calls = 0
        self.reset_iteration_history_variables()

        self.history = {}
        self.history["first iterate"] = []
        self.history["first objective"] = []
        self.history["first gradient"] = []
        self.history["last iterate"] = []
        self.history["last objective"] = []
        self.history["last gradient"] = []
        self.history["num_calls"] = []


    def reset_iteration_history_variables(self):
        self._first_iterate = None
        self._first_objective = None
        self._first_gradient = None
        self._last_iterate = None
        self._last_objective = None
        self._last_gradient = None


    def evaluate_objective_and_gradient(self,
        params,
        scales, param_names, block_indices,
        input_yaml, num_procs,
        num_text_params, text_params_filename,
    ):

        self._num_calls += 1
        self.reset_iteration_history_variables()

        run_objective_binary(params,
            scales, param_names, block_indices,
            input_yaml, num_procs,
            num_text_params, text_params_filename,
            evaluate_gradient=True
        )

        obj = np.loadtxt("objective_value.txt")

        unscaled_params = transform_parameters(params, scales,
            transform_from_canonical=True)
        grad = grad_transform(np.loadtxt("objective_gradient.txt"),
            unscaled_params, scales)

        if self._num_calls == 1:
            self._first_iterate = unscaled_params.copy()
            self._first_objective = obj
            self._first_gradient = grad.copy()
        else:
            self._last_iterate = unscaled_params.copy()
            self._last_objective = obj
            self._last_gradient = grad.copy()

        return obj, grad


    def callback(self, x):
        self.history["first iterate"].append(self._first_iterate)
        self.history["first objective"].append(self._first_objective)
        self.history["first gradient"].append(self._first_gradient)

        self.history["last iterate"].append(self._last_iterate)
        self.history["last objective"].append(self._last_objective)
        self.history["last gradient"].append(self._last_gradient)

        self.history["num_calls"].append(self._num_calls)
        self._num_calls = 0
        self.reset_iteration_history_variables
