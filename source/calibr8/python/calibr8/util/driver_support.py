import numpy as np

import pickle
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


def evaluate_objective_and_gradient(
    params,
    scales, param_names, block_indices,
    input_yaml, num_procs,
    num_text_params, text_params_filename,
):
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

    return obj, grad


def evaluate_objective_or_gradient(
    params,
    scales, param_names, block_indices,
    input_yaml, num_procs,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    run_objective_binary(params,
        scales, param_names, block_indices,
        input_yaml, num_procs,
        num_text_params, text_params_filename,
        evaluate_gradient
    )

    if not evaluate_gradient:
        return np.loadtxt("objective_value.txt")
    else:
        unscaled_params = transform_parameters(params, scales,
            transform_from_canonical=True)
        grad = grad_transform(np.loadtxt("objective_gradient.txt"),
            unscaled_params, scales)

    return grad


class OptimizationIterator():
    def __init__(self, objective_args):
        self.objective_fun_and_grad = (
            lambda x: self.evaluate_objective_and_gradient(
                x, *objective_args
            )
        )

        self._iterate = None
        self._objective = None
        self._gradient = None
        self._num_calls = 0

        self.history = {}
        self.history["iterate"] = []
        self.history["objective"] = []
        self.history["gradient"] = []
        self.history["num_calls"] = []


    def evaluate_objective_and_gradient(self,
        params,
        scales, param_names, block_indices,
        input_yaml, num_procs,
        num_text_params, text_params_filename,
    ):
        self._num_calls += 1

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
            self._iterate = unscaled_params.copy()
            self._objective = obj
            self._gradient = grad.copy()

        return obj, grad


    def callback(self, x):
        self.history["iterate"].append(self._iterate)
        self.history["objective"].append(self._objective)
        self.history["gradient"].append(self._gradient)
        self.history["num_calls"].append(self._num_calls)

        with open("optimization_history.pkl", "wb") as file:
            pickle.dump(self.history, file)

        self._num_calls = 0
