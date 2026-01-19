import numpy as np

import pickle
import subprocess
import yaml

from concurrent.futures import ProcessPoolExecutor

from calibr8.util.input_file_io import (
    IndentDumper,
    update_yaml_input_file_parameters
)
from calibr8.util.parameter_transforms import (
    grad_transform,
    transform_parameters
)


def get_run_command(
    num_procs, use_srun, evaluate_gradient=True, problem_idx=None
):
    if use_srun:
        cmd = ["srun", "--exact"]
    else:
        cmd = ["mpirun"]

    cmd += ["-n", f"{num_procs}", "objective"]

    if problem_idx is None:
        problem_str = ""
    else:
        problem_str = "_" + str(problem_idx)

    cmd += [f"run{problem_str}.yaml"]

    if evaluate_gradient:
        cmd += ["true"]
    else:
        cmd += ["false"]

    if problem_idx is not None:
        cmd += [f"{problem_idx}"]

    return cmd


def run_objective_binaries(params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    unscaled_params = transform_parameters(params, scales,
        transform_from_canonical=True)

    num_params = len(params)
    num_input_file_params = num_params - num_text_params

    if text_params_filename is not None:
        np.savetxt(text_params_filename, unscaled_params[-num_text_params:])

    run_commands = []

    for idx, input_yaml in enumerate(input_yamls):

        if num_input_file_params > 0:
            update_yaml_input_file_parameters(input_yaml,
                param_names[:num_input_file_params],
                unscaled_params[:num_input_file_params],
                block_indices[:num_input_file_params]
            )

        run_file = f"run_{idx}.yaml"
        with open(run_file, "w") as file:
            yaml.dump(input_yaml, file,
                default_flow_style=False, sort_keys=False,
                Dumper=IndentDumper
            )

        run_commands.append(get_run_command(
            num_procs, use_srun, evaluate_gradient, idx
        ))

    with ProcessPoolExecutor() as executor:
        executor.map(subprocess.run, run_commands)


def evaluate_objective_and_gradient(
    params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
):
    run_objective_binaries(params,
        scales, param_names, block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_params_filename,
        evaluate_gradient=True
    )

    obj = 0.

    unscaled_params = transform_parameters(params, scales,
        transform_from_canonical=True)
    unscaled_grad = np.zeros(len(unscaled_params))

    for idx in range(len(input_yamls)):
        obj_file = f"objective_value_{idx}.txt"
        obj += np.loadtxt(obj_file)

        unscaled_grad_file = f"objective_gradient_{idx}.txt"
        unscaled_grad += np.loadtxt(unscaled_grad_file)

    grad = grad_transform(
        unscaled_grad,
        unscaled_params, scales
    )

    return obj, grad


def evaluate_objective_or_gradient(
    params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    run_objective_binaries(params,
        scales, param_names, block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_params_filename,
        evaluate_gradient
    )

    if not evaluate_gradient:

        obj = 0.
        for idx in range(len(input_yamls)):
            obj_file = f"objective_value_{idx}.txt"
            obj += np.loadtxt(obj_file)
        return obj

    else:

        unscaled_params = transform_parameters(params, scales,
            transform_from_canonical=True)
        unscaled_grad = np.zeros(len(unscaled_params))
        for idx in range(len(input_yamls)):
            unscaled_grad_file = f"objective_gradient_{idx}.txt"
            unscaled_grad += np.loadtxt(unscaled_grad_file)

        grad = grad_transform(
            unscaled_grad,
            unscaled_params, scales
        )

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
        input_yamls, num_procs, use_srun,
        num_text_params, text_params_filename,
    ):
        self._num_calls += 1

        obj, grad = evaluate_objective_and_gradient(
            params,
            scales, param_names, block_indices,
            input_yamls, num_procs, use_srun,
            num_text_params, text_params_filename,
        )

        unscaled_params = transform_parameters(params, scales,
            transform_from_canonical=True
        )

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
