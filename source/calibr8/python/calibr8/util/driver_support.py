import numpy as np

import pickle
import subprocess
import yaml

from concurrent.futures import ThreadPoolExecutor

from calibr8.util.input_file_io import (
    IndentDumper,
    update_yaml_input_file_parameters
)
from calibr8.util.parameter_transforms import (
    grad_transform,
    transform_parameters
)


def subprocess_with_output(command, output_file):
    try:
        with open(output_file, "w") as file:
            subprocess.run(command, stdout=file, stderr=file, check=True)
        return 0
    except subprocess.CalledProcessError:
        return 1


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


def run_objective_binaries(
    params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    unscaled_params = transform_parameters(
        params, scales, transform_from_canonical=True
    )

    num_params = len(params)
    num_input_file_params = num_params - num_text_params

    if text_params_filename is not None:
        np.savetxt(text_params_filename, unscaled_params[-num_text_params:])

    run_commands = []
    run_output_files = []

    for idx, input_yaml in enumerate(input_yamls):

        if num_input_file_params > 0:
            update_yaml_input_file_parameters(
                input_yaml,
                param_names[:num_input_file_params],
                unscaled_params[:num_input_file_params],
                block_indices[:num_input_file_params]
            )

        run_input_file = f"run_{idx}.yaml"
        run_output_file = f"run_{idx}.out"
        with open(run_input_file, "w") as file:
            yaml.dump(
                input_yaml, file,
                default_flow_style=False, sort_keys=False,
                Dumper=IndentDumper
            )

        run_commands.append(get_run_command(
            num_procs, use_srun, evaluate_gradient, idx
        ))
        run_output_files.append(run_output_file)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            subprocess_with_output, run_commands, run_output_files
        ))

    if np.sum(results) == 0:
        return True
    else:
        print("Simulation failure!")
        return False


def evaluate_objective_and_gradient(
    params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
    failure_penalty=1.0e12
):
    success = run_objective_binaries(
        params,
        scales, param_names, block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_params_filename,
        evaluate_gradient=True
    )

    if not success:
        return float(failure_penalty), None, False

    obj = 0.0
    unscaled_params = transform_parameters(
        params, scales, transform_from_canonical=True
    )
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

    return float(obj), np.asarray(grad, dtype=float), True


def evaluate_objective_or_gradient(
    params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    run_objective_binaries(
        params,
        scales, param_names, block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_params_filename,
        evaluate_gradient
    )

    if not evaluate_gradient:
        obj = 0.0
        for idx in range(len(input_yamls)):
            obj_file = f"objective_value_{idx}.txt"
            obj += np.loadtxt(obj_file)
        return float(obj)

    unscaled_params = transform_parameters(
        params, scales, transform_from_canonical=True
    )
    unscaled_grad = np.zeros(len(unscaled_params))
    for idx in range(len(input_yamls)):
        unscaled_grad_file = f"objective_gradient_{idx}.txt"
        unscaled_grad += np.loadtxt(unscaled_grad_file)

    grad = grad_transform(
        unscaled_grad,
        unscaled_params, scales
    )
    return grad


class OptimizationIterator:
    """
    Objective/gradient wrapper for SciPy optimizers.

    failure_mode options:
      - "penalty_inward": on failure, return a finite penalty objective and a
        small inward-pointing fake gradient
      - "repeat_last": on failure, return the previous successful objective and
        gradient; if no previous successful eval exists, fall back to penalty_inward
    """
    def __init__(
        self,
        objective_args,
        failure_penalty=1.0e12,
        failure_mode="penalty_inward",
        x_match_tol=1.0e-14,
        grad_norm_window=25,
        fake_grad_fallback_norm=1.0,
        fake_grad_scale=1.0e-3,
        fake_grad_cap=1.0,
        eps=1.0e-12
    ):
        self.objective_args = objective_args
        self.failure_penalty = float(failure_penalty)
        self.failure_mode = str(failure_mode)
        self.x_match_tol = float(x_match_tol)

        self.grad_norm_window = int(grad_norm_window)
        self.fake_grad_fallback_norm = float(fake_grad_fallback_norm)
        self.fake_grad_scale = float(fake_grad_scale)
        self.fake_grad_cap = float(fake_grad_cap)
        self.eps = float(eps)

        self._last_x = None
        self._last_obj = None
        self._last_grad = None
        self._last_success = False

        self._last_success_x = None
        self._last_success_obj = None
        self._last_success_grad = None
        self._successful_grad_norms = []

        self.reset_history()

        self.objective_fun_and_grad = (
            lambda x: self._objective_fun_and_grad(x)
        )

    def reset_history(self):
        self.history = {
            "accepted_x_canonical": [],
            "accepted_x_unscaled": [],
            "accepted_obj": [],
            "accepted_grad": [],
            "accepted_grad_norm": [],
            "accepted_obj_is_known": [],
            "call_history": []
        }

    def summarize_run(self):
        ch = self.history.get("call_history", [])
        any_fail = any(not c.get("success", True) for c in ch)
        best = None
        for c in ch:
            if c.get("success", False) and np.isfinite(c.get("objective", np.inf)):
                if best is None or c["objective"] < best["objective"]:
                    best = c
        return {"any_failures": any_fail, "best": best}

    def _robust_target_grad_norm(self):
        if len(self._successful_grad_norms) == 0:
            return self.fake_grad_fallback_norm
        return float(np.median(self._successful_grad_norms))

    def _make_inward_fake_grad(self, x):
        x = np.asarray(x, dtype=float)

        if self._last_success_x is None:
            direction = np.ones_like(x, dtype=float)
        else:
            direction = x - np.asarray(self._last_success_x, dtype=float)

        nrm = float(np.linalg.norm(direction))
        if (not np.isfinite(nrm)) or (nrm < self.eps):
            direction = np.ones_like(x, dtype=float)
            nrm = float(np.linalg.norm(direction))

        target = self._robust_target_grad_norm()
        fake_norm = min(self.fake_grad_scale * target, self.fake_grad_cap)
        fake_norm = max(fake_norm, self.eps)

        return (fake_norm / (nrm + self.eps)) * direction

    def _handle_failure(self, x):
        if self.failure_mode == "repeat_last":
            if self._last_success_obj is not None and self._last_success_grad is not None:
                return (
                    float(self._last_success_obj),
                    np.asarray(self._last_success_grad, dtype=float).copy(),
                    False,
                    "repeat_last"
                )

        grad = self._make_inward_fake_grad(x)
        return float(self.failure_penalty), grad, False, "penalty_inward"

    def _objective_fun_and_grad(self, x):
        x = np.array(x, copy=True)

        obj_true, grad_true, success = evaluate_objective_and_gradient(
            x, *self.objective_args,
            failure_penalty=self.failure_penalty
        )

        if success:
            obj = float(obj_true)
            grad = np.array(grad_true, copy=True)
            grad_norm = float(np.linalg.norm(grad))
            failure_response = None

            if np.isfinite(grad_norm):
                self._successful_grad_norms.append(grad_norm)
                if len(self._successful_grad_norms) > self.grad_norm_window:
                    self._successful_grad_norms.pop(0)

            self._last_success_x = x.copy()
            self._last_success_obj = float(obj)
            self._last_success_grad = grad.copy()
        else:
            obj, grad, _, failure_response = self._handle_failure(x)
            grad = np.asarray(grad, dtype=float)
            grad_norm = float(np.linalg.norm(grad))

        self._last_x = x.copy()
        self._last_obj = float(obj)
        self._last_grad = grad.copy()
        self._last_success = bool(success)

        self.history["call_history"].append({
            "x_canonical": self._last_x.copy(),
            "objective": self._last_obj,
            "grad": self._last_grad.copy(),
            "grad_norm": grad_norm,
            "success": self._last_success,
            "failure_response": failure_response
        })

        return obj, grad

    def _x_matches_last(self, xk):
        if self._last_x is None:
            return False
        return np.allclose(xk, self._last_x, atol=self.x_match_tol, rtol=0.0)

    def callback(self, xk, res=None):
        xk = np.array(xk, copy=True)

        self.history["accepted_x_canonical"].append(xk.copy())
        self.history["accepted_x_unscaled"].append(
            transform_parameters(
                xk, self.objective_args[0], transform_from_canonical=True
            )
        )

        if self._x_matches_last(xk):
            obj = float(self._last_obj)
            grad = self._last_grad.copy()
            grad_norm = float(np.linalg.norm(grad))
            known = True
        else:
            obj = np.nan
            grad = np.full_like(xk, np.nan, dtype=float)
            grad_norm = np.nan
            known = False

        self.history["accepted_obj"].append(obj)
        self.history["accepted_grad"].append(grad)
        self.history["accepted_grad_norm"].append(grad_norm)
        self.history["accepted_obj_is_known"].append(known)

        with open("optimization_history.pkl", "wb") as file:
            pickle.dump(self.history, file)

    def callback_trust_constr(self, xk, res=None):
        xk = np.array(xk, copy=True)

        self.history["accepted_x_canonical"].append(xk.copy())
        self.history["accepted_x_unscaled"].append(
            transform_parameters(
                xk, self.objective_args[0], transform_from_canonical=True
            )
        )

        obj = np.nan
        if res is not None and np.isfinite(getattr(res, "fun", np.nan)):
            obj = float(res.fun)
            obj_known = True
        elif self._x_matches_last(xk) and np.isfinite(self._last_obj):
            obj = float(self._last_obj)
            obj_known = True
        else:
            obj_known = False

        if self._x_matches_last(xk) and (self._last_grad is not None) and np.all(np.isfinite(self._last_grad)):
            grad = self._last_grad.copy()
            grad_known = True
        else:
            grad = np.full_like(xk, np.nan, dtype=float)
            grad_known = False

        grad_norm = float(np.linalg.norm(grad)) if grad_known else np.nan

        self.history["accepted_obj"].append(obj)
        self.history["accepted_grad"].append(grad)
        self.history["accepted_grad_norm"].append(grad_norm)
        self.history["accepted_obj_is_known"].append(bool(obj_known))

        tc = {
            "nit": int(getattr(res, "nit", -1)) if res is not None else -1,
            "optimality": float(getattr(res, "optimality", np.nan)) if res is not None else np.nan,
            "constr_violation": float(getattr(res, "constr_violation", np.nan)) if res is not None else np.nan,
        }
        self.history.setdefault("trust_constr", []).append(tc)

        with open("optimization_history.pkl", "wb") as file:
            pickle.dump(self.history, file)
