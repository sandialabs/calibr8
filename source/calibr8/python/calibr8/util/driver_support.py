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
    failure_penalty=1.0e30
):
    """
    Evaluate true objective and gradient by running the external FE objective.

    Returns (obj, grad, success_flag). If the FE run fails, returns
    (failure_penalty, None, False).
    """
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

    return float(obj), grad, True


def evaluate_objective_or_gradient(
    params,
    scales, param_names, block_indices,
    input_yamls, num_procs, use_srun,
    num_text_params, text_params_filename,
    evaluate_gradient
):
    """
    Backwards-compatible helper for --run_objective_only mode.
    """
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
    Wraps objective+gradient evaluation for SciPy L-BFGS-B and logs all calls.

    On FE failure: returns a large penalty objective and a *normalized fake
    gradient* (magnitude based on the median of recent successful gradient norms),
    pointing away from the failing point back toward the last successful point.

    Stores full gradient vectors at *accepted* iterates (when known) in:
      history["accepted_grad"] (canonical space)
    """
    def __init__(
        self,
        objective_args,
        failure_penalty=1.0e30,
        x_match_tol=1.0e-14,
        grad_norm_window=25,
        fake_grad_fallback_norm=1.0,
        eps=1.0e-12
    ):
        self.objective_args = objective_args
        self.failure_penalty = float(failure_penalty)
        self.x_match_tol = float(x_match_tol)

        self.grad_norm_window = int(grad_norm_window)
        self.fake_grad_fallback_norm = float(fake_grad_fallback_norm)
        self.eps = float(eps)

        self._last_x = None
        self._last_obj = None
        self._last_grad = None
        self._last_success = False

        self._last_success_x = None
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
            "accepted_grad": [],          # <-- full gradient vectors at accepted iters
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

    def _make_fake_grad(self, x):
        g0 = self._robust_target_grad_norm()

        x = np.asarray(x, dtype=float)
        if self._last_success_x is None:
            direction = x.copy()
        else:
            direction = x - np.asarray(self._last_success_x, dtype=float)

        nrm = float(np.linalg.norm(direction))
        if (not np.isfinite(nrm)) or (nrm < self.eps):
            direction = np.ones_like(x, dtype=float)
            nrm = float(np.linalg.norm(direction))

        return (g0 / (nrm + self.eps)) * direction

    def _objective_fun_and_grad(self, x):
        x = np.array(x, copy=True)

        obj, grad, success = evaluate_objective_and_gradient(
            x, *self.objective_args,
            failure_penalty=self.failure_penalty
        )

        if success:
            grad = np.array(grad, copy=True)
            grad_norm = float(np.linalg.norm(grad))
            if np.isfinite(grad_norm):
                self._successful_grad_norms.append(grad_norm)
                if len(self._successful_grad_norms) > self.grad_norm_window:
                    self._successful_grad_norms.pop(0)
            self._last_success_x = x.copy()
        else:
            grad = self._make_fake_grad(x)
            grad_norm = float(np.linalg.norm(grad))

        self._last_x = x.copy()
        self._last_obj = float(obj)
        self._last_grad = grad.copy()
        self._last_success = bool(success)

        self.history["call_history"].append({
            "x_canonical": self._last_x.copy(),
            "objective": self._last_obj,
            "grad_norm": grad_norm,
            "success": self._last_success
        })

        return obj, grad

    def _x_matches_last(self, xk):
        if self._last_x is None:
            return False
        return np.allclose(xk, self._last_x, atol=self.x_match_tol, rtol=0.0)

    def callback(self, xk, res=None):
        """
        Called by SciPy once per accepted iteration. We avoid extra FE runs, so
        we only log f and grad if the accepted xk matches the last evaluation point.
        """
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

        # Convenience: overwrite per-run history
        with open("optimization_history.pkl", "wb") as file:
            pickle.dump(self.history, file)

    def callback_trust_constr(self, xk, res=None):
        """
        Callback specialized for trust-constr.

        - Always logs accepted xk.
        - Logs accepted objective from res.fun (authoritative for trust-constr).
        - Logs gradient only if the last evaluation point matches xk (rare but possible),
          otherwise stores NaNs for grad but does NOT NaN the objective.
        """

        xk = np.array(xk, copy=True)

        self.history["accepted_x_canonical"].append(xk.copy())
        self.history["accepted_x_unscaled"].append(
            transform_parameters(
                xk, self.objective_args[0], transform_from_canonical=True
            )
        )

        # Objective: prefer res.fun for trust-constr
        obj = np.nan
        if res is not None and np.isfinite(getattr(res, "fun", np.nan)):
            obj = float(res.fun)
            obj_known = True
        elif self._x_matches_last(xk) and np.isfinite(self._last_obj):
            obj = float(self._last_obj)
            obj_known = True
        else:
            obj_known = False

        # Gradient: only known if last evaluation matches xk (unless you add a cache)
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

        # Replace your old boolean with something that reflects what is actually known:
        self.history["accepted_obj_is_known"].append(bool(obj_known))

        # Optional: store trust-constr diagnostics from res if present
        tc = {
            "nit": int(getattr(res, "nit", -1)) if res is not None else -1,
            "optimality": float(getattr(res, "optimality", np.nan)) if res is not None else np.nan,
            "constr_violation": float(getattr(res, "constr_violation", np.nan)) if res is not None else np.nan,
        }
        self.history.setdefault("trust_constr", []).append(tc)

        with open("optimization_history.pkl", "wb") as file:
            pickle.dump(self.history, file)
