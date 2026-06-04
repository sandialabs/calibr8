import numpy as np
import pickle
import yaml

from scipy.optimize import minimize

from calibr8.util.driver_support import (
    evaluate_objective_and_gradient,
    evaluate_objective_or_gradient,
    OptimizationIterator
)

from calibr8.util.input_file_io import (
    cleanup_files,
    get_opt_options,
    setup_opt_parameters,
    setup_text_parameters,
    standard_parser,
    write_output_file
)


def project_to_bounds(x, bounds):
    x = np.array(x, dtype=float, copy=True)
    for i, (lb, ub) in enumerate(bounds):
        if lb is not None:
            x[i] = max(x[i], lb)
        if ub is not None:
            x[i] = min(x[i], ub)
    return x


def make_trust_bounds(xc, Delta, global_bounds):
    """
    Construct the local trust-box bounds used by adaptive L-BFGS-B.

    The adaptive optimizer uses:
      1. wide global optimization bounds, and
      2. a smaller local trust box centered at the current accepted point xc

    The actual bounds passed to SciPy L-BFGS-B are the intersection of:
      - [xc - Delta, xc + Delta]
      - global_bounds

    Parameters
    ----------
    xc : array_like
        Current accepted center in canonical parameter space.
    Delta : float or array_like
        Trust-box radius. If scalar, the same radius is used for all
        parameters. If vector, each parameter has its own radius.
    global_bounds : sequence
        Global parameter bounds in canonical space.

    Returns
    -------
    bounds : list of (lb, ub)
        Local bounds for the next short L-BFGS-B burst.
    """
    xc = np.asarray(xc, dtype=float)
    n = xc.size
    Delta = np.full(n, float(Delta)) if np.isscalar(Delta) else np.asarray(Delta, dtype=float)

    if Delta.size != n:
        raise ValueError("Delta must be scalar or length-n vector")

    bounds = []
    for (lb_g, ub_g), xci, di in zip(global_bounds, xc, Delta):
        lb = xci - di
        ub = xci + di

        if lb_g is not None:
            lb = max(lb, float(lb_g))
        if ub_g is not None:
            ub = min(ub, float(ub_g))

        if not (lb < ub):
            eps = 1.0e-12
            mid = xci
            if lb_g is not None and mid < lb_g:
                mid = lb_g
            if ub_g is not None and mid > ub_g:
                mid = ub_g
            lb, ub = mid - eps, mid + eps

        bounds.append((lb, ub))

    return bounds


def binding_mask(x, bounds, tol=1.0e-6):
    """
    Identify which parameters are close to the current local trust-box boundary.

    This is used by the adaptive optimizer to decide whether a successful step
    "pressed" against the current trust box. If so, and if the step quality is
    good, the trust radius for those components may be enlarged.

    Parameters
    ----------
    x : array_like
        Point to test.
    bounds : list of (lb, ub)
        Local bounds for the current burst.
    tol : float
        Distance tolerance for declaring a coordinate "binding".

    Returns
    -------
    mask : ndarray[bool]
        True where x is within tol of either local bound.
    """
    x = np.asarray(x, dtype=float)
    mask = np.zeros_like(x, dtype=bool)

    for i, (lb, ub) in enumerate(bounds):
        if lb is not None and abs(x[i] - lb) <= tol:
            mask[i] = True
        if ub is not None and abs(x[i] - ub) <= tol:
            mask[i] = True

    return mask


def any_failures(call_history):
    return any(not c.get("success", True) for c in call_history)


def best_success(call_history):
    best = None
    for c in call_history:
        if c.get("success", False) and np.isfinite(c.get("objective", np.inf)):
            if best is None or c["objective"] < best["objective"]:
                best = c
    return best


def get_grad_from_call(call_record):
    g = call_record.get("grad", None)
    if g is None:
        return None
    g = np.asarray(g, dtype=float)
    if np.all(np.isfinite(g)):
        return g
    return None


def adaptive_lbfgsb(
    x0,
    global_bounds,
    objective_args,
    lbfgsb_options,
    history_pkl="adaptive_history.pkl",
    failure_mode="penalty_inward",
    failure_penalty=1.0e12,
    Delta0=0.2,
    Delta_min=1.0e-4,
    Delta_max=1.0,
    shrink_fail=0.25,
    shrink_bad=0.5,
    grow_good=1.5,
    bind_tol=1.0e-6,
    grow_on_binding=True,
    burst_init=5,
    burst_min=3,
    burst_max=20,
    burst_grow=1.5,
    burst_shrink=0.75,
    rel_impr_tol=1.0e-8,
    eta_accept=0.1,
    eta_keep=0.25,
    eta_grow=0.75,
    pred_floor=1.0e-12,
    max_restarts=50
):
    """
    Adaptive trust-box L-BFGS-B for simulation-based optimization.

    This routine is designed for problems where:
      - global parameter bounds should remain wide, but
      - large optimizer steps can move into "bad" parameter regions that cause
        the simulation to fail.

    The method keeps a current accepted center xc and repeatedly runs a short
    L-BFGS-B solve inside a local trust box around xc. After each burst, the
    trial result is accepted or rejected and the trust-box radius Delta is
    updated.

    Core idea
    ---------
    Global bounds stay wide so parameters are not artificially pinned. However,
    each inner L-BFGS-B solve only sees a much smaller local box:
        [xc - Delta, xc + Delta] intersect global_bounds

    This prevents the optimizer from taking the large initial steps that often
    occur with standard L-BFGS-B.

    Parameters
    ----------
    x0 : array_like
        Initial parameter vector in canonical coordinates. Must correspond to
        a successful simulation, because adaptive optimization needs a valid
        starting center.

    global_bounds : sequence
        Global optimization bounds in canonical coordinates.

    objective_args : tuple
        Arguments required by the simulator-backed objective/gradient evaluator.

    lbfgsb_options : dict
        Options passed into SciPy's L-BFGS-B for each short inner burst.

    history_pkl : str
        Output pickle file that stores the full adaptive restart history.

    failure_mode : {"penalty_inward", "repeat_last"}
        How the inner objective wrapper handles failed simulator calls.

        "penalty_inward":
            Return a finite penalty objective and a small inward-pointing fake
            gradient. This is the recommended default.

        "repeat_last":
            On failure, return the previous successful objective and gradient.
            This is closer to the older behavior and may be useful for
            comparison, though it is less principled.

    failure_penalty : float
        Objective value returned on failed simulation calls when using penalty
        handling. This should be finite, but much worse than any valid objective
        value expected in practice.

    Delta0 : float or array_like
        Initial trust-box radius. If scalar, the same radius is used for all
        parameters. If vector, each parameter has its own starting radius.

        Larger values allow more aggressive early motion.
        Smaller values make the method safer but more conservative.

    Delta_min : float or array_like
        Minimum trust-box radius. Once radii shrink to this scale, the adaptive
        search is effectively saying that it no longer trusts movement below
        this length scale.

    Delta_max : float or array_like
        Maximum trust-box radius. Prevents the local box from growing too large
        even after several successful steps.

    shrink_fail : float
        Multiplicative shrink factor applied to Delta when a burst contains
        failures and the trial is rejected.

        Example: 0.25 means "cut trust radius to 25% of its previous size".

    shrink_bad : float
        Multiplicative shrink factor applied to Delta when a burst succeeds but
        the resulting step quality is weak.

        This is less severe than shrink_fail.

    grow_good : float
        Multiplicative growth factor applied to trust radii when an accepted
        step is good and indicates the current trust box may be too small.

    bind_tol : float
        Numerical tolerance used to decide whether a coordinate is near the
        local trust-box boundary.

    grow_on_binding : bool
        If True, trust radii are only enlarged in components that are close to
        the local trust-box boundary. This helps avoid unnecessary growth in
        directions that were not actively used.

    burst_init : int
        Initial number of L-BFGS-B iterations allowed in each inner burst.

    burst_min : int
        Minimum burst length.

    burst_max : int
        Maximum burst length.

    burst_grow : float
        Multiplicative factor used to increase burst length after a successful,
        productive burst.

    burst_shrink : float
        Multiplicative factor used to reduce burst length after failures or
        poor progress.

    rel_impr_tol : float
        Relative-improvement threshold used to classify whether a burst made
        meaningful progress, as opposed to only tiny numerical changes.

    eta_accept : float
        Minimum ratio of actual reduction to predicted reduction required to
        accept a trial step.

        The ratio is:
            rho = (fc - ft) / pred

        where fc is the current center objective, ft is the trial objective,
        and pred is an approximate first-order predicted reduction:
            pred = max(-gc^T (xt - xc), pred_floor)

    eta_keep : float
        If a trial is accepted but rho is below this threshold, the trust box is
        still shrunk because the model agreement was weak.

    eta_grow : float
        If a trial is accepted and rho exceeds this threshold, and the step is
        boundary-active, the trust box may be enlarged.

    pred_floor : float
        Small positive floor applied to the predicted reduction denominator to
        avoid divide-by-zero or near-zero instabilities.

    max_restarts : int
        Maximum number of outer adaptive iterations (trust-box updates).

    Returns
    -------
    overall : dict
        Complete adaptive optimization history, including:
          - settings
          - restart records
          - best overall point found

    Notes
    -----
    This method is especially appropriate when failures correspond to invalid or
    numerically unstable parameter vectors, rather than noisy objective values.
    In that setting, failed points should be interpreted as rejected trial
    regions, and the main corrective action should be to shrink the trust box.
    """
    x0 = project_to_bounds(x0, global_bounds)
    n = len(x0)

    if np.isscalar(Delta0):
        Delta = np.full(n, float(Delta0))
    else:
        Delta = np.asarray(Delta0, dtype=float).copy()

    Delta_min_vec = np.full(n, float(Delta_min)) if np.isscalar(Delta_min) else np.asarray(Delta_min, dtype=float)
    Delta_max_vec = np.full(n, float(Delta_max)) if np.isscalar(Delta_max) else np.asarray(Delta_max, dtype=float)

    fc, gc, success0 = evaluate_objective_and_gradient(
        x0, *objective_args, failure_penalty=failure_penalty
    )
    if not success0:
        raise RuntimeError("Initial parameter set failed objective/gradient evaluation.")

    xc = x0.copy()
    cur_burst = int(burst_init)

    overall = {
        "optimizer": "adaptive_lbfgsb",
        "global_bounds": global_bounds,
        "settings": {
            "failure_mode": failure_mode,
            "failure_penalty": failure_penalty,
            "Delta0": Delta0,
            "Delta_min": Delta_min,
            "Delta_max": Delta_max,
            "shrink_fail": shrink_fail,
            "shrink_bad": shrink_bad,
            "grow_good": grow_good,
            "bind_tol": bind_tol,
            "grow_on_binding": grow_on_binding,
            "burst_init": burst_init,
            "burst_min": burst_min,
            "burst_max": burst_max,
            "burst_grow": burst_grow,
            "burst_shrink": burst_shrink,
            "rel_impr_tol": rel_impr_tol,
            "eta_accept": eta_accept,
            "eta_keep": eta_keep,
            "eta_grow": eta_grow,
            "pred_floor": pred_floor,
            "max_restarts": max_restarts
        },
        "initial_center": {
            "x": xc.copy(),
            "f": float(fc),
            "grad": np.asarray(gc, dtype=float).copy()
        },
        "restarts": [],
        "best_overall": {"x": xc.copy(), "f": float(fc)}
    }

    for r in range(max_restarts):
        local_bounds = make_trust_bounds(xc, Delta, global_bounds)
        Delta_prev = Delta.copy()

        opt_it = OptimizationIterator(
            objective_args,
            failure_penalty=failure_penalty,
            failure_mode=failure_mode
        )

        opts = dict(lbfgsb_options)
        opts["maxiter"] = cur_burst

        res = minimize(
            fun=opt_it.objective_fun_and_grad,
            x0=xc,
            method="L-BFGS-B",
            jac=True,
            bounds=local_bounds,
            options=opts,
            callback=opt_it.callback
        )

        with open("minimize_results.pkl", "wb") as file:
            pickle.dump(res, file)

        ch = opt_it.history.get("call_history", [])
        failed = any_failures(ch)
        best = best_success(ch)

        accepted = False
        improved = False
        xt = xc.copy()
        ft = np.inf
        gt = None
        bind = np.zeros(n, dtype=bool)
        ared = -np.inf
        pred = np.nan
        rho = -np.inf
        step_norm = 0.0

        if best is not None:
            xt = np.array(best["x_canonical"], copy=True)
            ft = float(best["objective"])
            gt = get_grad_from_call(best)
            bind = binding_mask(xt, local_bounds, tol=bind_tol)

            s = xt - xc
            step_norm = float(np.linalg.norm(s))
            ared = float(fc - ft)
            pred = max(float(-np.dot(gc, s)), float(pred_floor))
            rho = ared / pred
            rel_impr = ared / max(abs(fc), 1.0)

            improved = np.isfinite(ft) and (ared > 0.0) and (rel_impr > rel_impr_tol)
            accepted = np.isfinite(ft) and (ared > 0.0) and np.isfinite(rho) and (rho >= eta_accept)

        n_fail = int(sum(1 for c in ch if not c.get("success", True)))
        n_call = int(len(ch))
        did_work = (getattr(res, "nit", 0) > 0) or (n_call > 1)

        restart_record = {
            "restart": r,
            "accepted": bool(accepted),
            "improved": bool(improved),
            "xc_start": xc.copy(),
            "fc_start": float(fc),
            "gc_start": np.asarray(gc, dtype=float).copy(),
            "x_trial": xt.copy(),
            "f_trial": float(ft) if np.isfinite(ft) else np.inf,
            "Delta_start": Delta.copy(),
            "burst_maxiter_used": int(cur_burst),
            "local_bounds": local_bounds,
            "res": res,
            "num_calls": n_call,
            "num_failures": n_fail,
            "any_failures": failed,
            "did_work": bool(did_work),
            "ared": float(ared) if np.isfinite(ared) else ared,
            "pred": float(pred) if np.isfinite(pred) else pred,
            "rho": float(rho) if np.isfinite(rho) else rho,
            "step_norm": float(step_norm),
            "binding_mask": bind,
            "best_success": best,
            "history": opt_it.history
        }
        overall["restarts"].append(restart_record)

        if accepted:
            xc = xt.copy()
            fc = float(ft)

            if gt is None:
                fc_eval, gc_eval, success_eval = evaluate_objective_and_gradient(
                    xc, *objective_args, failure_penalty=failure_penalty
                )
                if not success_eval:
                    xc = restart_record["xc_start"].copy()
                    fc = float(restart_record["fc_start"])
                    gc = restart_record["gc_start"].copy()
                    Delta = np.maximum(Delta * shrink_fail, Delta_min_vec)
                    cur_burst = max(int(np.floor(cur_burst * burst_shrink)), int(burst_min))
                else:
                    fc = float(fc_eval)
                    gc = np.asarray(gc_eval, dtype=float).copy()
            else:
                gc = gt.copy()

            if fc < overall["best_overall"]["f"]:
                overall["best_overall"] = {"x": xc.copy(), "f": float(fc)}

            if np.isfinite(rho) and (rho >= eta_grow) and did_work and grow_on_binding and np.any(bind):
                grow_vec = np.where(bind, float(grow_good), 1.0)
                Delta = np.minimum(Delta * grow_vec, Delta_max_vec)
            elif np.isfinite(rho) and (rho < eta_keep):
                Delta = np.maximum(Delta * shrink_bad, Delta_min_vec)

            if (not failed) and improved:
                cur_burst = min(int(np.ceil(cur_burst * burst_grow)), int(burst_max))
            elif failed:
                cur_burst = max(int(np.floor(cur_burst * burst_shrink)), int(burst_min))
        else:
            Delta = np.maximum(Delta * (shrink_fail if failed else shrink_bad), Delta_min_vec)
            cur_burst = max(int(np.floor(cur_burst * burst_shrink)), int(burst_min))

        with open(history_pkl, "wb") as f:
            pickle.dump(overall, f)

        delta_changed = np.any(np.abs(Delta - Delta_prev) > 0.0)

        if np.all(Delta <= Delta_min_vec) and (not improved):
            break

        if (not accepted) and (not did_work) and (not delta_changed):
            break

    return overall


def main():
    parser = standard_parser()
    parser.add_argument("-o", "--output_file", type=str,
        default="calibrated_params.txt",
        help="output file that contains the calibrated parameters")
    parser.add_argument("--run_objective_only", action="store_true",
        help="flag to indicate whether to run objective only (no optimization)")
    parser.add_argument("--adaptive", action="store_true",
        help="use adaptive restart trust-box L-BFGS-B loop")
    parser.add_argument("--trust_region", action="store_true",
        help="run the trust region optimizer")
    parser.add_argument("--failure_mode", type=str,
        default="penalty_inward",
        choices=["penalty_inward", "repeat_last"],
        help="simple failure handling mode for standard/adaptive L-BFGS-B")
    args = parser.parse_args()

    input_files = args.input_files

    num_procs = args.num_procs
    output_file = args.output_file
    use_srun = args.use_srun
    run_objective_only = args.run_objective_only
    failure_mode = args.failure_mode

    text_parameters_initial_values_file = args.text_parameters_initial_values_file
    text_parameters_scales_file = args.text_parameters_scales_file
    text_parameters_opt_values_filename = args.text_parameters_opt_values_filename

    text_params_data = setup_text_parameters(
        text_parameters_initial_values_file,
        text_parameters_scales_file,
        text_parameters_opt_values_filename
    )
    num_text_params = len(text_params_data[0])

    input_yamls = [
        yaml.safe_load(open(input_file, "r")) for input_file in input_files
    ]

    opt_param_names, opt_param_scales, opt_param_block_indices, \
        opt_init_params, opt_bounds = \
        setup_opt_parameters(input_yamls[0], text_params_data)

    num_iters, gradient_tol, max_ls_evals = get_opt_options(input_yamls[0])

    l_bfgs_b_opts = {
        "maxiter": num_iters,
        "gtol": gradient_tol,
        "maxls": max_ls_evals,
        "ftol": 10.0 * np.finfo(float).eps
    }

    objective_args = (
        opt_param_scales, opt_param_names, opt_param_block_indices,
        input_yamls, num_procs, use_srun,
        num_text_params, text_parameters_opt_values_filename
    )

    if run_objective_only:
        evaluate_gradient = False
        evaluate_objective_or_gradient(
            opt_init_params, *objective_args, evaluate_gradient
        )
        cleanup_files(evaluate_gradient)
        return

    evaluate_gradient = True

    if args.trust_region:
        tr_opts = {
            "maxiter": num_iters,
            "gtol": gradient_tol,
            "verbose": 3,
            "barrier_tolerance": gradient_tol,
        }
        opt_iterator = OptimizationIterator(
            objective_args,
            failure_mode=failure_mode
        )
        res = minimize(
            fun=opt_iterator.objective_fun_and_grad,
            x0=opt_init_params,
            method="trust-constr",
            jac=True,
            bounds=opt_bounds,
            options=tr_opts,
            callback=opt_iterator.callback_trust_constr
        )

        with open("minimize_results.pkl", "wb") as file:
            pickle.dump(res, file)

        write_output_file(
            res.x, opt_param_scales, opt_param_names,
            output_file
        )
        cleanup_files(evaluate_gradient)
        return

    if args.adaptive:
        overall = adaptive_lbfgsb(
            x0=opt_init_params,
            global_bounds=opt_bounds,
            objective_args=objective_args,
            lbfgsb_options=l_bfgs_b_opts,
            history_pkl="adaptive_history.pkl",
            failure_mode=failure_mode,
            failure_penalty=1.0e12,
            Delta0=0.2,
            Delta_min=1.0e-4,
            Delta_max=1.0,
            shrink_fail=0.25,
            shrink_bad=0.5,
            grow_good=1.5,
            bind_tol=1.0e-6,
            grow_on_binding=True,
            burst_init=5,
            burst_min=3,
            burst_max=20,
            burst_grow=1.5,
            burst_shrink=0.75,
            rel_impr_tol=1.0e-8,
            eta_accept=0.1,
            eta_keep=0.25,
            eta_grow=0.75,
            pred_floor=1.0e-12,
            max_restarts=50
        )

        x_best = overall["best_overall"]["x"]
        with open("adaptive_history.pkl", "wb") as f:
            pickle.dump(overall, f)

        write_output_file(
            x_best, opt_param_scales, opt_param_names,
            output_file
        )
        cleanup_files(evaluate_gradient)
        return

    opt_iterator = OptimizationIterator(
        objective_args,
        failure_mode=failure_mode
    )
    res = minimize(
        fun=opt_iterator.objective_fun_and_grad,
        x0=opt_init_params,
        method="L-BFGS-B",
        jac=True,
        bounds=opt_bounds,
        options=l_bfgs_b_opts,
        callback=opt_iterator.callback
    )

    with open("minimize_results.pkl", "wb") as file:
        pickle.dump(res, file)

    write_output_file(
        res.x, opt_param_scales, opt_param_names,
        output_file
    )

    cleanup_files(evaluate_gradient)


if __name__ == "__main__":
    main()
