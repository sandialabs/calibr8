import numpy as np
import pickle
import yaml

from scipy.optimize import minimize

from calibr8.util.driver_support import (
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
    Builds local bounds as intersection of:
      - trust box centered at xc with radius Delta (scalar or vector)
      - global_bounds (lb/ub can be float or None)
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
            eps = 1e-12
            mid = xci
            if lb_g is not None and mid < lb_g:
                mid = lb_g
            if ub_g is not None and mid > ub_g:
                mid = ub_g
            lb, ub = mid - eps, mid + eps
        bounds.append((lb, ub))
    return bounds


def binding_mask(x, bounds, tol=1e-6):
    """True where x is within tol of either bound (ignores None bounds)."""
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


def adaptive_lbfgsb(
    x0,
    global_bounds,
    objective_args,
    lbfgsb_options,
    history_pkl="adaptive_history.pkl",
    # trust box (can pass scalars or vectors)
    Delta0=1.0,
    Delta_min=1e-4,
    Delta_max=1.0,
    shrink=0.5,
    grow=1.5,
    bind_tol=1e-6,
    grow_on_binding=True,
    # burst length
    burst_init=10,
    burst_min=5,
    burst_max=100,
    burst_grow=2.0,
    reset_burst_on_fail=True,
    # improvement criterion for burst ramping
    rel_impr_tol=1e-5,
    max_restarts=50
):
    x0 = project_to_bounds(x0, global_bounds)
    n = len(x0)

    # vectorize Delta settings
    if np.isscalar(Delta0):
        Delta = np.full(n, float(Delta0))
    else:
        Delta = np.asarray(Delta0, dtype=float).copy()

    Delta_min_vec = np.full(n, float(Delta_min)) if np.isscalar(Delta_min) else np.asarray(Delta_min, dtype=float)
    Delta_max_vec = np.full(n, float(Delta_max)) if np.isscalar(Delta_max) else np.asarray(Delta_max, dtype=float)

    xc = x0.copy()
    cur_burst = int(burst_init)

    overall = {
        "global_bounds": global_bounds,
        "settings": {
            "Delta0": Delta0, "Delta_min": Delta_min, "Delta_max": Delta_max,
            "shrink": shrink, "grow": grow,
            "burst_init": burst_init, "burst_min": burst_min, "burst_max": burst_max,
            "burst_grow": burst_grow,
            "rel_impr_tol": rel_impr_tol,
            "grow_on_binding": grow_on_binding, "bind_tol": bind_tol
        },
        "restarts": [],
        "best_overall": {"x": xc.copy(), "f": np.inf}
    }

    for r in range(max_restarts):
        local_bounds = make_trust_bounds(xc, Delta, global_bounds)
        Delta_prev = Delta.copy()  # <-- track for treadmill stop

        opt_it = OptimizationIterator(objective_args)
        opts = dict(lbfgsb_options)
        opts["maxiter"] = cur_burst

        f_start = overall["best_overall"]["f"]

        res = minimize(
            fun=opt_it.objective_fun_and_grad,
            x0=xc,
            method="L-BFGS-B",
            jac=True,
            bounds=local_bounds,
            options=opts,
            callback=opt_it.callback
        )

        # keep last inner result for continuity
        with open("minimize_results.pkl", "wb") as file:
            pickle.dump(res, file)

        ch = opt_it.history.get("call_history", [])
        failed = any_failures(ch)
        best = best_success(ch)

        if best is not None:
            x_next = np.array(best["x_canonical"], copy=True)
            f_next = float(best["objective"])
        else:
            x_next = xc.copy()
            f_next = np.inf

        if np.isfinite(f_next) and f_next < overall["best_overall"]["f"]:
            overall["best_overall"] = {"x": x_next.copy(), "f": f_next}

        improved = np.isfinite(f_next) and np.isfinite(f_start) and (
            (f_start - f_next) / max(abs(f_start), 1.0) > rel_impr_tol
        )
        if (not np.isfinite(f_start)) and np.isfinite(f_next):
            improved = True

        bind = binding_mask(x_next, local_bounds, tol=bind_tol)

        if (not failed) and (not improved) and (not np.any(bind)):
            break

        n_fail = int(sum(1 for c in ch if not c.get("success", True)))
        n_call = int(len(ch))

        # did the inner solve actually do anything?
        did_work = (getattr(res, "nit", 0) > 0) or (n_call > 1)

        restart_record = {
            "restart": r,
            "xc_start": xc.copy(),
            "Delta_start": Delta.copy(),               # vector
            "burst_maxiter_used": int(cur_burst),
            "local_bounds": local_bounds,
            "res": res,
            "num_calls": n_call,
            "num_failures": n_fail,
            "any_failures": failed,
            "improved": bool(improved),
            "did_work": bool(did_work),
            "f_start_best_overall": f_start,
            "best_success": best,
            "x_next": x_next.copy(),
            "f_next": f_next,
            "binding_mask": bind,
            "history": opt_it.history
        }
        overall["restarts"].append(restart_record)

        with open(history_pkl, "wb") as f:
            pickle.dump(overall, f)

        # ---- Update burst length ----
        if failed and reset_burst_on_fail:
            cur_burst = int(burst_min)
        else:
            if (not failed) and improved:
                cur_burst = min(int(np.ceil(cur_burst * burst_grow)), int(burst_max))
            elif failed:
                cur_burst = max(int(np.floor(cur_burst / burst_grow)), int(burst_min))
            else:
                cur_burst = max(int(np.floor(cur_burst / burst_grow)), int(burst_min))

        # ---- Update Delta and center ----
        if failed:
            # robust: shrink all components
            Delta = np.maximum(Delta * shrink, Delta_min_vec)
            xc = x_next.copy()
        else:
            xc = x_next.copy()
            # robust: grow only binding components, but only if inner solve took steps
            if did_work and grow_on_binding and np.any(bind):
                grow_vec = np.where(bind, float(grow), 1.0)
                Delta = np.minimum(Delta * grow_vec, Delta_max_vec)

        # Stop "nit=0 treadmill": if no failure, no improvement, no work, and Delta didn't change
        delta_changed = np.any(np.abs(Delta - Delta_prev) > 0.0)
        if (not failed) and (not improved) and (not did_work) and (not delta_changed):
            break

        # termination: stable and all radii tiny
        if (not failed) and np.all(Delta <= Delta_min_vec):
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
    args = parser.parse_args()

    input_files = args.input_files

    num_procs = args.num_procs
    output_file = args.output_file
    use_srun = args.use_srun
    run_objective_only = args.run_objective_only

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
        "barrier_tolerance": gradient_tol,
        "maxls": max_ls_evals,
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
        }
        opt_iterator = OptimizationIterator(objective_args)
        res = minimize(
            fun=opt_iterator.objective_fun_and_grad,
            x0=opt_init_params,
            method="trust-constr",
            jac=True,
            bounds=opt_bounds,
            options=tr_opts,
            callback=opt_iterator.callback_trust_constr
        )
    else:
        if args.adaptive:
            overall = adaptive_lbfgsb(
                x0=opt_init_params,
                global_bounds=opt_bounds,
                objective_args=objective_args,
                lbfgsb_options=l_bfgs_b_opts,
                history_pkl="adaptive_history.pkl",
                # ---- suggested defaults (tune) ----
                Delta0=1.0,          # scalar => vectorized internally
                Delta_min=1e-4,
                Delta_max=1.0,
                shrink=0.5,
                grow=1.5,
                bind_tol=1e-6,
                grow_on_binding=True,
                burst_init=10,
                burst_min=5,
                burst_max=100,
                burst_grow=2.0,
                reset_burst_on_fail=True,
                rel_impr_tol=1e-5,
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

        # non-adaptive single run
        opt_iterator = OptimizationIterator(objective_args)
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
