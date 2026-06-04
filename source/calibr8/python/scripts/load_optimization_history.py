import os
import numpy as np
import pickle


def _count_failures(call_history):
    if not call_history:
        return 0, 0
    n_call = len(call_history)
    n_fail = sum(1 for c in call_history if not c.get("success", True))
    return int(n_fail), int(n_call)


def _first_success(call_history):
    if not call_history:
        return None
    for c in call_history:
        if c.get("success", False):
            return c
    return None


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ------------------------------------------------------------------
# Primary optimization-history file from OptimizationIterator
# ------------------------------------------------------------------
opt_file = "optimization_history.pkl"
if not os.path.exists(opt_file):
    raise FileNotFoundError(f"Could not find {opt_file}")

opt_history = pickle.load(open(opt_file, "rb"))

# ---- Top-level convenience variables for interactive inspection ----
call_history = opt_history.get("call_history", [])
accepted_x_canonical = opt_history.get("accepted_x_canonical", [])
accepted_x_unscaled = opt_history.get("accepted_x_unscaled", [])
accepted_obj = opt_history.get("accepted_obj", [])
accepted_grad = opt_history.get("accepted_grad", [])
accepted_grad_norm = opt_history.get("accepted_grad_norm", [])
accepted_obj_is_known = opt_history.get("accepted_obj_is_known", [])

# Convenient arrays
accepted_obj = np.asarray(accepted_obj, dtype=float) if len(accepted_obj) else np.array([])
accepted_grad_norm = np.asarray(accepted_grad_norm, dtype=float) if len(accepted_grad_norm) else np.array([])
accepted_obj_is_known = (
    np.asarray(accepted_obj_is_known, dtype=bool)
    if len(accepted_obj_is_known) else np.array([], dtype=bool)
)

num_accepted_iterations = len(accepted_x_canonical)

# Call-level diagnostics
num_calls = len(call_history)
num_success_calls = sum(1 for c in call_history if c.get("success", False))
num_failed_calls = num_calls - num_success_calls

call_objective = (
    np.array([c.get("objective", np.nan) for c in call_history], dtype=float)
    if num_calls else np.array([])
)
call_grad_norm = (
    np.array([c.get("grad_norm", np.nan) for c in call_history], dtype=float)
    if num_calls else np.array([])
)
call_success = (
    np.array([bool(c.get("success", False)) for c in call_history], dtype=bool)
    if num_calls else np.array([], dtype=bool)
)

failure_response = [c.get("failure_response", None) for c in call_history]
failure_response_counts = {}
for fr in failure_response:
    if fr is not None:
        failure_response_counts[fr] = failure_response_counts.get(fr, 0) + 1

# ---- Optional: load SciPy minimize result if present ----
minimize_results = None
minimize_file = "minimize_results.pkl"
if os.path.exists(minimize_file):
    minimize_results = pickle.load(open(minimize_file, "rb"))

# ---- Initial objective (from first successful call) ----
f0 = None
x0 = None
first_success = _first_success(call_history)
if first_success is not None:
    f0 = first_success.get("objective", None)
    x0 = first_success.get("x_canonical", None)

print(f"\n--- OptimizationIterator history from {opt_file} ---")

if f0 is None:
    print("Initial objective: <no successful evaluations recorded>")
else:
    print(f"Initial objective (first successful eval): {float(f0):.6e}")

if x0 is not None:
    x0 = np.asarray(x0, dtype=float)
    print("Initial x_canonical: min/max =", x0.min(), x0.max())

print(f"Accepted iterations: {num_accepted_iterations}")
if num_accepted_iterations:
    known = int(np.sum(accepted_obj_is_known)) if accepted_obj_is_known.size else 0
    print(f"Accepted iterations with known f,g: {known}/{num_accepted_iterations}")

    if accepted_obj.size:
        finite_obj = accepted_obj[np.isfinite(accepted_obj)]
        if finite_obj.size:
            print(f"Accepted objective: min={finite_obj.min():.6e}, last={finite_obj[-1]:.6e}")

    if accepted_grad_norm.size:
        finite_gn = accepted_grad_norm[np.isfinite(accepted_grad_norm)]
        if finite_gn.size:
            print(f"Accepted grad norm: last={finite_gn[-1]:.6e}")

print(f"Total fun/grad calls: {num_calls} (success={num_success_calls}, failed={num_failed_calls})")

if failure_response_counts:
    print("Failure responses used:")
    for k, v in sorted(failure_response_counts.items()):
        print(f"  {k}: {v}")

if minimize_results is not None:
    print("\nLoaded minimize_results.pkl:")
    print("  success:", getattr(minimize_results, "success", None))
    print("  status :", getattr(minimize_results, "status", None))
    print("  message:", getattr(minimize_results, "message", None))
    print("  nit    :", getattr(minimize_results, "nit", None))
    print("  nfev   :", getattr(minimize_results, "nfev", None))
    print("  njev   :", getattr(minimize_results, "njev", None))
else:
    print("\nminimize_results.pkl not found (skipping).")

# ------------------------------------------------------------------
# Optional adaptive-history file from adaptive_lbfgsb
# ------------------------------------------------------------------
adaptive_file = "adaptive_history.pkl"
adaptive_history = None
adaptive_restarts = []
adaptive_best_overall = None

if os.path.exists(adaptive_file):
    adaptive_history = pickle.load(open(adaptive_file, "rb"))
    adaptive_restarts = adaptive_history.get("restarts", [])
    adaptive_best_overall = adaptive_history.get("best_overall", None)
    adaptive_initial_center = adaptive_history.get("initial_center", None)

    print(f"\n--- Adaptive history from {adaptive_file} ---")
    print("Optimizer:", adaptive_history.get("optimizer", "adaptive_lbfgsb"))
    print("Completed restarts recorded:", len(adaptive_restarts))

    if adaptive_initial_center is not None:
        print(f"Initial center f: {_as_float(adaptive_initial_center.get('f', np.nan)):.6e}")

    if adaptive_best_overall is not None:
        print(f"Best overall f: {_as_float(adaptive_best_overall.get('f', np.nan)):.6e}")

    print("\nPer-restart summary (first 40):")
    header = (
        "r  acc burst calls fails fail%   "
        "Delta[min,max,mean]      "
        "f_start       f_trial       ared         rho       bind%   step"
    )
    print(header)

    for i, r in enumerate(adaptive_restarts[:40]):
        accepted = bool(r.get("accepted", False))
        burst = int(r.get("burst_maxiter_used", -1))

        ch = r.get("call_history", None)
        if ch is None:
            ch = r.get("history", {}).get("call_history", [])

        nf, nc = _count_failures(ch)
        fpct = 100.0 * (nf / max(nc, 1))

        D = r.get("Delta_start", None)
        if D is None:
            Dmin = Dmax = Dmean = np.nan
        else:
            D = np.asarray(D, dtype=float).ravel()
            Dmin = float(np.min(D))
            Dmax = float(np.max(D))
            Dmean = float(np.mean(D))

        bind = r.get("binding_mask", None)
        if bind is None:
            bind_pct = np.nan
        else:
            bind = np.asarray(bind, dtype=bool).ravel()
            bind_pct = 100.0 * float(np.mean(bind))

        f_start = _as_float(r.get("fc_start", np.nan))
        f_trial = _as_float(r.get("f_trial", np.nan))
        ared = _as_float(r.get("ared", np.nan))
        rho = _as_float(r.get("rho", np.nan))
        step = _as_float(r.get("step_norm", np.nan))

        print(
            f"{i:2d} {str(accepted):>4s} {burst:5d} {nc:5d} {nf:5d} {fpct:5.1f}  "
            f"[{Dmin:6.2e},{Dmax:6.2e},{Dmean:6.2e}]  "
            f"{f_start:11.4e} {f_trial:11.4e} {ared:11.4e} {rho:9.3e} "
            f"{bind_pct:6.1f} {step:8.2e}"
        )
else:
    print(f"\n{adaptive_file} not found (skipping adaptive summary).")
