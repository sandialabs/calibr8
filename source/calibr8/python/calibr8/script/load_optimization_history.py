import os
import numpy as np
import pickle


opt_file = "optimization_history.pkl"
if not os.path.exists(opt_file):
    raise FileNotFoundError(f"Could not find {opt_file}")

opt_history = pickle.load(open(opt_file, "rb"))

# ---- New history schema (from updated OptimizationIterator) ----
call_history = opt_history.get("call_history", [])
accepted_x_canonical = opt_history.get("accepted_x_canonical", [])
accepted_x_unscaled = opt_history.get("accepted_x_unscaled", [])
accepted_obj = opt_history.get("accepted_obj", [])
accepted_grad = opt_history.get("accepted_grad", [])
accepted_grad_norm = opt_history.get("accepted_grad_norm", [])
accepted_obj_is_known = opt_history.get("accepted_obj_is_known", [])

# Convenient arrays (may contain NaNs if obj/grad at accepted xk was unknown)
accepted_obj = np.asarray(accepted_obj, dtype=float) if len(accepted_obj) else np.array([])
accepted_grad_norm = np.asarray(accepted_grad_norm, dtype=float) if len(accepted_grad_norm) else np.array([])
accepted_obj_is_known = np.asarray(accepted_obj_is_known, dtype=bool) if len(accepted_obj_is_known) else np.array([], dtype=bool)

num_accepted_iterations = len(accepted_x_canonical)

# Call-level diagnostics
num_calls = len(call_history)
num_success_calls = sum(1 for c in call_history if c.get("success", False))
num_failed_calls = num_calls - num_success_calls

call_objective = np.array([c.get("objective", np.nan) for c in call_history], dtype=float) if num_calls else np.array([])
call_grad_norm = np.array([c.get("grad_norm", np.nan) for c in call_history], dtype=float) if num_calls else np.array([])
call_success = np.array([bool(c.get("success", False)) for c in call_history], dtype=bool) if num_calls else np.array([], dtype=bool)

# ---- Optional: load SciPy minimize result if present ----
minimize_results = None
minimize_file = "minimize_results.pkl"
if os.path.exists(minimize_file):
    minimize_results = pickle.load(open(minimize_file, "rb"))

# ---- Initial objective (from first successful call) ----
f0 = None
x0 = None
for c in call_history:
    if c.get("success", False):
        f0 = c.get("objective", None)
        x0 = c.get("x_canonical", None)
        break

if f0 is None:
    print("Initial objective: <no successful evaluations recorded>")
else:
    print(f"Initial objective (first successful eval): {float(f0):.6e}")

if x0 is not None:
    x0 = np.asarray(x0, dtype=float)
    print("Initial x_canonical: min/max =", x0.min(), x0.max())

# ---- Quick summary (variables are available for interactive inspection) ----
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
