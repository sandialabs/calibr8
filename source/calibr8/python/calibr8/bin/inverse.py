import numpy as np
import pickle
import yaml

from scipy.optimize import minimize

from calibr8.util.adaptive_lbfgsb import adaptive_lbfgsb

from calibr8.util.driver_support import (
    evaluate_objective_or_gradient,
    OptimizationIterator
)

from calibr8.util.input_file_io import (
    cleanup_files,
    get_adaptive_options,
    get_opt_options,
    setup_opt_parameters,
    setup_text_parameters,
    standard_parser,
    write_output_file
)


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

    num_iters, gradient_tol, max_ls_evals, barrier_tol = get_opt_options(input_yamls[0])

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

    def finalize(x_opt):
        write_output_file(x_opt, opt_param_scales, opt_param_names, output_file)
        cleanup_files(evaluate_gradient)

    def run_scipy(method, options):
        opt_iterator = OptimizationIterator(objective_args, failure_mode=failure_mode)
        # trust-constr uses a different callback (it inspects the result object)
        callback = (opt_iterator.callback_trust_constr if method == "trust-constr"
                    else opt_iterator.callback)
        res = minimize(
            fun=opt_iterator.objective_fun_and_grad,
            x0=opt_init_params,
            method=method,
            jac=True,
            bounds=opt_bounds,
            options=options,
            callback=callback,
        )
        with open("minimize_results.pkl", "wb") as file:
            pickle.dump(res, file)
        return res

    if args.trust_region:
        tr_opts = {
            "maxiter": num_iters,
            "gtol": gradient_tol,
            "verbose": 3,
            "barrier_tol": barrier_tol,
        }
        res = run_scipy("trust-constr", tr_opts)
        finalize(res.x)
    elif args.adaptive:
        adaptive_opts = get_adaptive_options(input_yamls[0])
        overall = adaptive_lbfgsb(
            x0=opt_init_params,
            global_bounds=opt_bounds,
            objective_args=objective_args,
            lbfgsb_options=l_bfgs_b_opts,
            history_pkl="adaptive_history.pkl",
            failure_mode=failure_mode,
            failure_penalty=1.0e12,
            Delta0=adaptive_opts["Delta0"],
            Delta_min=1.0e-4,
            Delta_max=adaptive_opts["Delta_max"],
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
            rel_impr_tol=1.0e-5,
            eta_accept=0.1,
            eta_keep=0.25,
            eta_grow=0.75,
            pred_floor=1.0e-12,
            max_restarts=50
        )
        finalize(overall["best_overall"]["x"])
    else:
        res = run_scipy("L-BFGS-B", l_bfgs_b_opts)
        finalize(res.x)


if __name__ == "__main__":
    main()
