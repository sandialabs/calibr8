#!/usr/bin/env python
"""Check recovered (calibrated) parameters against known truth values.

Usage:
  check_recovered_params.py <calibrated_params.txt> NAME=TRUTH [NAME=TRUTH ...] [--rtol R]

Reads the `name: value` file written by the python_inverse driver and exits 0
if every named parameter is within relative tolerance R (default 0.05) of its
truth value, or 1 (with a report) otherwise.
"""
import argparse
import sys


def read_params(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, value = line.split(":", 1)
            params[name.strip()] = float(value)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file")
    parser.add_argument("expected", nargs="+", help="NAME=VALUE truth pairs")
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()

    recovered = read_params(args.params_file)

    ok = True
    print(f"{'param':>8} {'recovered':>18} {'truth':>18} {'rel err':>12}  status")
    for pair in args.expected:
        name, truth_str = pair.split("=")
        truth = float(truth_str)
        if name not in recovered:
            print(f"{name:>8} {'MISSING':>18}")
            ok = False
            continue
        got = recovered[name]
        rel = abs(got - truth) / max(abs(truth), 1e-30)
        status = "ok" if rel <= args.rtol else "FAIL"
        if rel > args.rtol:
            ok = False
        print(f"{name:>8} {got:18.8e} {truth:18.8e} {rel:12.3e}  {status}")

    if not ok:
        print(f"\nFAILED: one or more parameters outside rtol={args.rtol}")
        return 1
    print(f"\nPASSED: all parameters within rtol={args.rtol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
