# calibr8 (Python tools)

Python drivers and utilities for
[Calibr8](https://github.com/sandialabs/calibr8): parameter calibration and
gradient checks driven by the `objective` finite element binary.

## Installation

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/). From this
directory:

```sh
uv venv --python 3.13 --prompt calibr8
uv pip install -e ".[test]"
```

`-e` installs in editable mode; the `[test]` extra adds `pytest` (drop it,
`uv pip install -e .`, for a runtime-only install).

## Console scripts

- `python_inverse` — calibrate parameters from an inverse YAML deck
  (L-BFGS-B or trust-region)
- `python_gradient_fd_check` — finite-difference check of the objective gradient

## Tests

```sh
python -m pytest ../test/python/unit
```
