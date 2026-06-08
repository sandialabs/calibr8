"""Unit tests for OptimizationIterator failure handling.

The simulator-backed evaluator is monkeypatched out so these run with no FE
solve and no `objective` binary. They cover the two failure modes the driver
relies on to survive bad parameter regions:
  - penalty_inward: finite penalty objective + a fake gradient whose negative
    (the optimizer's step direction) points back toward the last good point;
  - repeat_last: reuse the previous successful objective/gradient, falling back
    to penalty_inward when no success has happened yet.
"""
import numpy as np
import pytest

import calibr8.util.driver_support as ds
from calibr8.util.driver_support import OptimizationIterator


def patch_eval(monkeypatch, fn):
    """Replace the FE-backed evaluator with a pure-Python fake."""
    monkeypatch.setattr(ds, "evaluate_objective_and_gradient", fn)


def make_iter(**kwargs):
    # objective_args[0] is used as `scales` by the callbacks; a length-2 list of
    # Nones means identity transforms for the 2-D test points below.
    return OptimizationIterator(objective_args=([None, None],), **kwargs)


def test_success_records_value_grad_and_history(monkeypatch):
    patch_eval(monkeypatch, lambda x, *a, failure_penalty=1e12: (2.0, np.array([1.0, -1.0]), True))

    it = make_iter()
    obj, grad = it.objective_fun_and_grad(np.array([0.5, 0.5]))

    assert obj == 2.0
    assert grad == pytest.approx([1.0, -1.0])

    assert len(it.history["call_history"]) == 1
    rec = it.history["call_history"][0]
    assert rec["success"] is True
    assert rec["objective"] == 2.0
    assert rec["failure_response"] is None

    assert it._last_success_obj == 2.0
    assert it._last_success_x == pytest.approx([0.5, 0.5])


def test_penalty_inward_failure_returns_finite_penalty_and_inward_step(monkeypatch):
    calls = {"n": 0}

    def fake(x, *a, failure_penalty=1e12):
        calls["n"] += 1
        if calls["n"] == 1:
            return 1.0, np.array([3.0, 4.0]), True   # establish a "last good" anchor
        return float(failure_penalty), None, False

    patch_eval(monkeypatch, fake)

    it = make_iter(failure_mode="penalty_inward", failure_penalty=1e12)

    x_good = np.array([0.0, 0.0])
    it.objective_fun_and_grad(x_good)
    x_bad = np.array([1.0, 1.0])
    obj, grad = it.objective_fun_and_grad(x_bad)

    assert np.isfinite(obj)
    assert obj == 1e12

    # fake gradient is a positive multiple of (x_bad - x_good), so the optimizer's
    # negative-gradient step points back toward the last good point.
    direction = x_bad - x_good
    cos = np.dot(grad, direction) / (np.linalg.norm(grad) * np.linalg.norm(direction))
    assert cos == pytest.approx(1.0)
    assert 0.0 < np.linalg.norm(grad) <= it.fake_grad_cap + 1e-12

    rec = it.history["call_history"][-1]
    assert rec["success"] is False
    assert rec["failure_response"] == "penalty_inward"

    # a failure must not overwrite the cached last-good state
    assert it._last_success_obj == 1.0
    assert it._last_success_x == pytest.approx(x_good)


def test_repeat_last_failure_reuses_last_success(monkeypatch):
    calls = {"n": 0}

    def fake(x, *a, failure_penalty=1e12):
        calls["n"] += 1
        if calls["n"] == 1:
            return 5.0, np.array([2.0, 3.0]), True
        return float(failure_penalty), None, False

    patch_eval(monkeypatch, fake)

    it = make_iter(failure_mode="repeat_last")
    it.objective_fun_and_grad(np.array([0.0, 0.0]))
    obj, grad = it.objective_fun_and_grad(np.array([9.0, 9.0]))

    assert obj == 5.0
    assert grad == pytest.approx([2.0, 3.0])
    rec = it.history["call_history"][-1]
    assert rec["success"] is False
    assert rec["failure_response"] == "repeat_last"


def test_repeat_last_without_prior_success_falls_back_to_penalty(monkeypatch):
    patch_eval(monkeypatch, lambda x, *a, failure_penalty=1e12: (float(failure_penalty), None, False))

    it = make_iter(failure_mode="repeat_last")
    obj, grad = it.objective_fun_and_grad(np.array([1.0, 2.0]))

    assert obj == it.failure_penalty
    assert np.all(np.isfinite(grad))
    rec = it.history["call_history"][-1]
    assert rec["failure_response"] == "penalty_inward"


def test_callback_records_accepted_iterate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)   # callback writes optimization_history.pkl to CWD
    patch_eval(monkeypatch, lambda x, *a, failure_penalty=1e12: (7.0, np.array([1.0, 1.0]), True))

    it = make_iter()
    x = np.array([0.2, 0.4])
    it.objective_fun_and_grad(x)
    it.callback(x)

    assert it.history["accepted_x_canonical"][-1] == pytest.approx(x)
    assert it.history["accepted_obj_is_known"][-1] is True
    assert it.history["accepted_obj"][-1] == 7.0
    assert (tmp_path / "optimization_history.pkl").exists()
