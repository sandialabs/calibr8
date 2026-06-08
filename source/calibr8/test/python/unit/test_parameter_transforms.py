"""Unit tests for calibr8.util.parameter_transforms.

These cover the canonical<->physical parameter transforms and, most importantly,
the gradient chain rule factor d(physical)/d(canonical) used to map the
simulator's unscaled gradient into the optimizer's canonical space.
"""
import numpy as np
import pytest

from calibr8.util.parameter_transforms import (
    value_transform,
    transform_parameters,
    first_deriv_transform,
    grad_transform,
    get_opt_bounds,
    get_opt_bounds_by_type,
)


# --- value_transform: physical <-> canonical round trips -------------------

@pytest.mark.parametrize("scale, physical", [
    (None, 3.7),                  # no scaling: identity
    (100.0, 250.0),               # float scale: log transform about a ref value
    ([800.0, 1200.0], 950.0),     # [lb, ub]: linear transform, canonical in [-1, 1]
])
def test_value_transform_round_trip(scale, physical):
    canonical = value_transform(physical, scale, transform_from_canonical=False)
    back = value_transform(canonical, scale, transform_from_canonical=True)
    assert back == pytest.approx(physical)


def test_none_scale_is_identity():
    assert value_transform(2.5, None, False) == 2.5
    assert value_transform(2.5, None, True) == 2.5


def test_log_transform_maps_ref_to_zero():
    ref = 100.0
    assert value_transform(ref, ref, False) == pytest.approx(0.0)
    assert value_transform(0.0, ref, True) == pytest.approx(ref)


def test_bounds_transform_endpoints_and_midpoint():
    bounds = [800.0, 1200.0]
    assert value_transform(1000.0, bounds, False) == pytest.approx(0.0)
    assert value_transform(800.0, bounds, False) == pytest.approx(-1.0)
    assert value_transform(1200.0, bounds, False) == pytest.approx(1.0)
    assert value_transform(0.0, bounds, True) == pytest.approx(1000.0)
    assert value_transform(-1.0, bounds, True) == pytest.approx(800.0)
    assert value_transform(1.0, bounds, True) == pytest.approx(1200.0)


def test_bounds_transform_clips_out_of_range_physical():
    bounds = [800.0, 1200.0]
    assert value_transform(700.0, bounds, False) == pytest.approx(-1.0)
    assert value_transform(1300.0, bounds, False) == pytest.approx(1.0)


def test_transform_parameters_vectorized():
    scales = [None, 100.0, [0.0, 10.0]]
    physical = np.array([3.0, 100.0, 5.0])
    canonical = transform_parameters(physical, scales, False)
    assert canonical == pytest.approx([3.0, 0.0, 0.0])
    back = transform_parameters(canonical, scales, True)
    assert back == pytest.approx(physical)


# --- gradient chain rule factor: d(physical)/d(canonical) ------------------

def test_grad_factor_bounds_is_half_span():
    # physical = span*canonical + mean, span = 0.5*(ub - lb)
    assert first_deriv_transform(950.0, [800.0, 1200.0]) == pytest.approx(200.0)


def test_grad_factor_log_is_physical_value():
    # physical = ref*exp(canonical) => d(physical)/d(canonical) = physical
    assert first_deriv_transform(250.0, 100.0) == pytest.approx(250.0)


def test_grad_transform_applies_factor_per_component():
    grad_phys = np.array([2.0, 3.0, 5.0])
    physical = np.array([250.0, 950.0, 4.0])
    scales = [100.0, [800.0, 1200.0], None]
    g = grad_transform(grad_phys, physical, scales)
    # log: 2*250 ; bounds: 3*200 ; none (unscaled): 5*1
    assert g == pytest.approx([2.0 * 250.0, 3.0 * 200.0, 5.0 * 1.0])


def test_grad_factor_none_scale_is_one():
    # d(physical)/d(canonical) for an unscaled parameter is 1
    assert first_deriv_transform(3.7, None) == pytest.approx(1.0)


# --- optimization bounds by scale type -------------------------------------

def test_get_opt_bounds_by_type():
    assert get_opt_bounds_by_type(100.0) == [None, None]          # log: unbounded
    assert get_opt_bounds_by_type(None) == [None, None]           # identity: unbounded
    assert get_opt_bounds_by_type([800.0, 1200.0]) == [-1.0, 1.0]  # bounds: canonical box


def test_get_opt_bounds_vector():
    scales = [100.0, None, [0.0, 10.0]]
    assert get_opt_bounds(scales) == [[None, None], [None, None], [-1.0, 1.0]]
