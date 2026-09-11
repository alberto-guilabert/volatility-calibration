from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from volcal.rough_heston import PARAMETER_ORDER, RoughHestonParams


PARAMS = RoughHestonParams(0.1, 1.5, 0.04, 0.4, 0.03, -0.7)


def test_immutable_and_explicit_vector_order():
    assert PARAMETER_ORDER == ("H", "kappa", "theta", "sigma", "v0", "rho")
    expected = np.array([0.1, 1.5, 0.04, 0.4, 0.03, -0.7])
    np.testing.assert_array_equal(PARAMS.to_vector(), expected)
    assert RoughHestonParams.from_vector(expected) == PARAMS
    mapping = dict(zip(reversed(PARAMETER_ORDER), reversed(expected)))
    np.testing.assert_array_equal(RoughHestonParams(**mapping).to_vector(), expected)
    expected[0] = 0.2
    assert PARAMS.H == 0.1
    with pytest.raises(FrozenInstanceError):
        PARAMS.H = 0.2
    assert PARAMS.alpha == 0.6


@pytest.mark.parametrize("key,value", [
    ("H", 0), ("H", -0.1), ("H", 0.5001), ("kappa", 0), ("kappa", -1),
    ("theta", -0.01), ("sigma", 0), ("sigma", -1), ("v0", -0.01),
    ("rho", -1.001), ("rho", 1.001),
])
def test_parameter_bounds(key, value):
    with pytest.raises(ValueError, match=key):
        replace(PARAMS, **{key: value})


@pytest.mark.parametrize("key", PARAMETER_ORDER)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "0.1", 0.1j])
def test_parameters_are_finite_real_scalars(key, value):
    with pytest.raises(ValueError, match=key):
        replace(PARAMS, **{key: value})


@pytest.mark.parametrize("rho", [-1, 1])
def test_allowed_boundaries(rho):
    p = replace(PARAMS, H=0.5, theta=0, v0=0, rho=rho)
    assert p.alpha == 1


@pytest.mark.parametrize("values", [[], [1, 2], np.ones((2, 3)), np.ones(6, complex)])
def test_invalid_parameter_vector(values):
    with pytest.raises(ValueError, match="vector"):
        RoughHestonParams.from_vector(values)
