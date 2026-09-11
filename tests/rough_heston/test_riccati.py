from math import gamma

import numpy as np
import pytest

from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.characteristic_function import (
    characteristic_function_from_riccati,
    leading_riccati_term,
    riccati_rhs,
)


PARAMS = RoughHestonParams(0.1, 1.5, 0.04, 0.4, 0.03, -0.7)


def test_rhs_at_zero_path_and_zero_forcing():
    assert riccati_rhs(2, 0, PARAMS) == -2 - 1j
    assert riccati_rhs(0, 0, PARAMS) == 0j
    assert riccati_rhs(-1j, 0, PARAMS) == 0j
    # Independently expand a real h at u=2: constants + linear + quadratic.
    np.testing.assert_allclose(riccati_rhs(2, 0.2, PARAMS), -2.2968 - 1.112j,
                               rtol=0, atol=5e-16)


def test_rhs_broadcasting_and_input_immutability():
    u = np.array([0, 1, 2 - 0.5j])
    h = np.array([[0], [0.1j]])
    before_u, before_h = u.copy(), h.copy()
    actual = riccati_rhs(u, h, PARAMS)
    assert actual.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert actual[i, j] == riccati_rhs(u[j], h[i, 0], PARAMS)
    np.testing.assert_array_equal(u, before_u)
    np.testing.assert_array_equal(h, before_h)


def test_leading_term_fractional_integral_of_constant_forcing():
    t = np.array([0.0, 0.01, 0.1])
    expected = (-2 - 1j) * t**0.6 / gamma(1.6)
    np.testing.assert_allclose(leading_riccati_term(2, t, PARAMS), expected,
                               rtol=3e-16, atol=0)


def test_cf_construction_retains_trapezoidal_rule():
    # An arbitrary linear path makes the integrand quadratic. This checks the
    # trapezoidal error term analytically, not against another numerical sum.
    T, steps, u, slope = 0.7, 8, 2 - 0.5j, 0.2 + 0.1j
    h = slope * np.linspace(0, T, steps + 1)
    p = PARAMS
    c0 = -0.5 * p.v0 * (u**2 + 1j*u)
    c1 = (p.kappa*p.theta + p.v0*(1j*u*p.rho*p.sigma-p.kappa)) * slope
    c2 = 0.5*p.v0*p.sigma**2*slope**2
    expected = np.exp(c0*T + c1*T**2/2 + c2*T**3*(1/3 + 1/(6*steps**2)))
    actual = characteristic_function_from_riccati(u, h, T, p)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-16)
    assert isinstance(actual, complex)
    vector = characteristic_function_from_riccati(
        np.array([u, u]), np.column_stack([h, h]), T, p
    )
    np.testing.assert_array_equal(vector, [actual, actual])


def test_cf_zero_time():
    np.testing.assert_array_equal(
        characteristic_function_from_riccati([1, 2], np.zeros((3, 2)), 0, PARAMS),
        [1, 1],
    )


@pytest.mark.parametrize("h", [np.zeros((2, 2)), [0], [1, 0], [0, np.nan]])
def test_invalid_riccati_path(h):
    with pytest.raises(ValueError, match="h"):
        characteristic_function_from_riccati(1, h, 0.1, PARAMS)


def test_invalid_rhs_and_asymptotic_inputs():
    with pytest.raises(ValueError, match="finite"):
        riccati_rhs(np.inf, 0, PARAMS)
    with pytest.raises(ValueError, match="t"):
        leading_riccati_term(1, -0.1, PARAMS)
