"""Low-level and convergence checks, independent of any market data.

Algebraic weight tolerances reflect float64 cancellation in second differences.
Convergence assertions compare separate refinements, not identities. Frozen
legacy regression values, when used below, are compatibility checks only.
"""

from math import gamma
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from volcal.rough_heston import RoughHestonParams, rough_heston_cf_adams as cf
from volcal.rough_heston.characteristic_function import leading_riccati_term
from volcal.rough_heston.characteristic_function.adams import (
    _adams_weights, _solve_riccati_path,
)


PARAMS = RoughHestonParams(0.1, 1.5, 0.04, 0.4, 0.04, -0.7)
FREQUENCIES = np.array([0.1, 1, 3, 10, 1 - 0.5j, 3 - 0.5j])


@pytest.mark.parametrize("alpha", [0.51, 0.6, 0.8, 1.0])
def test_weights_integrate_constant_and_linear_functions(alpha):
    delta, steps = 0.03, 20
    weights = _adams_weights(alpha, delta, steps, gamma(alpha + 2))
    times = np.arange(steps + 1) * delta
    assert np.all(weights >= 0)
    assert np.all(weights[:, 0] == 0)
    for k in range(1, steps + 1):
        assert np.all(weights[k + 1:, k] == 0)
        np.testing.assert_allclose(
            weights[:k + 1, k].sum(), times[k]**alpha/gamma(alpha + 1),
            rtol=3e-13, atol=0,
        )
        np.testing.assert_allclose(
            times[:k + 1] @ weights[:k + 1, k],
            times[k]**(alpha + 1)/gamma(alpha + 2), rtol=3e-13, atol=0,
        )


def test_classical_weights_are_trapezoidal_and_not_cached():
    a = _adams_weights(1.0, 0.25, 4, gamma(3))
    np.testing.assert_array_equal(a[:, 4], [0.125, 0.25, 0.25, 0.25, 0.125])
    b = _adams_weights(1.0, 0.25, 4, gamma(3))
    a[0, 4] = 99
    assert b[0, 4] == 0.125


def _path(u, T, steps=100, corrections=2):
    p = PARAMS
    delta = T / steps
    weights = _adams_weights(p.alpha, delta, steps, gamma(p.alpha + 2))
    times_alpha = (np.arange(steps + 1) * delta)**p.alpha
    path, failed = _solve_riccati_path(
        u, times_alpha, weights, corrections, gamma(p.alpha + 1),
        p.kappa, p.sigma, p.rho,
    )
    assert failed == -1
    return path


def test_solver_has_leading_fractional_asymptotic():
    u = 1 - 0.5j
    errors = []
    for T in (1e-4, 1e-6):
        path = _path(u, T)
        assert path[0] == 0j
        leading = leading_riccati_term(u, T, PARAMS)
        errors.append(abs(path[-1]/leading - 1))
    assert errors[1] < 0.1*errors[0]  # scales as T**alpha, alpha=0.6
    assert errors[1] < 1e-3


def test_zero_forcing_in_actual_solver_not_only_public_shortcut():
    for u in (0j, -1j):
        np.testing.assert_array_equal(_path(u, 0.5), np.zeros(101))


def test_time_resolution_convergence():
    values = [cf(FREQUENCIES, 0.5, PARAMS, time_steps=n) for n in (100, 200, 400, 800)]
    differences = [np.max(abs(a-b)) for a, b in zip(values, values[1:])]
    assert differences[1] < 0.5*differences[0], differences
    assert differences[2] < 0.5*differences[1], differences
    assert differences[2] < 3e-6, differences


def test_picard_iteration_convergence_at_fixed_time_grid():
    # A deliberately coarse time grid makes the fixed-point correction error
    # measurable. This tests Picard convergence, not continuum-time accuracy.
    values = [cf(FREQUENCIES, 0.5, PARAMS, time_steps=50, picard_iterations=n)
              for n in (1, 2, 3, 4, 6, 8)]
    errors = [np.max(abs(value-values[-1])) for value in values[:-1]]
    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-6, errors


@pytest.mark.parametrize("key", ["time_steps", "picard_iterations"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.nan, np.inf, "2"])
def test_invalid_numerical_counts(key, value):
    with pytest.raises(ValueError, match=key):
        cf(1, 0.5, PARAMS, **{key: value})


@pytest.mark.parametrize("T", [-1, np.nan, np.inf, 1j, True])
def test_invalid_maturity(T):
    with pytest.raises(ValueError, match="T"):
        cf(1, T, PARAMS)


@pytest.mark.parametrize("u", [np.nan, np.inf, [1, np.inf], [[1, 2]]])
def test_invalid_frequencies(u):
    with pytest.raises(ValueError, match="u"):
        cf(u, 0.5, PARAMS)


def test_params_must_use_validated_contract():
    with pytest.raises(TypeError, match="RoughHestonParams"):
        cf(1, 0.5, {})


def test_determinism_shapes_and_no_mutation():
    u = np.array([1 - 0.5j, 2, 3 - 0.5j])
    original = u.copy()
    result = cf(u, 0.1, PARAMS, time_steps=100)
    np.testing.assert_array_equal(result, cf(u, 0.1, PARAMS, time_steps=100))
    np.testing.assert_array_equal(u, original)
    assert result.shape == u.shape and result.dtype == np.complex128
    assert isinstance(cf(u[0], 0.1, PARAMS, time_steps=100), complex)
    assert result[0] == cf(u[0], 0.1, PARAMS, time_steps=100)
    assert cf([], 0.1, PARAMS).shape == (0,)


def test_nonfinite_solver_result_is_an_informative_error():
    with pytest.raises(FloatingPointError, match="u=.*time_steps=8.*picard_iterations=2"):
        cf(1000 + 1000j, 2, PARAMS, time_steps=8)


@pytest.mark.parametrize("H,T,steps,corrections,expected", [
    (0.1, 0.5, 200, 2, [
        0.9892957580463726-0.008980214683438245j,
        0.9144230755363009-0.007213776513061401j,
        0.9881665182056298+0.0009035177316407984j,
        0.920496796426707+0.016869652438081164j,
    ]),
    (0.3, 0.1, 100, 3, [
        0.9979534619008313-0.0019459952404562006j,
        0.9818560402831833-0.004577422898646039j,
        0.9975347567447456+5.832701808758084e-05j,
        0.98201185233887+0.0012634227168481075j,
    ]),
    (0.5, 1.0, 200, 2, [
        0.9784581885503651-0.017502274078517177j,
        0.8375632073466328-0.008365843818115596j,
        0.9766438725896365+0.0020072218807787793j,
        0.8500088811258208+0.03457605206910724j,
    ]),
])
def test_synthetic_legacy_regression(H, T, steps, corrections, expected):
    # Fresh synthetic outputs from the read-only AdamsBL implementation, not
    # cached data or fitted market parameters. Generated with NumPy 1.24.2 and
    # Numba 0.61.0. Direct old/new comparison was bit-for-bit equal. Allow a
    # small platform-roundoff budget here; this is not the mathematical oracle.
    p = RoughHestonParams(H, 1.5, 0.04, 0.4, 0.04, -0.7)
    actual = cf([1, 3, 1-0.5j, 3-0.5j], T, p,
                time_steps=steps, picard_iterations=corrections)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def test_import_does_not_load_plotting_or_market_data():
    # Use a fresh interpreter, independent of modules imported by pytest plugins.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    subprocess.run([
        sys.executable, "-B", "-c",
        "import sys; import volcal.rough_heston; "
        "assert 'matplotlib' not in sys.modules; "
        "assert 'volcal.market_data' not in sys.modules; "
        "assert not any('migration_source' in name for name in sys.modules)",
    ], check=True, env=env, capture_output=True, text=True)
