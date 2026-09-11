"""Modified Adams/Boyarchenko-Levendorskii characteristic function.

The scheme subtracts the leading t**alpha term and scales by 1+abs(u).
It is not the standard Adams predictor-corrector. The first correction uses
the previous time node's remainder; subsequent Picard corrections use the
current node's iterate, exactly as in the legacy implementation.

Only numerical kernels use Numba. No fastmath, file cache, global coefficient
cache, plotting, or import-time computation is used. The dense O(M**2) weight
matrix belongs to a single CF call and is released afterwards. Runtime is
O(len(u)*M**2); frequency paths are independent and computed in parallel.

Reference: Boyarchenko, de Innocentis & Levendorskii, arXiv:2508.15080v1,
section 3.2. Paper parameters gamma, nu map to kappa, sigma/kappa here.
"""

from math import gamma
from numbers import Integral

import numpy as np
from numba import njit, prange

from ..params import RoughHestonParams
from .riccati import _frequencies, _log_cf_from_path, _validate_maturity, _validate_params


@njit(cache=False)
def _adams_weights(alpha, delta, steps, gamma2):
    """A[j,k] integrates nodal RHS values up to t_k with the fractional kernel."""
    const = delta**alpha / gamma2
    weights = np.zeros((steps + 1, steps + 1), dtype=np.float64)
    x = np.arange(0, steps + 2, dtype=np.float64)
    powers = x**(alpha + 1.0)
    differences = powers[2:] - 2.0*powers[1:-1] + powers[:-2]
    for k in range(1, steps + 1):
        column = const * differences[:k]
        weights[:k, k] = column[::-1]
        previous = k - 1
        weights[0, k] = const * (
            previous**(alpha + 1.0) - (previous - alpha)*(previous + 1.0)**alpha
        )
        weights[k, k] = const
    return weights


@njit(cache=False)
def _modified_rhs(u, leading, remainder, kappa, sigma, rho):
    return ((1j*u*rho*sigma - kappa) * (leading + remainder)
            + 0.5*sigma**2*(1.0 + np.abs(u))
            * (leading + remainder)*(leading + remainder))


@njit(cache=False)
def _solve_riccati_path(u, times_alpha, weights, picard_iterations,
                        gamma1, kappa, sigma, rho):
    steps = len(times_alpha) - 1
    scale = 1.0 + np.abs(u)
    coefficient = (u*u + 1j*u) / scale
    leading = np.empty(steps + 1, dtype=np.complex128)
    for k in range(steps + 1):
        leading[k] = -(0.5 / gamma1) * coefficient * times_alpha[k]

    remainder = np.zeros(steps + 1, dtype=np.complex128)
    predictor = np.zeros(steps + 1, dtype=np.complex128)
    modified = np.zeros(steps + 1, dtype=np.complex128)
    modified[0] = _modified_rhs(u, leading[0], remainder[0], kappa, sigma, rho)
    failed_step = -1
    for k in range(steps):
        acc = 0.0 + 0.0j
        for m in range(k + 1):
            acc += weights[m, k + 1] * modified[m]
        predictor[k + 1] = acc
        remainder[k + 1] = predictor[k + 1] + weights[k + 1, k + 1] * _modified_rhs(
            u, leading[k + 1], remainder[k], kappa, sigma, rho
        )
        for _ in range(picard_iterations - 1):
            remainder[k + 1] = predictor[k + 1] + weights[k + 1, k + 1] * _modified_rhs(
                u, leading[k + 1], remainder[k + 1], kappa, sigma, rho
            )
        modified[k + 1] = _modified_rhs(
            u, leading[k + 1], remainder[k + 1], kappa, sigma, rho
        )
        if not np.isfinite(remainder[k + 1]) or not np.isfinite(modified[k + 1]):
            failed_step = k + 1
            break

    path = np.empty(steps + 1, dtype=np.complex128)
    for k in range(steps + 1):
        path[k] = scale * (remainder[k] + leading[k])
        if not np.isfinite(path[k]) and failed_step == -1:
            failed_step = k
    return path, failed_step


@njit(cache=False, parallel=True)
def _adams_cf_kernel(u, T, alpha, kappa, theta, sigma, v0, rho,
                     weights, picard_iterations, steps, gamma1):
    delta = T / steps
    times = np.empty(steps + 1, dtype=np.float64)
    times_alpha = np.empty(steps + 1, dtype=np.float64)
    for k in range(steps + 1):
        times[k] = k * delta
    for k in range(steps + 1):
        times_alpha[k] = times[k]**alpha
    values = np.empty(len(u), dtype=np.complex128)
    failures = np.full(len(u), -1, dtype=np.int64)
    for j in prange(len(u)):
        path, failed_step = _solve_riccati_path(
            u[j], times_alpha, weights, picard_iterations, gamma1, kappa, sigma, rho
        )
        if failed_step != -1:
            failures[j] = failed_step
            values[j] = np.nan + 1j*np.nan
        else:
            exponent = _log_cf_from_path(u[j], path, delta, kappa, theta, sigma, v0, rho)
            values[j] = np.exp(exponent)
            if not np.isfinite(exponent) or not np.isfinite(values[j]):
                failures[j] = steps
    return values, failures


def rough_heston_cf_adams(u, T, params: RoughHestonParams, *,
                          time_steps: int = 1000, picard_iterations: int = 2):
    """Return E[exp(i*u*log(S_T/F_T))] using modified Adams.

    u: finite complex scalar or one-dimensional array; scalar input returns
    a Python complex, vector input returns a complex128 array of the same shape.
    T: nonnegative maturity in years. Numerical counts must be positive integers.
    picard_iterations includes the first correction (using the previous remainder).

    T=0, u=0 and u=-1j are handled exactly. The last identity represents the
    normalized stock martingale, assumed in this risk-neutral model; the zero
    Riccati solution also holds at -1j. No carry or log-spot phase is included.
    Other complex u require existence of the corresponding moment.

    Raises ValueError for invalid input/configuration and FloatingPointError
    with the frequency and resolution if the numerical path or CF is nonfinite.
    Finite output alone is not an accuracy guarantee: check time/Picard convergence.
    There is no clipping, automatic refinement, or alternate-method fallback.
    """
    _validate_params(params)
    T = _validate_maturity(T)
    for key, value in (("time_steps", time_steps), ("picard_iterations", picard_iterations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    time_steps, picard_iterations = int(time_steps), int(picard_iterations)
    frequencies = _frequencies(u)
    flat = frequencies.reshape(-1)
    result = np.ones(flat.size, dtype=np.complex128)
    active = (flat != 0.0) & (flat != -1j)
    if T > 0.0 and np.any(active):
        delta = T / time_steps
        if delta == 0.0:
            raise ValueError("T / time_steps underflows to zero")
        weights = _adams_weights(params.alpha, delta, time_steps, gamma(params.alpha + 2.0))
        values, failures = _adams_cf_kernel(
            flat[active], T, params.alpha, params.kappa, params.theta,
            params.sigma, params.v0, params.rho, weights,
            picard_iterations, time_steps, gamma(params.alpha + 1.0),
        )
        bad = np.flatnonzero(failures != -1)
        if bad.size:
            j = bad[0]
            raise FloatingPointError(
                f"Modified Adams produced a nonfinite path/CF at u={flat[active][j]}, "
                f"step={failures[j]}, T={T}, time_steps={time_steps}, "
                f"picard_iterations={picard_iterations}, params={params}. "
                "Check resolution and the complex-frequency moment domain."
            )
        result[active] = values
    return complex(result[0]) if frequencies.ndim == 0 else result
