r"""Mathematical contract for the normalized rough-Heston CF.

With alpha = H + 1/2 and h(0,u) = 0,

    R(u,h) = -0.5*(u**2 + i*u) + (i*u*rho*sigma-kappa)*h
             + 0.5*sigma**2*h**2,
    h(t,u) = integral_0^t (t-s)**(alpha-1)*R(u,h(s,u)) ds / Gamma(alpha).

The output convention is phi(u,T) = E[exp(i*u*log(S_T/F_T))], where
F_T = S_0*exp((r-q)*T). Spot, carry and discounting are not inputs here.

For the zero-initial-condition fractional equation, I**(1-alpha) h(T)
= integral_0^T R(u,h(t,u)) dt. Consequently,

    log(phi(u,T)) = integral_0^T [kappa*theta*h(t,u) + v0*R(u,h(t,u))] dt.

This identity is exact for the Riccati solution; using an approximate path
and a trapezoidal rule introduces numerical error. At alpha=1, the v0
term is v0*h(T). Complex frequencies represent analytic continuation where
the corresponding moment exists; parameter validation does not certify an
arbitrary complex-frequency moment domain.

References: El Euch & Rosenbaum, arXiv:1609.02108; Boyarchenko,
de Innocentis & Levendorskii, arXiv:2508.15080v1, sections 2 and 3.
"""

from math import gamma
from numbers import Real

import numpy as np
from numba.extending import register_jitable

from ..params import RoughHestonParams


def _validate_params(params):
    if not isinstance(params, RoughHestonParams):
        raise TypeError("params must be a RoughHestonParams instance")


def _validate_maturity(T):
    if (isinstance(T, (bool, np.bool_)) or not isinstance(T, Real)
            or not np.isfinite(T) or T < 0):
        raise ValueError("T must be a finite real number >= 0")
    return float(T)


def _frequencies(u):
    array = np.asarray(u, dtype=np.complex128)
    if array.ndim > 1 or not np.all(np.isfinite(array)):
        raise ValueError("u must be a finite complex scalar or one-dimensional vector")
    return array


@register_jitable
def _rhs(u, h, kappa, sigma, rho):
    # Retain the legacy arithmetic order (including u*u and h*h).
    return (-0.5 * (u*u + 1j*u)
            + (1j*u*rho*sigma - kappa) * h + 0.5 * sigma**2 * h*h)


def riccati_rhs(u, h, params: RoughHestonParams):
    """Evaluate R(u,h), broadcasting finite complex inputs using NumPy rules."""
    _validate_params(params)
    u = np.asarray(u, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(h)):
        raise ValueError("u and h must be finite")
    return _rhs(u, h, params.kappa, params.sigma, params.rho)


def leading_riccati_term(u, t, params: RoughHestonParams):
    """Return -0.5*(u**2+i*u)*t**alpha/Gamma(alpha+1), broadcasting u,t."""
    _validate_params(params)
    u = np.asarray(u, dtype=np.complex128)
    t = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("u must be finite and t must be finite and >= 0")
    return -(0.5 / gamma(params.alpha + 1.0)) * (u*u + 1j*u) * t**params.alpha


@register_jitable
def _log_cf_from_path(u, h, delta, kappa, theta, sigma, v0, rho):
    # Endpoints first, then ascending interior nodes: the legacy summation order.
    g0 = kappa*theta*h[0] + v0*_rhs(u, h[0], kappa, sigma, rho)
    gm = kappa*theta*h[-1] + v0*_rhs(u, h[-1], kappa, sigma, rho)
    acc = 0.0 + 0.0j
    acc += 0.5 * (g0 + gm)
    for k in range(1, len(h) - 1):
        acc += kappa*theta*h[k] + v0*_rhs(u, h[k], kappa, sigma, rho)
    return delta * acc


def characteristic_function_from_riccati(u, h, T, params: RoughHestonParams):
    """Construct phi from a path on a uniform grid including 0 and T.

    h has shape (time_steps+1,) for scalar u, or (time_steps+1, len(u))
    for a vector. At least two nodes and a zero initial path value are required.
    The ordinary trapezoidal construction is also used by the Adams kernel.
    """
    _validate_params(params)
    T = _validate_maturity(T)
    u = _frequencies(u)
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim != u.ndim + 1 or h.shape[1:] != u.shape or h.shape[0] < 2:
        raise ValueError("h must have shape (time_steps+1, *u.shape), with >= 2 nodes")
    if not np.all(np.isfinite(h)) or np.any(h[0] != 0):
        raise ValueError("h must be finite with h(0,u) = 0")
    result = np.ones(u.size, dtype=np.complex128)
    if T != 0:
        paths = h.reshape(h.shape[0], u.size)
        for j, frequency in enumerate(u.ravel()):
            exponent = _log_cf_from_path(
                frequency, paths[:, j], T / (h.shape[0] - 1),
                params.kappa, params.theta, params.sigma, params.v0, params.rho,
            )
            if not np.isfinite(exponent):
                raise FloatingPointError(f"nonfinite CF exponent at u={frequency}")
            result[j] = np.exp(exponent)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite characteristic function from Riccati path")
    return complex(result[0]) if u.ndim == 0 else result
