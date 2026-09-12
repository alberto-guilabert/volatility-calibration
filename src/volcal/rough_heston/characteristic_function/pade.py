"""Two-point Padé CFs with fixed time quadrature and no solver fallback.

The legacy expanded formulas match n small-time and n formal inverse-power
large-time coefficients. These are approximations, not exact CFs; accuracy
and the complex moment domain remain the caller's responsibility. Order 5
has a singular formal asymptotic recurrence at alpha=2/3; it is not repaired.
"""

from numbers import Integral

import numpy as np
from scipy.special import roots_legendre

from .pade_coefficients import COEFFICIENTS
from .riccati import _frequencies, _validate_maturity, _validate_params, riccati_rhs


def _error(reason, order, u, T, params):
    return FloatingPointError(
        f"Padé order={order}, u={u}, T={T}, params={params}: {reason}"
    )


def _checked_coefficients(u, T, params, order):
    with np.errstate(all='ignore'):
        p, q = COEFFICIENTS[order](np.complex128(u), params)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise _error('nonfinite coefficients; singular matching equations or gamma ratios',
                     order, u, T, params)
    # Inspect the whole time interval, including poles between quadrature nodes.
    end = T**params.alpha
    with np.errstate(all='ignore'):
        scaled = q * end**np.arange(len(q))
    if not np.all(np.isfinite(scaled)):
        raise _error('nonfinite scaled denominator', order, u, T, params)
    try:
        roots = np.polynomial.polynomial.polyroots(scaled)
    except np.linalg.LinAlgError as exc:
        raise _error('denominator root calculation failed', order, u, T, params) from exc
    for root in roots:
        if -1e-10 <= root.real <= 1+1e-10 and abs(root.imag) <= 1e-10:
            x = np.clip(root.real, 0, 1)
            magnitude = abs(np.polynomial.polynomial.polyval(x, scaled))
            raise _error(f'rational pole/near pole at t**alpha={x*end}; '
                         f'denominator magnitude={magnitude}', order, u, T, params)
    return p, q


def _evaluate(t, u, T, params, order, p, q):
    y = np.asarray(t)**params.alpha
    with np.errstate(all='ignore'):
        den = np.polynomial.polynomial.polyval(y, q)
        scale = np.polynomial.polynomial.polyval(abs(y), abs(q))
        bad = abs(den) <= 1e-12*scale
        if np.any(bad):
            raise _error(f'near-zero denominator magnitude={np.min(abs(den))}',
                         order, u, T, params)
        h = np.polynomial.polynomial.polyval(y, p)/den
    if not np.all(np.isfinite(h)):
        raise _error('nonfinite rational path', order, u, T, params)
    return h


def rough_heston_cf_pade(u, T, params, *, order=4):
    """Return normalized-forward log-return CF, using Padé order 2, 3, 4 or 5.

    Order 4 is the recommended/default general-purpose approximation.
    Order 5 may provide better accuracy in some regions, but is unsupported
    at alpha=2/3. No automatic fallback between Padé orders or to Adams occurs.

    Scalar input returns Python complex; a 1-D vector returns complex128.
    Exact T=0, u=0 and u=-1j identities precede coefficient evaluation: both
    special frequencies have R(u,0)=0, hence the zero Riccati solution.
    H=0.5 uses zero inverse-power tails, without changing model parameters.

    Integrate kappa*theta*h + v0*R(u,h) using fixed 128-point Gauss-Legendre
    time quadrature with t=T*x**4 to smooth the origin. No Fourier integration,
    adaptive refinement, clipping, parameter perturbation or Adams fallback.
    Raises ValueError for invalid configuration and FloatingPointError with
    model context for singular coefficients, near poles or nonfinite output.
    """
    _validate_params(params)
    T = _validate_maturity(T)
    if (isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral)
            or order not in COEFFICIENTS):
        raise ValueError('order must be one of the supported integers: 2, 3, 4, 5')
    frequencies = _frequencies(u)
    result = np.ones(frequencies.size, dtype=np.complex128)
    if T > 0:
        nodes, weights = roots_legendre(128)
        x = (nodes+1)/2
        t, w = T*x**4, 2*T*weights*x**3
        for j, z in enumerate(frequencies.reshape(-1)):
            if z == 0 or z == -1j:
                continue
            p, q = _checked_coefficients(z, T, params, order)
            h = _evaluate(t, z, T, params, order, p, q)
            with np.errstate(all='ignore'):
                exponent = np.dot(w, params.kappa*params.theta*h
                                  + params.v0*riccati_rhs(z, h, params))
                result[j] = np.exp(exponent)
            if not np.isfinite(exponent) or not np.isfinite(result[j]):
                raise _error('nonfinite CF/exponent', order, z, T, params)
    return complex(result[0]) if frequencies.ndim == 0 else result
