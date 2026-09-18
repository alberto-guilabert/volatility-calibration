"""Pure price diagnostics. Tolerances are numerical allowances, not repairs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Diagnostic:
    name: str
    residuals: tuple
    violations: tuple

    @property
    def passed(self):
        return not self.violations


def _report(name, residuals, bad):
    return Diagnostic(name, tuple(np.asarray(residuals).reshape(-1).tolist()),
                      tuple(np.flatnonzero(np.asarray(bad).reshape(-1)).tolist()))


def _tolerance(atol):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')


def finite_values(values):
    a = np.asarray(values)
    return _report('finite_values', a, ~np.isfinite(a))


def no_arbitrage_bounds(*, T, K, option_params, option_type='call', F=None):
    from .sinh import _real_array
    k = _real_array(K, 'K')
    market = _real_array(option_params, 'option_params')
    t = _real_array(T, 'T')
    types = np.asarray(option_type)
    if t.ndim or t < 0 or k.ndim > 1 or not k.size or np.any(k <= 0):
        raise ValueError('invalid maturity or strikes')
    if market.shape != (3,) or market[0] <= 0:
        raise ValueError('invalid option_params')
    if types.ndim and types.shape != k.shape:
        raise ValueError('option_type must match strikes')
    if not np.all((types == 'call') | (types == 'put')):
        raise ValueError('invalid option_type')
    if F is not None:
        F = _real_array(F, 'F')
        if F.ndim or F <= 0:
            raise ValueError('F must be a positive scalar')
    s = market[0]*np.exp(-market[2]*t) if F is None else F*np.exp(-market[1]*t)
    d = k*np.exp(-market[1]*t)
    if not np.all(np.isfinite(d)) or not np.isfinite(s):
        raise ValueError('nonfinite discounted bounds')
    return np.maximum(np.where(types == 'call', s-d, d-s), 0), np.where(types == 'call', s, d)


def price_bounds(prices, *, atol=1e-8, **kwargs):
    _tolerance(atol)
    lo, hi = no_arbitrage_bounds(**kwargs)
    a = np.asarray(prices, dtype=float)
    if a.shape != np.asarray(lo).shape:
        raise ValueError('prices must match strikes')
    excess = np.maximum(lo-a, a-hi)
    return _report('price_bounds', excess, ~np.isfinite(a) | (excess > atol))


def put_call_parity(calls, puts, *, T, K, option_params, atol=1e-8, F=None):
    _tolerance(atol)
    # Validate the common financial inputs through the bounds API.
    no_arbitrage_bounds(T=T, K=K, option_params=option_params, F=F)
    c, p, k = np.asarray(calls), np.asarray(puts), np.asarray(K)
    if c.shape != k.shape or p.shape != k.shape:
        raise ValueError('calls and puts must match strikes')
    s, r, q = option_params
    parity = (s*np.exp(-q*T)-k*np.exp(-r*T) if F is None
              else np.exp(-r*T)*(F-k))
    residual = c-p-parity
    return _report('put_call_parity', residual, ~np.isfinite(residual) | (abs(residual) > atol))


def _curve(K, prices):
    k, p = np.asarray(K, dtype=float), np.asarray(prices, dtype=float)
    if k.ndim != 1 or p.shape != k.shape or not k.size or not np.all(np.isfinite(k)) or np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError('strikes must be positive, finite and strictly increasing; prices must match')
    return k, p


def strike_monotonicity(K, prices, *, option_type='call', atol=1e-8):
    _tolerance(atol)
    if option_type not in ('call', 'put'):
        raise ValueError('invalid option_type')
    _, p = _curve(K, prices)
    residual = np.diff(p)*(1 if option_type == 'call' else -1)
    return _report('strike_monotonicity', residual, ~np.isfinite(residual) | (residual > atol))


def strike_convexity(K, prices, *, atol=1e-8):
    """Compare successive secant slopes on irregular strikes; atol is in slope units."""
    _tolerance(atol)
    k, p = _curve(K, prices)
    residual = -np.diff(np.diff(p)/np.diff(k))
    return _report('strike_convexity', residual, ~np.isfinite(residual) | (residual > atol))
