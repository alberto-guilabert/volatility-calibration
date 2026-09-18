"""Black prices, vegas and IVs with an authoritative forward (no spot/carry fit).

Rates are continuously compounded; volatility is decimal and vega is price
per unit decimal volatility. Array inputs broadcast for price and vega.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _inputs(iv, T, K, F, r):
    values = []
    for name, value in [('iv', iv), ('T', T), ('K', K), ('F', F), ('r', r)]:
        a = np.asarray(value)
        if a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
            raise ValueError(f'{name} must be finite real values')
        if name in ('K', 'F') and np.any(a <= 0):
            raise ValueError(f'{name} must be positive')
        if name in ('T', 'iv') and np.any(a < 0):
            raise ValueError(f'{name} must be nonnegative')
        values.append(a.astype(float))
    return np.broadcast_arrays(*values)


def _terms(iv, T, K, F, r):
    iv, T, K, F, r = _inputs(iv, T, K, F, r)
    with np.errstate(over='raise', invalid='raise'):
        discount = np.exp(-r*T)
        std = iv*np.sqrt(T)
        log_m = np.log(F)-np.log(K)
        d1 = np.divide(log_m, std, out=np.zeros_like(std), where=std > 0) + .5*std
    return discount, std, d1, K, F, T


def price(iv, T, K, *, F, r, option_type='call'):
    """Discounted Black call/put; zero time or volatility returns intrinsic."""
    d, std, d1, k, f, _ = _terms(iv, T, K, F, r)
    types = np.asarray(option_type)
    if not np.all((types == 'call') | (types == 'put')):
        raise ValueError('option_type must contain call or put')
    d2 = d1-std
    call = d*(f*norm.cdf(d1)-k*norm.cdf(d2))
    put = d*(k*norm.cdf(-d2)-f*norm.cdf(-d1))
    intrinsic = d*np.maximum(np.where(types == 'call', f-k, k-f), 0)
    result = np.where(std > 0, np.where(types == 'call', call, put), intrinsic)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('nonfinite Black price')
    return result


def vega(iv, T, K, *, F, r):
    """Forward Black vega, including the right-hand limit at zero volatility."""
    d, std, d1, k, f, t = _terms(iv, T, K, F, r)
    result = d*f*norm.pdf(d1)*np.sqrt(t)
    result = np.where((std == 0) & (f != k), 0., result)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('nonfinite Black vega')
    return result


def iv_solver(market_price, T, K, *, F, r, option_type='call',
              sigma_hi=5., max_expand=3, tol=1e-12):
    """Scalar Black IV; reject out-of-bounds/unbracketed prices explicitly."""
    for value in (market_price, T, K, F, r, sigma_hi, tol):
        a = np.asarray(value)
        if a.ndim or a.dtype.kind not in 'iuf' or not np.isfinite(a):
            raise ValueError('IV inversion requires finite real scalars')
    if T <= 0 or sigma_hi <= 0 or tol <= 0:
        raise ValueError('T, sigma_hi and tol must be positive')
    if isinstance(max_expand, bool) or not isinstance(max_expand, (int, np.integer)) or max_expand < 0:
        raise ValueError('max_expand must be a nonnegative integer')
    lower = float(price(0., T, K, F=F, r=r, option_type=option_type))
    upper = np.exp(-r*T)*(F if option_type == 'call' else K)
    if market_price < lower or market_price >= upper:
        raise ValueError('market price outside finite-IV Black bounds')
    if market_price == lower:
        return 0.

    def residual(sigma):
        return float(price(sigma, T, K, F=F, r=r, option_type=option_type))-market_price

    high = sigma_hi
    for _ in range(max_expand):
        if residual(high) >= 0:
            break
        high *= 2
    if residual(high) < 0:
        raise ValueError('Black IV solver failed to bracket a root')
    return float(brentq(residual, 0., high, xtol=tol, rtol=1e-10))
