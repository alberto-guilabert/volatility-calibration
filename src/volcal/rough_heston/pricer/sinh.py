"""Shared covered-call Fourier inversion for normalized-forward CFs.

For X=log(S_T/F_T), m=log(F_T/K), the discounted covered call is
 K exp(-rT)/(2*pi) integral exp(i*u*m) phi(u)/(u*(u+i)) du.
Indeed the Fourier transform of min(exp(x),1), on -1<Im(u)<0,
is 1/(u*(u+i)). Calls and puts subtract this from discounted spot
and strike respectively. With u(y)=i*omega1+b*sinh(y+i*omega),
u(-y)=-conj(u(y)); the negative half is the conjugate of the positive
half for a real-valued distribution. Hence the prefactor becomes 1/pi.
Both endpoints of the finite positive-half trapezoid have weight 1/2.

The supplied callable takes one complex vector and returns the same shape.
It must obey CF conjugate symmetry and be analytic along the deformation,
with decaying tails. These assumptions cannot be inferred from finite values.
"""
import numpy as np

from .bounds import SinhConfig


def fourier_nodes(config):
    """Generate frequencies, full Jacobian and finite trapezoid weights."""
    if not isinstance(config, SinhConfig):
        raise TypeError('integration_config must be SinhConfig')
    y = config.spacing*np.arange(config.nodes+1)
    with np.errstate(over='ignore', invalid='ignore'):
        z = y + 1j*config.omega
        u = 1j*config.omega1 + config.b*np.sinh(z)
        jacobian = config.b*np.cosh(z)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError('nonfinite sinh nodes or Jacobian')
    weights = np.full(u.size, config.spacing)
    weights[[0, -1]] *= 0.5
    return u, jacobian, weights


def evaluate_cf(cf, u):
    """Evaluate once for the whole strike batch; preserve solver exceptions."""
    values = np.asarray(cf(u), dtype=complex)
    if values.shape != u.shape:
        raise ValueError(f'CF returned shape {values.shape}; expected {u.shape}')
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise FloatingPointError(f'nonfinite CF at node {bad[0]}, u={u[bad[0]]}')
    return values


def covered_call_integral(log_moneyness, u, jacobian, weights, values):
    """Dimensionless quadrature, before strike and discount multiplication."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        integrand = (np.exp(1j*np.outer(log_moneyness, u))
                     * (values*jacobian/(u*(u+1j)))[None, :])
        result = np.real(integrand @ weights)/np.pi
    if not np.all(np.isfinite(integrand)) or not np.all(np.isfinite(result)):
        raise FloatingPointError('nonfinite Fourier integrand/quadrature; check contour and moneyness')
    return result


def payoff_conversion(covered_call, discounted_spot, discounted_strike, option_type):
    """Reconstruct calls/puts without clipping or altering finite results."""
    return np.where(option_type == 'call', discounted_spot-covered_call,
                    discounted_strike-covered_call)


def _real_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must contain finite real numbers')
    return a.astype(float)


def vanilla_price_from_cf(cf, *, T, K, option_params, option_type='call',
                          integration_config=SinhConfig(), F=None):
    """Price scalar/1-D strikes with scalar or matching call/put labels.

    cf(u) is the normalized CF at T (bind maturity in the callable).
    F, when supplied, is a positive scalar authoritative forward; spot and q
    remain source inputs and do not determine carry. T=0 returns intrinsic
    against F (or spot when F is absent) without evaluating cf. Explicit numerical
    resolution is the caller's responsibility; finite prices are not clipped.
    """
    maturity = _real_array(T, 'T')
    strikes = _real_array(K, 'K')
    market = _real_array(option_params, 'option_params')
    if maturity.ndim != 0 or maturity < 0:
        raise ValueError('T must be a nonnegative scalar')
    if strikes.ndim > 1 or strikes.size == 0 or np.any(strikes <= 0):
        raise ValueError('K must be a positive scalar or nonempty 1-D vector')
    if market.shape != (3,) or market[0] <= 0:
        raise ValueError('option_params must be (positive S0, r, q)')
    types = np.asarray(option_type)
    if types.ndim != 0 and types.shape != strikes.shape:
        raise ValueError('option_type must be scalar or match K shape')
    if not np.all((types == 'call') | (types == 'put')):
        raise ValueError('option_type must contain only call or put')
    if not isinstance(integration_config, SinhConfig):
        raise TypeError('integration_config must be SinhConfig')
    T = float(maturity)
    S0, r, q = market
    if F is not None:
        F = _real_array(F, 'F')
        if F.ndim or F <= 0:
            raise ValueError('F must be a positive scalar')
        F = float(F)
    if T == 0:
        underlying = S0 if F is None else F
        result = np.where(types == 'call', np.maximum(underlying-strikes, 0),
                          np.maximum(strikes-underlying, 0))
    else:
        with np.errstate(over='ignore', invalid='ignore'):
            disc_spot = S0*np.exp(-q*T) if F is None else F*np.exp(-r*T)
            disc_strikes = strikes*np.exp(-r*T)
            m = (np.log(S0)-np.log(strikes)+(r-q)*T if F is None
                 else np.log(F)-np.log(strikes))
        if (not np.isfinite(disc_spot) or disc_spot <= 0
                or not np.all(np.isfinite(disc_strikes)) or np.any(disc_strikes <= 0)
                or not np.all(np.isfinite(m))):
            raise FloatingPointError('discounting or log-moneyness exceeds float64 range')
        u, jac, weights = fourier_nodes(integration_config)
        values = evaluate_cf(cf, u)
        integral = covered_call_integral(np.atleast_1d(m), u, jac, weights, values)
        result = payoff_conversion(disc_strikes*integral, disc_spot, disc_strikes, types)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('nonfinite reconstructed price')
    return float(np.asarray(result).reshape(-1)[0]) if strikes.ndim == 0 else result
