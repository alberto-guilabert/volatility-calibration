"""Independent series recurrences and polynomial matching residuals."""
from dataclasses import replace
from math import gamma

import numpy as np
import pytest
from scipy.special import gamma as cgamma, rgamma

from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.characteristic_function.pade_coefficients import COEFFICIENTS

PARAMS = RoughHestonParams(.1, 1.5, .04, .4, .04, -.7)


def series(u, params, n):
    a = params.alpha
    c = params.sigma**2/2
    linear = 1j*u*params.rho*params.sigma-params.kappa
    constant = -(u*u+1j*u)/2
    b = np.zeros(n+1, complex)
    for k in range(n):
        rhs = constant if k == 0 else linear*b[k]+c*sum(b[j]*b[k-j] for j in range(1,k))
        b[k+1] = rhs*gamma(1+k*a)/gamma(1+(k+1)*a)
    d = np.sqrt(linear**2-4*c*constant)
    g = np.zeros(n, complex)
    g[0] = (-linear-d)/(2*c)
    if a != 1:
        for k in range(1,n):
            derivative = (rgamma(1-a)*g[0] if k == 1 else
                          cgamma(1-(k-1)*a)*rgamma(1-k*a)*g[k-1])
            g[k] = (c*sum(g[j]*g[k-j] for j in range(1,k))-derivative)/d
    return b,g


def residuals(p,q,b,g):
    n=len(q)-1
    small = p-np.convolve(q,b)[:n+1]
    large = p[::-1]-np.convolve(q[::-1],g)[:n+1]
    return small,large[:n]


@pytest.mark.parametrize('n', [2,3,4,5])
@pytest.mark.parametrize('H', [.05,.1,.2,.3,.49,.499999,.5])
@pytest.mark.parametrize('u', [.001,1,10,2-.5j])
def test_independent_matching(n,H,u):
    params=replace(PARAMS,H=H)
    p,q=COEFFICIENTS[n](np.complex128(u),params)
    b,g=series(u,params,n)
    small,large=residuals(p,q,b,g)
    scale=max(1,np.max(abs(p)),np.max(abs(np.convolve(q,b))),np.max(abs(np.convolve(q[::-1],g))))
    assert np.max(abs(small))/scale < 2e-10
    assert np.max(abs(large))/scale < 2e-10
    y=np.linspace(0,2**params.alpha,201)
    den=np.polynomial.polynomial.polyval(y,q)
    condition=abs(den)/np.polynomial.polynomial.polyval(y,abs(q))
    assert np.min(condition)>1e-5


def test_order_five_asymptotic_recurrence_singularity_is_not_repaired():
    # alpha=2/3: g3 has a nonzero quadratic contribution, while Gamma(1-3a)
    # in g4 diverges. This is not a removable 0*infinity from the linear term.
    p=replace(PARAMS,H=2/3-.5)
    with np.errstate(all='ignore'):
        numerator,denominator=COEFFICIENTS[5](np.complex128(1),p)
    assert not np.all(np.isfinite(denominator))
    _,g=series(1,p,4)
    assert abs(g[3])>1e-5


@pytest.mark.parametrize('n',[2,3,4])
def test_reciprocal_gamma_zero_at_two_thirds(n):
    params=replace(PARAMS,H=2/3-.5)
    p,q=COEFFICIENTS[n](np.complex128(1),params)
    b,g=series(1,params,n)
    small,large=residuals(p,q,b,g)
    np.testing.assert_allclose(small,0,atol=1e-12)
    np.testing.assert_allclose(large,0,atol=1e-12)


@pytest.mark.parametrize('n',[2,3,4,5])
def test_reciprocal_gamma_zero_at_three_quarters(n):
    params=replace(PARAMS,H=.25)
    p,q=COEFFICIENTS[n](np.complex128(1),params)
    b,g=series(1,params,n)
    small,large=residuals(p,q,b,g)
    np.testing.assert_allclose(small,0,atol=1e-12)
    np.testing.assert_allclose(large,0,atol=1e-12)
