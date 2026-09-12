from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from volcal.rough_heston import rough_heston_cf_pade as cf
from volcal.rough_heston.characteristic_function import pade
from test_pade_coefficients import PARAMS


@pytest.mark.parametrize('order',[2,3,4,5])
@pytest.mark.parametrize('H',[.05,.2,.499999,.5])
def test_identities_shapes_and_continuity(order,H):
    p=replace(PARAMS,H=H)
    u=np.array([0,-1j,1,2-.5j]); saved=u.copy(); before=p.to_vector()
    assert isinstance(cf(1+.2j,.5,p,order=order),complex)
    z=cf(u,.5,p,order=order)
    assert z.shape==u.shape and z.dtype==np.complex128
    np.testing.assert_array_equal(z[:2],[1,1])
    np.testing.assert_array_equal(cf(u,0,p,order=order),np.ones(4))
    np.testing.assert_array_equal(u,saved)
    np.testing.assert_array_equal(p.to_vector(),before)
    np.testing.assert_array_equal(z,cf(u,.5,p,order=order))
    assert cf([],1,p,order=order).shape==(0,)
    for center in [0,-1j]:
        errors=[abs(cf(center+eps,.5,p,order=order)-1) for eps in [1e-3,1e-5,1e-7]]
        assert errors[1]<.02*errors[0] and errors[2]<.02*errors[1]
        assert errors[-1]<1e-8
    u=np.linspace(0,20,41)
    z=cf(u,.5,p,order=order)
    np.testing.assert_allclose(cf(-u,.5,p,order=order),z.conjugate(),atol=1e-14,rtol=0)
    assert max(abs(z))<=1+1e-14


@pytest.mark.parametrize('order',[True,False,0,1,6,7,2.,'5',None])
def test_invalid_order(order):
    with pytest.raises(ValueError,match='supported integers'):
        cf(0,0,PARAMS,order=order)


@pytest.mark.parametrize('u,T',[(np.nan,1),(np.inf,1),([[1]],1),(1,-1),(1,np.inf),(1,True)])
def test_invalid_inputs(u,T):
    with pytest.raises(ValueError):cf(u,T,PARAMS)


def test_invalid_params():
    with pytest.raises(TypeError):cf(1,1,{})


@pytest.mark.parametrize('order',[2,3,4,5])
def test_special_identities_precede_coefficients(monkeypatch,order):
    def forbidden(*args):raise AssertionError('coefficients must not be evaluated')
    monkeypatch.setitem(pade.COEFFICIENTS,order,forbidden)
    assert cf(3j,0,PARAMS,order=order)==1
    np.testing.assert_array_equal(cf([0,-1j],1,PARAMS,order=order),[1,1])


@pytest.mark.parametrize('root',[.37,.37+1e-12j])
def test_poles_between_nodes(monkeypatch,root):
    monkeypatch.setitem(pade.COEFFICIENTS,2,lambda *args:(np.array([0,1,0]),np.array([1,-1/root,0])))
    with pytest.raises(FloatingPointError,match='order=2.*u=.*T=.*params=.*denominator magnitude'):
        cf(1,1,PARAMS,order=2)


def test_near_zero_evaluated_denominator():
    with pytest.raises(FloatingPointError,match='denominator magnitude'):
        pade._evaluate(np.array([1.]),1,1,PARAMS,2,np.array([0,1]),np.array([1,-1+1e-14]))


def test_nonfinite_coefficients(monkeypatch):
    monkeypatch.setitem(pade.COEFFICIENTS,3,lambda *args:(np.array([0,np.nan]),np.array([1,1])))
    with pytest.raises(FloatingPointError,match='order=3.*nonfinite coefficients'):
        cf(1,1,PARAMS,order=3)


def test_singular_asymptotic_configuration():
    params=replace(PARAMS,H=2/3-.5)
    # The general-purpose default remains usable where order 5 is unsupported.
    value=cf(1,1,params)
    assert np.isfinite(value)
    assert value==cf(1,1,params,order=4)
    with pytest.raises(FloatingPointError,match='order=5.*gamma ratios'):
        cf(1,1,params,order=5)


@pytest.mark.parametrize('order',[2,3,4,5])
def test_fixed_time_quadrature_against_independent_adaptive_integral(order):
    p=replace(PARAMS,H=.05);u=3-.5j;T=2
    num,den=pade._checked_coefficients(u,T,p,order)
    def integrand(t):
        h=np.polynomial.polynomial.polyval(t**p.alpha,num)/np.polynomial.polynomial.polyval(t**p.alpha,den)
        return p.kappa*p.theta*h+p.v0*(-.5*(u*u+1j*u)+(1j*u*p.rho*p.sigma-p.kappa)*h+.5*p.sigma**2*h*h)
    value=quad(lambda t:integrand(t).real,0,T,epsabs=1e-12)[0]+1j*quad(lambda t:integrand(t).imag,0,T,epsabs=1e-12)[0]
    assert abs(cf(u,T,p,order=order)-np.exp(value))<2e-11
