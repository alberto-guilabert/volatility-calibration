"""Fourier validation independent of either rough-Heston CF."""
import numpy as np
import pytest

from volcal.rough_heston.pricer import SinhConfig, vanilla_price_from_cf
from volcal.rough_heston.pricer.sinh import fourier_nodes, covered_call_integral
from volcal.utils.black_scholes import price


@pytest.mark.parametrize('T', [.001, .02, .5, 3.])
@pytest.mark.parametrize('omega', [-.015, 0., .015])
@pytest.mark.parametrize('kind', ['call', 'put'])
def test_lognormal(T, omega, kind):
    K = np.array([50., 80., 100., 120., 200.])
    cf = lambda u: np.exp(-.5*.25**2*T*(u*u+1j*u))
    actual = vanilla_price_from_cf(cf, T=T, K=K, option_params=(100,.04,.015),
        option_type=kind, integration_config=SinhConfig(omega=omega, spacing=.005, nodes=2000))
    expected = price(.25,T,K,(100,.04,.015),kind)
    error = max(abs(actual-expected))
    print(f'BS T={T} omega={omega} {kind}: {error:.9g}')
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)


def test_nodes_jacobian_endpoints_and_full_contour():
    cfg = SinhConfig(omega=.12, b=.8, spacing=.1, nodes=20)
    u,j,w = fourier_nodes(cfg)
    y=np.arange(21)*.1
    np.testing.assert_allclose(u,1j*cfg.omega1+cfg.b*np.sinh(y+1j*cfg.omega))
    eps=1e-5
    derivative=cfg.b*(np.sinh(y+eps+1j*cfg.omega)-np.sinh(y-eps+1j*cfg.omega))/(2*eps)
    np.testing.assert_allclose(j,derivative,rtol=1e-9)
    np.testing.assert_allclose(w, [.05]+[.1]*19+[.05])
    cf=lambda z:np.exp(-.1*(z*z+1j*z))
    half=covered_call_integral([-.2,.3],u,j,w,cf(u))
    z=np.r_[-np.conj(u[:0:-1]),u]
    jac=np.r_[np.conj(j[:0:-1]),j]
    weights=np.full(41,.1);weights[[0,-1]]*=.5
    full=np.real(np.exp(1j*np.outer([-.2,.3],z))@(cf(z)*jac/(z*(z+1j))*weights))/(2*np.pi)
    np.testing.assert_allclose(half,full,atol=1e-15)


def test_fourier_convergence_independent_controls():
    cf=lambda u:np.exp(-.5*.2**2*.5*(u*u+1j*u))
    def error(h,n):
        a=vanilla_price_from_cf(cf,T=.5,K=[80,100,120],option_params=(100,.03,.01),
            integration_config=SinhConfig(spacing=h,nodes=n))
        return max(abs(a-price(.2,.5,np.array([80,100,120]),(100,.03,.01))))
    spacing=[error(h,round(6/h)) for h in [.4,.2,.1,.05]]
    truncation=[error(.05,n) for n in [60,80,100,120]]
    print('BS spacing errors',spacing,'truncation errors',truncation)
    assert spacing[-1]<2e-11 and spacing[0]>spacing[1]>spacing[2]
    assert truncation[-1]<2e-11 and truncation[0]>truncation[1]>truncation[2]


@pytest.mark.parametrize('kwargs', [{'omega1':0},{'omega1':-1},{'omega':np.pi/2},
    {'b':0},{'spacing':0},{'nodes':0},{'nodes':2.5},{'spacing':np.nan},{'nodes':True},
    {'spacing':10,'nodes':100}])
def test_invalid_contour(kwargs):
    with pytest.raises(ValueError): SinhConfig(**kwargs)


@pytest.mark.parametrize('cf,exception',[(lambda u:np.full(u.shape,np.nan),FloatingPointError),
                                      (lambda u:1.,ValueError)])
def test_bad_cf(cf,exception):
    with pytest.raises(exception):
        vanilla_price_from_cf(cf,T=1,K=100,option_params=(100,0,0))


def test_finite_bad_prices_are_not_clipped():
    p=vanilla_price_from_cf(lambda u:np.full(u.shape,2.),T=1,K=100,option_params=(100,0,0))
    assert p<0
