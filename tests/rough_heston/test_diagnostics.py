import numpy as np
import pytest
from volcal.rough_heston.pricer import (finite_values, no_arbitrage_bounds,
    price_bounds, put_call_parity, strike_monotonicity, strike_convexity, price_change)


def test_bounds_and_parity_independent():
    args=dict(T=1, K=np.array([80.,100.,130.]), option_params=(100,0,0))
    lo, hi=no_arbitrage_bounds(**args)
    np.testing.assert_array_equal(lo,[20,0,0])
    np.testing.assert_array_equal(hi,[100]*3)
    c=np.array([23.,8.,1.]); p=c-100+args['K']
    assert price_bounds(c,**args).passed
    assert price_bounds(p,option_type='put',**args).passed
    assert put_call_parity(c,p,**args).passed
    assert not put_call_parity(c,p+.01,**args).passed
    assert not price_bounds([-1,101,0],**args).passed
    np.testing.assert_array_equal(c,[23,8,1])


def test_curves_irregular_strikes():
    assert strike_monotonicity([80,100,130],[23,8,1]).passed
    assert strike_convexity([80,100,130],[23,8,1]).passed
    assert not strike_convexity([80,100,130],[23,20,1]).passed
    assert not strike_monotonicity([80,100],[1,2]).passed
    assert strike_monotonicity([80,100],[1,2],option_type='put').passed
    with pytest.raises(ValueError): strike_convexity([100,80],[1,2])


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_finite(bad):
    assert finite_values([1,2]).passed
    assert finite_values([1,bad]).violations==(1,)


def test_near_zero_and_tiny_negative():
    d,r,ok=price_change([0,1e-12],[1e-10,0],atol=1e-9,rtol=1e-6)
    assert ok and r==(1.,1.)
    assert price_change([0],[0],atol=1e-9,rtol=0)==((0.,),(0.,),True)
    assert not price_change([0],[1e-10],atol=0,rtol=1e-6)[2]
    p=np.array([-1e-10])
    assert price_bounds(p,T=1,K=[120],option_params=(100,0,0)).passed
    assert p[0]<0
