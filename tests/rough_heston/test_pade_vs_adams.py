"""Full-complex approximation budgets, not monotonic-order guarantees.

Grid: H=.05,.2,.49,.5; T=.01,.5,2; (rho,sigma)=(-.7,.4),(.7,.8),(0,.1);
u=.1,1,3,10,1-.5j,3-.5j. Adams 1600->3200 change must be <5e-6.
The relatively large Padé budgets document finite-order approximation error,
especially near the classical limit; they do not certify pricing accuracy.
"""
from dataclasses import replace
import numpy as np
import pytest

from volcal.rough_heston import rough_heston_cf_adams as adams, rough_heston_cf_pade as pade
from test_pade_coefficients import PARAMS
from test_cf_identities import classical_heston_cf

U=np.array([.1,1,3,10,1-.5j,3-.5j])
ABS_BUDGET={2:.050,3:.019,4:.008,5:.0035}
REL_BUDGET={2:.16,3:.065,4:.028,5:.013}


@pytest.fixture(scope='module')
def grid():
    rows=[]
    for H in [.05,.2,.49,.5]:
        for T in [.01,.5,2]:
            for rho,sigma in [(-.7,.4),(.7,.8),(0,.1)]:
                p=replace(PARAMS,H=H,rho=rho,sigma=sigma)
                coarse=adams(U,T,p,time_steps=1600)
                fine=adams(U,T,p,time_steps=3200)
                assert np.max(abs(fine-coarse))<5e-6
                rows.append((p,T,fine))
    return rows


@pytest.mark.parametrize('order',[2,3,4,5])
def test_full_complex_grid(grid,order):
    absolute=[];relative=[];heston=[]
    for p,T,ref in grid:
        value=pade(U,T,p,order=order)
        diff=abs(value-ref)
        absolute.extend(diff);relative.extend(diff/abs(ref))
        if p.H==.5:
            heston.extend(abs(value-classical_heston_cf(U,T,p)))
    stats=(max(absolute),max(relative),max(heston))
    print(f'order={order} max_abs={stats[0]:.9g} max_rel={stats[1]:.9g} Heston={stats[2]:.9g}')
    assert stats[0]<ABS_BUDGET[order],stats
    assert stats[1]<REL_BUDGET[order],stats
    assert stats[2]<ABS_BUDGET[order],stats


@pytest.mark.parametrize('order',[2,3,4,5])
def test_continuity_at_heston_limit(order):
    exact=pade(U,.5,replace(PARAMS,H=.5),order=order)
    near=pade(U,.5,replace(PARAMS,H=.5-1e-8),order=order)
    np.testing.assert_allclose(near,exact,atol=1e-8,rtol=0)
