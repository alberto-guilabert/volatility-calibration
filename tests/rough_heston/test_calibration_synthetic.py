"""Small deterministic optimizer budgets, never production DE searches."""
import numpy as np
import pytest
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.calibrator import PreparedQuotes, CalibrationConfig, calibrate
from volcal.rough_heston.pricer import (RoughHestonPricer, PadeConfig, AdamsConfig,
    SinhConfig, RefinementConfig, FourierRefinementConfig)
from volcal.utils.black_scholes import iv_solver

TRUE = RoughHestonParams(.2,1.5,.04,.4,.04,-.7)
GRID = SinhConfig(spacing=.05,nodes=120)
PADE = RoughHestonPricer('pade',PadeConfig(4),GRID)


def surface(pricer, maturities=(.25,.75)):
    rows=[]
    for t in maturities:
        r,q=.02+.02*t,.01+.005*t
        k=np.array([90.,100.,110.,90.,100.,110.])
        types=np.array(['call']*3+['put']*3)
        p=pricer.vanilla_price(T=t,K=k,option_params=(100,r,q),option_type=types,rough_heston_params=TRUE)
        for strike,typ,value in zip(k,types,p):
            rows.append((t,strike,100.,r,q,typ,value,iv_solver(value,t,strike,(100,r,q),typ)))
    return PreparedQuotes(*zip(*rows))


@pytest.mark.parametrize('fixed',[.2,None],ids=['fixed_H','free_H'])
def test_synthetic_fit(fixed):
    result=calibrate(surface(PADE),PADE,CalibrationConfig(fixed_H=fixed,de_popsize=3,de_maxiter=2,lbfgsb_maxiter=100),
        refinement_config=RefinementConfig(atol=2e-6,rtol=1e-6,
            fourier=FourierRefinementConfig(max_nodes=2400,truncation_increment=20)))
    print('synthetic',fixed,result.params,result.initial_loss,result.final_loss,
          result.final_repricing.max_market_price_error,result.runtime,
          result.de.evaluations,result.lbfgsb.evaluations,result.failed_numerical_evaluations)
    assert result.final_loss < result.initial_loss*1e-4
    assert result.final_repricing.max_market_price_error < .003
    assert result.final_repricing.passed
    assert all(r.converged for _,r in result.refinement)
    assert result.params.H==fixed if fixed is not None else 0 < result.params.H <= .5
    assert result.total_objective_evaluations < 1000


def test_adams_market_pade_fit_adams_validation():
    adams=RoughHestonPricer('adams',AdamsConfig(1600),GRID)
    result=calibrate(surface(adams,(.5,)),PADE,
        CalibrationConfig(fixed_H=.2,de_popsize=2,de_maxiter=1,lbfgsb_maxiter=25),
        adams_validation_pricer=adams)
    a=result.adams_validation
    print('cross',result.final_loss,a.loss,a.max_market_price_error,a.max_market_iv_error,
          a.max_calibration_price_difference,result.runtime)
    assert result.final_loss < result.initial_loss*.01
    assert a.passed and a.iv_comparisons==6
    assert a.max_market_price_error < .03
    assert a.max_calibration_price_difference > 0
    assert result.numerical_configuration==PADE and a.configuration==adams
