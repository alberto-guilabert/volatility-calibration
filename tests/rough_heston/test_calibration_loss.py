from dataclasses import replace
import numpy as np
import pytest
from volcal.rough_heston import RoughHestonParams, PARAMETER_ORDER
from volcal.rough_heston.calibrator import (PreparedQuotes, CalibrationConfig,
    ObjectiveConfig, CalibrationObjective)
from volcal.rough_heston.pricer import RoughHestonPricer, PadeConfig
from volcal.utils.black_scholes import vega


def quotes(**kw):
    data = dict(T=[.5, .5], K=[100., 110.], S0=[100., 100.], r=[.03, .04],
                q=[.01, .02], option_type=['call', 'put'], market_price=[6., 12.], market_iv=[.2, .25])
    return PreparedQuotes(**dict(data, **kw))


def test_loss_vega_and_floor():
    q = quotes(market_vega=[0., 4.])
    objective = CalibrationObjective(q, RoughHestonPricer('pade'),
        CalibrationConfig(objective=ObjectiveConfig(vega_floor=2.)))
    assert objective.loss([4., 8.]) == 1.
    assert objective.loss(q.market_price) == 0.
    o = CalibrationObjective(quotes(), RoughHestonPricer())
    assert np.allclose(o.scale, [vega(.2, .5, 100, (100,.03,.01)), vega(.25,.5,110,(100,.04,.02))])
    plain = CalibrationObjective(quotes(market_iv=None), RoughHestonPricer(),
        CalibrationConfig(objective=ObjectiveConfig(convention='price_mse')))
    assert plain.loss([4., 8.]) == 10.
    with pytest.raises(ValueError): CalibrationObjective(quotes(market_iv=None), RoughHestonPricer())


@pytest.mark.parametrize('kw', [dict(T=[]), dict(K=[100]), dict(T=.5), dict(T=[[.5,.5]]),
    dict(T=[0,.5]), dict(K=[-1,100]), dict(S0=[np.nan,100]), dict(r=[0,np.inf]),
    dict(option_type=['CALL','put']), dict(market_price=[-1,2]), dict(market_iv=[0,.2]),
    dict(market_vega=[-1,1]), dict(q=['a','b'])])
def test_invalid_quotes(kw):
    with pytest.raises(ValueError): quotes(**kw)


def test_quote_copy_and_grouping():
    k = [100., 110.]
    q = quotes(K=k)
    k[0] = 1
    assert q.K == (100.,110.)
    assert len(q.groups()) == 2  # same maturity, distinct r/q


@pytest.mark.parametrize('kw', [dict(bounds=[(0,1)]*5), dict(fixed_H=.01), dict(fixed_H=.5),
    dict(fixed_H=np.nan), dict(de_popsize=0), dict(seed=-1), dict(de_maxiter=-1),
    dict(failure_penalty=np.inf), dict(failure_penalty=0), dict(lbfgsb_gtol=-1)])
def test_invalid_config(kw):
    with pytest.raises(ValueError): CalibrationConfig(**kw)


@pytest.mark.parametrize('kw',[dict(vega_floor=0),dict(vega_floor=np.nan),dict(convention='iv_mse')])
def test_invalid_objective(kw):
    with pytest.raises(ValueError): ObjectiveConfig(**kw)


def test_parameter_order():
    v = [.2,1.5,.04,.4,.05,-.7]
    assert PARAMETER_ORDER == ('H','kappa','theta','sigma','v0','rho')
    assert CalibrationConfig().parameters(v) == RoughHestonParams(*v)
    assert CalibrationConfig(fixed_H=.2).parameters(v[1:]) == RoughHestonParams(*v)
    with pytest.raises(ValueError): CalibrationConfig(fixed_H=.2).parameters(v)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-.1])
def test_bad_prices_penalized(monkeypatch,bad):
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.full(len(kw['K']),bad))
    obj = CalibrationObjective(quotes(), RoughHestonPricer('pade'))
    vec = [.2,1.5,.04,.4,.04,-.7]
    assert obj(vec) == obj.config.failure_penalty
    assert len(obj.failures) == obj.evaluations == 1
    assert obj.failures[0].parameters == tuple(vec)
    assert obj.failures[0].category == 'invalid_prices'
    assert obj.best_params is None


def test_real_pade_singularity():
    obj = CalibrationObjective(quotes(), RoughHestonPricer('pade',PadeConfig(5)))
    assert obj([1/6,1.5,.04,.4,.04,-.7]) == obj.config.failure_penalty
    assert 'singular' in obj.failures[0].message


def test_tiny_negative_not_clipped(monkeypatch):
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.full(len(kw['K']),-1e-10))
    obj = CalibrationObjective(quotes(), RoughHestonPricer('pade'))
    assert obj([.2,1.5,.04,.4,.04,-.7]) == obj.loss([-1e-10,-1e-10])
    assert not obj.failures


@pytest.mark.parametrize('error',[FloatingPointError('nonfinite CF'),ValueError('invalid numerical config')])
def test_numerical_errors(monkeypatch,error):
    def fail(self,**kw): raise error
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',fail)
    obj=CalibrationObjective(quotes(),RoughHestonPricer('pade'))
    assert obj([.2,1.5,.04,.4,.04,-.7]) == obj.config.failure_penalty
    assert str(error) in obj.failures[0].message


def test_output_shape_is_numerical_failure(monkeypatch):
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.array([1.,2.,3.]))
    obj=CalibrationObjective(quotes(),RoughHestonPricer('pade'))
    assert obj([.2,1.5,.04,.4,.04,-.7])==obj.config.failure_penalty
    assert 'shape' in obj.failures[0].message


def test_programming_error_propagates(monkeypatch):
    def broken(self,**kw): raise AttributeError('programming defect')
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',broken)
    obj=CalibrationObjective(quotes(),RoughHestonPricer('pade'))
    with pytest.raises(AttributeError): obj([.2,1.5,.04,.4,.04,-.7])
    assert not obj.failures
