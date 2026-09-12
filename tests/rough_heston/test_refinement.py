from dataclasses import FrozenInstanceError, replace
from time import perf_counter
import ast
import inspect
import numpy as np
import pytest
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.pricer import (RoughHestonPricer, AdamsConfig, PadeConfig,
    SinhConfig, RefinementConfig, FourierRefinementConfig, AdamsRefinementConfig,
    refine_prices, ConvergenceError)

ARGS=dict(T=.5,K=[80,100,120],option_params=(100,.04,.015),
          rough_heston_params=RoughHestonParams(.2,1.5,.04,.4,.04,-.7))
POLICY=RefinementConfig(atol=2e-6,rtol=1e-6,
    fourier=FourierRefinementConfig(max_nodes=2400,truncation_increment=20))


@pytest.mark.parametrize('method',['adams','pade'])
def test_stable_and_history(method):
    p=RoughHestonPricer(method,AdamsConfig(400) if method=='adams' else PadeConfig(),
                       SinhConfig(spacing=.05,nodes=100))
    p.vanilla_price(**ARGS)  # warm compilation before measurement
    start=perf_counter(); result=refine_prices(p,config=POLICY,**ARGS)
    print(method,'seconds',perf_counter()-start,'reason',result.termination_reason)
    for s in result.history:
        print(s.stage,s.configuration.cf_config,s.configuration.integration_config,
              s.absolute_change,s.failure)
    assert result.converged
    assert result.initial_configuration==p
    assert result.cf_method==method
    assert all(s.configuration.cf_method==method for s in result.history)
    if method=='pade':
        assert all(s.configuration.cf_config.order==4 for s in result.history)
        assert result.warnings
    else:
        time=[s for s in result.history if s.stage=='adams_time']
        if len(time)>1: assert max(time[-1].absolute_change)<max(time[0].absolute_change)
    previous=p
    for s in result.history[1:]:
        if s.stage.startswith('validation/'): continue
        a,b=previous.integration_config,s.configuration.integration_config
        if s.stage=='fourier_spacing':
            assert a.spacing*a.nodes==b.spacing*b.nodes
            assert previous.cf_config==s.configuration.cf_config
        elif s.stage=='fourier_truncation':
            assert a.spacing==b.spacing and b.nodes>a.nodes
        else: assert a==b
        previous=s.configuration
    with pytest.raises(FrozenInstanceError): result.final_configuration.cf_method='pade'
    with pytest.raises(FrozenInstanceError): POLICY.atol=1
    assert isinstance(result.final_prices,tuple)


def test_adams_resolution_difference_decreases():
    values=[RoughHestonPricer('adams',AdamsConfig(n),SinhConfig(spacing=.05,nodes=100)).vanilla_price(**ARGS)
            for n in [200,400,800]]
    assert max(abs(values[2]-values[1]))<max(abs(values[1]-values[0]))


@pytest.mark.parametrize('stage',['nodes','time','picard','iterations'])
def test_exhaustion(stage):
    p=RoughHestonPricer('adams',AdamsConfig(400),SinhConfig(spacing=.05,nodes=100))
    policy=POLICY
    if stage=='nodes': policy=replace(policy,fourier=FourierRefinementConfig(max_nodes=100))
    if stage=='time': policy=replace(policy,adams=AdamsRefinementConfig(max_time_steps=400))
    if stage=='picard': policy=replace(policy,adams=AdamsRefinementConfig(max_picard_iterations=2))
    if stage=='iterations': policy=replace(policy,atol=1e-15,rtol=0,max_iterations_per_stage=1)
    result=refine_prices(p,config=policy,**ARGS)
    assert not result.converged
    assert 'cap' in result.termination_reason or 'iteration limit' in result.termination_reason
    with pytest.raises(ConvergenceError) as e: refine_prices(p,config=policy,raise_on_failure=True,**ARGS)
    assert not e.value.result.converged


@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_nonfinite_prices(monkeypatch,bad):
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.full(len(kw['K']),bad))
    with np.errstate(invalid='ignore'):
        r=refine_prices(RoughHestonPricer('pade'),**ARGS)
    assert not r.converged and r.failures
    assert not r.history[0].diagnostics[0].passed


@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_nonfinite_cf(monkeypatch,bad):
    import volcal.rough_heston.pricer.main as main
    monkeypatch.setattr(main,'rough_heston_cf_pade',lambda u,*a,**kw: np.full(u.shape,bad))
    r=refine_prices(RoughHestonPricer('pade'),**ARGS)
    assert not r.converged and 'nonfinite CF' in r.failures[0]


def test_pade_singularity_no_fallback(monkeypatch):
    import volcal.rough_heston.pricer.main as main
    def forbidden(*a,**kw): raise AssertionError('fallback')
    monkeypatch.setattr(main,'rough_heston_cf_adams',forbidden)
    r=refine_prices(RoughHestonPricer('pade',PadeConfig(5)),
                    **dict(ARGS,rough_heston_params=replace(ARGS['rough_heston_params'],H=1/6)))
    assert not r.converged and r.pade_order==5
    assert 'order=5' in r.failures[0] and 'singular' in r.failures[0]


def test_known_wing_unclipped():
    r=refine_prices(RoughHestonPricer('pade',integration_config=SinhConfig(spacing=.02,nodes=350)),
        **dict(ARGS,T=.05,K=100*np.exp(.025*.05)*1.2,
               rough_heston_params=replace(ARGS['rough_heston_params'],H=.05)))
    assert not r.converged and -3e-6<r.final_prices[0]<0
    assert 'price_bounds' in r.failures[0]


def test_no_recursion():
    import volcal.rough_heston.pricer.refinement as module
    tree=ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==node.name for n in ast.walk(node))


@pytest.mark.parametrize('kwargs',[{'atol':-1},{'rtol':np.inf},{'max_iterations_per_stage':0}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError): RefinementConfig(**kwargs)


def _synthetic(monkeypatch, value):
    def price(self, **kw):
        k=np.asarray(kw['K']); t=kw['T']; s,r,q=kw['option_params']
        calls=np.full(k.shape,value(self))
        return np.where(kw['option_type']=='call',calls,calls-s*np.exp(-q*t)+k*np.exp(-r*t))
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',price)


@pytest.mark.parametrize('dimension',['spacing','truncation'])
def test_independent_fourier_detection(monkeypatch,dimension):
    def value(p):
        f=p.integration_config
        return 5+(f.spacing if dimension=='spacing' else 1/(f.spacing*f.nodes))
    _synthetic(monkeypatch,value)
    policy=RefinementConfig(atol=.001,rtol=0,max_iterations_per_stage=1)
    r=refine_prices(RoughHestonPricer('pade'),config=policy,**dict(ARGS,K=[100]))
    assert not r.converged
    assert r.history[-1].stage=='fourier_'+dimension
    assert r.history[-1].absolute_change[0]>.001


def test_tiny_negatives_retained_in_refinement(monkeypatch):
    _synthetic(monkeypatch,lambda p: -1e-10)
    r=refine_prices(RoughHestonPricer('pade'),**dict(ARGS,K=[120]))
    assert r.converged and r.final_prices==(-1e-10,)
    assert any('tiny negative' in w for w in r.warnings)


def test_parity_failure_in_refinement(monkeypatch):
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.array([10.,10.]))
    r=refine_prices(RoughHestonPricer('pade'),**dict(ARGS,K=[100]))
    assert not r.converged and 'put_call_parity' in r.failures[0]


def test_rational_pole_message_preserved(monkeypatch):
    import volcal.rough_heston.pricer.main as main
    def pole(*args,**kwargs): raise FloatingPointError('Padé order=4: rational pole/near pole at t**alpha=0.1')
    monkeypatch.setattr(main,'rough_heston_cf_pade',pole)
    r=refine_prices(RoughHestonPricer('pade'),**ARGS)
    assert not r.converged and 'rational pole/near pole' in r.failures[0]


def test_final_probe_rechecks_earlier_dimension(monkeypatch):
    _synthetic(monkeypatch,lambda p: 5+p.integration_config.spacing*(p.integration_config.spacing*p.integration_config.nodes)**4*1e-4)
    policy=RefinementConfig(atol=.1,rtol=0,fourier=FourierRefinementConfig(truncation_increment=10))
    r=refine_prices(RoughHestonPricer('pade'),config=policy,**dict(ARGS,K=[100]))
    assert any(s.stage=='validation/fourier_spacing' for s in r.history)
