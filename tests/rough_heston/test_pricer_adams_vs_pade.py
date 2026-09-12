"""Price-level budgets on a synthetic grid; no monotone-order guarantee."""
from time import perf_counter
import numpy as np
import pytest
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.pricer import AdamsConfig, PadeConfig, RoughHestonPricer, SinhConfig
from volcal.utils.black_scholes import iv_solver, vega

CASES=[(.05,.05,-.7,.4),(.2,.5,-.7,.4),(.49,2,.7,.8),(.5,1,0,.1)]


@pytest.fixture(scope='module')
def reference_grid():
    rows=[]
    for H,T,rho,sigma in CASES:
        params=RoughHestonParams(H,1.5,.04,sigma,.04,rho)
        cfg=SinhConfig(spacing=.02,nodes=350 if T<.1 else 275)
        K=100*np.exp(.025*T)*np.array([.8,1,1.2])
        args=dict(T=T,K=K,option_params=(100,.04,.015),rough_heston_params=params)
        coarse=RoughHestonPricer('adams',AdamsConfig(1600),cfg).vanilla_price(**args)
        fine=RoughHestonPricer('adams',AdamsConfig(3200),cfg).vanilla_price(**args)
        change=max(abs(fine-coarse))
        print('Adams 1600->3200',H,T,rho,sigma,change)
        assert change<1e-6
        rows.append((cfg,args,fine))
    return rows


@pytest.mark.parametrize('order,budget,iv_budget',[(2,.26,.006),(3,.095,.0025),
                                                 (4,.038,.001),(5,.016,.0005)])
def test_pade_price_and_iv_errors(reference_grid,order,budget,iv_budget):
    errors=[]; iv_errors=[]
    for cfg,args,ref in reference_grid:
        values=RoughHestonPricer('pade',PadeConfig(order),cfg).vanilla_price(**args)
        error=max(abs(values-ref));errors.append(error)
        local_iv=[]
        for k,a,b in zip(args['K'],values,ref):
            # Exclude near-intrinsic wings where inversion is ill-conditioned.
            ref_iv=iv_solver(b,args['T'],k,args['option_params'],'call')
            if vega(ref_iv,args['T'],k,args['option_params'])>1:
                local_iv.append(abs(iv_solver(a,args['T'],k,args['option_params'],'call')-ref_iv))
        iv_errors.extend(local_iv)
        print('Pade',order,'H,T',args['rough_heston_params'].H,args['T'],
              'max price',error,'max IV',max(local_iv))
    assert max(errors)<budget
    assert max(iv_errors)<iv_budget


def test_rough_fourier_convergence():
    args=dict(T=.5,K=[80,100,120],option_params=(100,.04,.015),
              rough_heston_params=RoughHestonParams(.2,1.5,.04,.4,.04,-.7))
    for method,cf_cfg in [('adams',AdamsConfig(1600)),('pade',PadeConfig())]:
        def run(h,n):
            return RoughHestonPricer(method,cf_cfg,SinhConfig(spacing=h,nodes=n)).vanilla_price(**args)
        spacing=[run(h,round(6/h)) for h in [.2,.1,.05,.025]]
        truncation=[run(.025,n) for n in [160,200,240,260]]
        ds=[max(abs(a-b)) for a,b in zip(spacing,spacing[1:])]
        dt=[max(abs(a-b)) for a,b in zip(truncation,truncation[1:])]
        print(method,'Fourier spacing changes',ds,'extent changes',dt)
        assert ds[-1]<1e-8 and ds[-1]<ds[0]
        assert dt[-1]<1e-7 and dt[-1]<dt[0]


def test_known_finite_pade_wing_violation():
    # A finite-order approximation can violate positivity even after Fourier
    # stabilization. Expose the raw violation and ensure it is not clipped.
    args=dict(T=.05,K=100*np.exp(.025*.05)*1.2,option_params=(100,.04,.015),
              rough_heston_params=RoughHestonParams(.05,1.5,.04,.4,.04,-.7))
    a=RoughHestonPricer('pade',PadeConfig(4),SinhConfig(spacing=.02,nodes=350)).vanilla_price(**args)
    b=RoughHestonPricer('pade',PadeConfig(4),SinhConfig(spacing=.01,nodes=750)).vanilla_price(**args)
    print('Pade 4 finite negative wing',a,b)
    assert -3e-6<a<0 and abs(a-b)<1e-8


def test_benchmark_batched_strikes():
    args=dict(T=.5,K=np.linspace(80,120,41),option_params=(100,.04,.015),
              rough_heston_params=RoughHestonParams(.2,1.5,.04,.4,.04,-.7))
    ref=RoughHestonPricer('adams',AdamsConfig(3200)).vanilla_price(**args)
    for method,config in [('adams',AdamsConfig(1600))]+[('pade',PadeConfig(o)) for o in (2,3,4,5)]:
        pricer=RoughHestonPricer(method,config)
        pricer.vanilla_price(**args)  # Warm JIT and runtime before timing.
        durations=[]
        for _ in range(3):
            start=perf_counter();value=pricer.vanilla_price(**args)
            durations.append(perf_counter()-start)
        error=max(abs(value-ref))
        print('BENCHMARK',method,config,'median seconds',np.median(durations),'max price error',error)
        assert np.all(np.isfinite(value))
