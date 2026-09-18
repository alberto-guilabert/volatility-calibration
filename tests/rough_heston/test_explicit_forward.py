"""Forward authority across pricing, loss, refinement and post-fit reporting."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from volcal.utils import black, black_scholes
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.calibrator import PreparedQuotes, CalibrationObjective, CalibrationConfig
from volcal.rough_heston.calibrator.loss import price_quotes, quote_iv
from volcal.rough_heston.calibrator.pipeline import _reprice, calibrate
from volcal.rough_heston.calibrator.profile import financial_metrics
from volcal.rough_heston.pricer import RoughHestonPricer, RefinementConfig, refine_prices
from volcal.rough_heston.pricer.diagnostics import no_arbitrage_bounds, put_call_parity


PARAMS = RoughHestonParams(.2, 1.5, .04, .4, .04, -.7)


@pytest.mark.parametrize('typ', ['call', 'put'])
def test_black_legacy_equivalence_and_iv_roundtrip(typ):
    t, s, r, q, sigma = .7, 100., .04, .015, .3
    f = s*np.exp((r-q)*t)
    k = np.array([80., 100., 120.])
    new = black.price(sigma, t, k, F=f, r=r, option_type=typ)
    old = black_scholes.price(sigma, t, k, (s, r, q), typ)
    np.testing.assert_allclose(new, old, atol=5e-14, rtol=0)
    np.testing.assert_allclose(black.vega(sigma, t, k, F=f, r=r),
        black_scholes.vega(sigma, t, k, (s, r, q)), atol=5e-14, rtol=0)
    for strike, price in zip(k, new):
        assert black.iv_solver(price, t, strike, F=f, r=r, option_type=typ) == pytest.approx(sigma, abs=1e-11)


def test_black_authority_parity_and_vega_derivative():
    args = dict(T=.5, K=np.array([90., 100., 110.]), F=115., r=.03)
    c = black.price(.25, **args)
    p = black.price(.25, **args, option_type='put')
    np.testing.assert_allclose(c-p, np.exp(-.03*.5)*(115-args['K']), atol=3e-14, rtol=0)
    assert np.all(c > black.price(.25, **dict(args, F=105.)))
    derivative = (black.price(.250001, **args)-black.price(.249999, **args))/2e-6
    np.testing.assert_allclose(black.vega(.25, **args), derivative, atol=2e-8, rtol=0)
    assert black.price(0, .5, 100, F=115, r=.03) == pytest.approx(15*np.exp(-.015))
    assert black.price(.2, 0, 100, F=115, r=.03) == 15
    assert black.vega(.2, 0, 100, F=115, r=.03) == 0
    with pytest.raises(ValueError, match='bounds'):
        black.iv_solver(1, .5, 100, F=115, r=.03)
    with pytest.raises(ValueError, match='bounds'):
        black.iv_solver(115*np.exp(-.015), .5, 100, F=115, r=.03)


@pytest.mark.parametrize('method', ['pade', 'adams'])
def test_rough_heston_equivalence_authority_and_parity(method):
    pricer = RoughHestonPricer(method)
    args = dict(T=.5, K=np.array([90., 100., 110.]), option_params=(100., .03, .01),
                rough_heston_params=PARAMS)
    consistent = 100*np.exp(.02*.5)
    old = pricer.vanilla_price(**args)
    new = pricer.vanilla_price(**args, F=consistent)
    np.testing.assert_allclose(new, old, atol=2e-12, rtol=0)
    c = pricer.vanilla_price(**args, F=115.)
    p = pricer.vanilla_price(**args, F=115., option_type='put')
    assert np.all(c > pricer.vanilla_price(**args, F=105.))
    assert np.max(abs(c-old)) > 5.
    # Same F and r: altering source spot/q cannot change prices.
    changed_source = pricer.vanilla_price(**dict(args, option_params=(230., .03, -.2)), F=115.)
    np.testing.assert_array_equal(changed_source, c)
    np.testing.assert_allclose(c-p, np.exp(-.03*.5)*(115-args['K']), atol=3e-14, rtol=0)
    assert put_call_parity(c, p, T=.5, K=args['K'], option_params=args['option_params'], F=115., atol=3e-14).passed
    assert not put_call_parity(c, p, T=.5, K=args['K'], option_params=args['option_params']).passed
    lo, hi = no_arbitrage_bounds(T=.5, K=args['K'], option_params=args['option_params'], F=115.)
    np.testing.assert_allclose(lo, np.exp(-.015)*np.maximum(115-args['K'], 0), atol=3e-14, rtol=0)
    assert hi == pytest.approx(115*np.exp(-.015))
    assert np.all(c >= lo-1e-8) and np.all(c <= hi+1e-8)
    assert pricer.vanilla_price(**dict(args, T=0, K=100.), F=115.) == 15.


def forward_quotes():
    # Identical legacy grouping keys, different explicit forwards.
    t, k, f, r = [.5]*4, [100.]*4, [110., 110., 115., 115.], [.03]*4
    types = ['call', 'put', 'call', 'put']
    prices = black.price(.25, t, k, F=f, r=r, option_type=types)
    return PreparedQuotes(T=t, K=k, S0=[70.]*4, r=r, q=[.2]*4, F=f,
                          option_type=types, market_price=prices, market_iv=[.25]*4)


@pytest.mark.parametrize('value', [0., [], [110.], [[110.]*4], [0.]*4,
                                    [-1.]*4, [np.nan]*4, [np.inf]*4, ['110']*4])
def test_forward_contract_rejects_invalid_columns(value):
    with pytest.raises(ValueError):
        replace(forward_quotes(), F=value)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, [110.], '110', 1j])
def test_pricer_and_diagnostics_reject_invalid_forward(value):
    args = dict(T=.5, K=100., option_params=(100., .03, .01), F=value)
    with pytest.raises(ValueError):
        RoughHestonPricer('pade').vanilla_price(**args, rough_heston_params=PARAMS)
    with pytest.raises(ValueError):
        no_arbitrage_bounds(**args)
    with pytest.raises(ValueError):
        black.iv_solver(5., .5, 100., F=value, r=.03)


def test_forward_copy_grouping_vega_and_iv_fallback():
    q = forward_quotes()
    source = list(q.F)
    copied = replace(q, F=source)
    source[0] = 200.
    assert copied.F == q.F
    assert len(q.groups()) == 2
    assert [indices for _, indices in q.groups()] == [(0, 1), (2, 3)]
    objective = CalibrationObjective(q, RoughHestonPricer('pade'))
    np.testing.assert_allclose(objective.scale, black.vega(.25, q.T, q.K, F=q.F, r=q.r))
    assert not np.allclose(objective.scale, black_scholes.vega(.25, np.array(q.T), np.array(q.K),
        (np.array(q.S0), np.array(q.r), np.array(q.q))))
    for i, price in enumerate(q.market_price):
        assert quote_iv(q, i, price) == pytest.approx(.25, abs=1e-11)
    supplied = CalibrationObjective(replace(q, market_vega=[3.]*4), RoughHestonPricer('pade'))
    np.testing.assert_array_equal(supplied.scale, [3.]*4)


@pytest.mark.parametrize('method', ['pade', 'adams'])
def test_real_pricing_repricing_and_profile_metrics(method):
    q = forward_quotes()
    pricer = RoughHestonPricer(method)
    prices = price_quotes(pricer, q, PARAMS)
    q = replace(q, market_price=prices,
                market_iv=[quote_iv(q, i, p) for i, p in enumerate(prices)])
    for with_iv in (q, replace(q, market_iv=None, market_vega=[1.]*4)):
        result = _reprice(CalibrationObjective(with_iv, pricer), PARAMS, pricer)
        assert result.passed and result.iv_comparisons == 4
        assert result.max_market_iv_error < 1e-12
        assert result.loss == 0.
        metrics = financial_metrics(result, with_iv)
        assert metrics['iv_comparisons'] == 4
        assert metrics['iv_rmse_bps'] < 1e-8


@pytest.mark.parametrize('method', ['pade', 'adams'])
def test_refinement_propagates_forward_at_every_stage(monkeypatch, method):
    seen = []
    def price(self, **kw):
        seen.append((self.cf_method, kw['F']))
        return black.price(.25, kw['T'], kw['K'], F=kw['F'], r=kw['option_params'][1],
                           option_type=kw['option_type'])
    monkeypatch.setattr(RoughHestonPricer, 'vanilla_price', price)
    result = refine_prices(RoughHestonPricer(method), T=.5, K=[100., 110., 120.],
        option_params=(70., .03, .2), F=115., rough_heston_params=PARAMS)
    assert result.converged
    assert len(seen) >= 5
    assert set(seen) == {(method, 115.)}
    assert all(d.passed for step in result.history for d in step.diagnostics)


def test_pipeline_forwards_to_adams_and_refinement_without_optimization(monkeypatch):
    import volcal.rough_heston.calibrator.pipeline as pipeline
    q = forward_quotes()
    seen, refinements = [], []
    def price(self, **kw):
        seen.append((self.cf_method, kw['F']))
        return black.price(.25, kw['T'], kw['K'], F=kw['F'], r=kw['option_params'][1],
                           option_type=kw['option_type'])
    def optimizer(objective, *args, **kwargs):
        x = PARAMS.to_vector()
        return SimpleNamespace(x=x, fun=objective(x), success=True, status=0,
                               message='test stub; no search', nfev=1, nit=0)
    def refinement(*args, **kw):
        refinements.append(kw['F'])
        return 'test sentinel'
    monkeypatch.setattr(RoughHestonPricer, 'vanilla_price', price)
    monkeypatch.setattr(pipeline, 'differential_evolution', optimizer)
    monkeypatch.setattr(pipeline, 'minimize', optimizer)
    monkeypatch.setattr(pipeline, 'refine_prices', refinement)
    result = calibrate(q, RoughHestonPricer('pade'), CalibrationConfig(),
        reference_params=PARAMS, refinement_config=RefinementConfig(),
        adams_validation_pricer=RoughHestonPricer('adams'))
    assert set(seen) == {('pade', 110.), ('pade', 115.), ('adams', 110.), ('adams', 115.)}
    assert refinements == [110., 115.]
    assert result.final_loss == 0.
    assert result.adams_validation.passed
    assert result.adams_validation.max_market_iv_error < 1e-11
