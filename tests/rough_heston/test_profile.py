"""Small profiles only: no dense production search in CI."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from volcal.rough_heston.calibrator import CalibrationConfig, CalibrationObjective
from volcal.rough_heston.calibrator.profile import h_grid, profile_h, fixed_config, financial_metrics, _select
from volcal.rough_heston.calibrator.synthetic import synthetic_quotes, TRUE_PARAMS
from volcal.rough_heston.pricer import RoughHestonPricer, PadeConfig, AdamsConfig, SinhConfig
from volcal.rough_heston.calibrator.result import RepricingResult

PADE = RoughHestonPricer('pade', PadeConfig(4), SinhConfig(spacing=.05, nodes=120))


def test_grid():
    grid = h_grid()
    assert len(grid) == 29
    assert grid[0] == .02 and grid[-1] == .30
    assert .20 in grid
    assert h_grid(.15, .25, .05) == [.15, .20, .25]
    assert h_grid(.2, .2, .01) == [.2]


@pytest.mark.parametrize('args', [(0, .3, .01), (.3, .2, .01), (.1, .6, .01),
    (.1, .3, 0), (.1, .3, -.01), (.1, .3, .03), (.1, np.nan, .01),
    (.1, .3, np.inf), (False, .3, .01), (.1, .3, 1e-8)])
def test_invalid_grid(args):
    with pytest.raises(ValueError):
        h_grid(*args)


def test_fixed_config_and_residual_equivalence():
    config = fixed_config(.02, CalibrationConfig())
    assert config.active_bounds == CalibrationConfig().bounds[1:]
    assert len(config.active_bounds) == 5
    vector = TRUE_PARAMS.to_vector()[1:]
    assert config.parameters(vector).H == .02
    with pytest.raises(ValueError):
        config.parameters(TRUE_PARAMS.to_vector())
    quotes = synthetic_quotes(PADE)
    objective = CalibrationObjective(quotes, PADE, config)
    assert np.mean(objective.residuals(vector)**2) == pytest.approx(objective(vector), rel=1e-14)
    assert objective.evaluations == 2
    invalid = vector.copy()
    invalid[0] = -1
    assert np.mean(objective.residuals(invalid)**2) == config.failure_penalty
    assert len(objective.failures) == 1


def test_selection_prefers_converged_and_preserves_failure():
    good = dict(valid=True, success=True, objective=2.)
    incomplete = dict(valid=True, success=False, objective=1.)
    assert _select([good, incomplete]) is good
    assert _select([incomplete]) is incomplete
    with pytest.raises(RuntimeError):
        _select([dict(valid=False, success=False)])


def test_missing_iv_coverage_is_explicit():
    quotes = synthetic_quotes(PADE)
    result = RepricingResult(PADE, prices=quotes.market_price, loss=0.)
    with patch('volcal.rough_heston.calibrator.profile.iv_solver', return_value=np.nan):
        metrics = financial_metrics(result, quotes)
    assert metrics['iv_comparisons'] == 0
    assert metrics['iv_rmse_bps'] is None
    assert metrics['price_rmse'] == 0.
    json.dumps(metrics, allow_nan=False)


def test_small_profile():
    from volcal.rough_heston.calibrator import profile as module
    # Smaller Fourier range avoids high-frequency Adams instability at the
    # deliberately under-optimized, broad-bound smoke-test parameters.
    pade = RoughHestonPricer('pade', PadeConfig(4), SinhConfig(spacing=.05, nodes=80))
    quotes = synthetic_quotes(pade)
    adams = RoughHestonPricer('adams', AdamsConfig(400), pade.integration_config)
    original_local, original_de = module.least_squares, module.differential_evolution
    dimensions = []

    def local(fun, x0, **kwargs):
        dimensions.append(len(x0))
        assert len(kwargs['bounds'][0]) == 5
        return original_local(fun, x0, **kwargs)

    def global_search(fun, bounds, **kwargs):
        assert len(bounds) == 5
        return original_de(fun, bounds, **kwargs)

    with patch.object(module, 'least_squares', local), patch.object(module, 'differential_evolution', global_search):
        result = profile_h(quotes, pade, adams, [.15, .20, .25],
            config=CalibrationConfig(de_popsize=1, de_maxiter=0), max_nfev=2)
    assert dimensions == [5]*8  # two endpoint starts, two continuations, twice
    assert len(result['profile']) == 3
    for row in result['profile']:
        assert row['params']['H'] == row['H']
        assert np.isfinite(row['objective'])
        assert row['objective_evaluations'] > 0 and row['runtime_seconds'] > 0
        assert isinstance(row['optimizer_success'], bool)
        assert row['optimizer_message']
        for name in ('pade', 'adams'):
            for metric in ('objective', 'price_rmse', 'max_price_error', 'iv_rmse_bps',
                           'iv_mae_bps', 'max_iv_error_bps'):
                assert row[f'{name}_{metric}'] is not None, (name, metric, row.get(f'{name}_failure'), row['params'])
                assert np.isfinite(row[f'{name}_{metric}'])
            assert row[f'{name}_iv_comparisons'] == len(quotes.T)
        for direction in ('forward', 'backward'):
            selected = row[direction]['selected']
            assert selected['params']['H'] == row['H']
            values = [selected['params'][k] for k in ('kappa', 'theta', 'sigma', 'v0', 'rho')]
            assert all(lo <= x <= hi for x, (lo, hi) in zip(values, CalibrationConfig().active_bounds[1:]))
        assert row['delta_objective'] >= 0
    json.dumps(result, allow_nan=False)
