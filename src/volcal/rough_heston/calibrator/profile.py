"""Fixed-H profile using the Phase-5 objective and independent repricing.

Each endpoint is seeded by deterministic DE plus an independent midpoint local
start. Two independent continuation sweeps run in opposite directions. Local
bounded least squares uses the exact Phase-5 residuals, multiplied by 1e4 for
numerical conditioning (reported losses are unscaled). Only the five nuisance
parameters enter optimizers. Select the lowest-loss converged local endpoint;
if none converged retain the lowest valid endpoint and flag it. A finite grid
and optimizer success do not certify global minima or structural identification.
"""
from dataclasses import asdict, replace
from decimal import Decimal
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .config import CalibrationConfig
from .loss import CalibrationObjective, quote_iv
from .pipeline import _reprice
from ..params import RoughHestonParams
from volcal.utils.black_scholes import iv_solver


def h_grid(h_min=.02, h_max=.30, h_step=.01):
    """Inclusive decimal grid; reject a nonintegral number of steps."""
    if any(isinstance(x, (bool, np.bool_)) or not np.isfinite(x)
           for x in (h_min, h_max, h_step)):
        raise ValueError('H grid values must be finite real numbers')
    if not 0 < h_min <= h_max <= .5 or h_step <= 0:
        raise ValueError('require 0 < h_min <= h_max <= .5 and h_step > 0')
    lo, hi, step = map(lambda x: Decimal(str(x)), (h_min, h_max, h_step))
    count = (hi - lo) / step
    n = int(round(count))
    if abs(count - n) > Decimal('1e-10') or n > 10000:
        raise ValueError('H range must contain an integral number of steps (at most 10000)')
    return [float(lo + i * step) for i in range(n + 1)]


def fixed_config(h, config):
    # H is fixed, never searched; extend only the inactive H interval if needed.
    low, high = config.bounds[0]
    return replace(config, fixed_H=float(h),
                   bounds=((min(low, h), max(high, h)),) + config.bounds[1:])


def financial_metrics(repricing, quotes):
    """Actual IV inversions; incomplete IV coverage never yields partial RMSE."""
    out = dict(objective=repricing.loss, financial_checks_passed=repricing.passed,
               failure=repricing.failure, iv_comparisons=0, price_rmse=None,
               max_price_error=None, iv_rmse_bps=None, iv_mae_bps=None,
               max_iv_error_bps=None)
    if not repricing.prices:
        return out
    error = np.asarray(repricing.prices) - quotes.market_price
    out.update(price_rmse=float(np.sqrt(np.mean(error**2))),
               max_price_error=float(np.max(np.abs(error))))
    errors = []
    for i, price in enumerate(repricing.prices):
        try:
            if quotes.F is None:
                args = (quotes.T[i], quotes.K[i],
                        (quotes.S0[i], quotes.r[i], quotes.q[i]), quotes.option_type[i])
                market = quotes.market_iv[i] if quotes.market_iv is not None else iv_solver(quotes.market_price[i], *args)
                model_iv = iv_solver(price, *args)
            else:
                market = quotes.market_iv[i] if quotes.market_iv is not None else quote_iv(quotes, i, quotes.market_price[i])
                model_iv = quote_iv(quotes, i, price)
            error = (model_iv - market) * 1e4
            if np.isfinite(error):
                errors.append(error)
        except (ValueError, FloatingPointError, OverflowError):
            pass
    out['iv_comparisons'] = len(errors)
    if len(errors) == len(quotes.T):
        errors = np.asarray(errors)
        out.update(iv_rmse_bps=float(np.sqrt(np.mean(errors**2))),
                   iv_mae_bps=float(np.mean(np.abs(errors))),
                   max_iv_error_bps=float(np.max(np.abs(errors))))
    return out


def _local(objective, start, max_nfev, tolerance):
    before, begin = objective.evaluations, perf_counter()
    bounds = np.asarray(objective.config.active_bounds)
    result = least_squares(lambda x: 1e4 * objective.residuals(x), start,
        bounds=(bounds[:, 0], bounds[:, 1]), x_scale=bounds[:, 1] - bounds[:, 0],
        max_nfev=max_nfev, ftol=tolerance, xtol=tolerance, gtol=tolerance)
    # Re-evaluate the returned point: finite-difference probes in the ledger
    # must not be mislabelled as successful optimizer endpoints.
    failures = len(objective.failures)
    loss = objective(result.x)
    valid = len(objective.failures) == failures
    return dict(params=asdict(objective.config.parameters(result.x)),
        objective=loss if valid else None, success=bool(result.success and valid),
        valid=valid, status=int(result.status), message=str(result.message),
        optimality=float(result.optimality), scipy_nfev=int(result.nfev),
        objective_evaluations=objective.evaluations-before,
        runtime_seconds=perf_counter()-begin)


def _select(attempts):
    valid = [a for a in attempts if a['valid']]
    if not valid:
        raise RuntimeError('no valid local profile endpoint')
    converged = [a for a in valid if a['success']]
    return min(converged or valid, key=lambda a: a['objective'])


def profile_h(quotes, pade, adams, grid, *, config=CalibrationConfig(de_maxiter=40),
              max_nfev=200, tolerance=1e-9, progress=None):
    """Return JSON-compatible profile, directional attempts and configurations.

    Runtime/evaluation counts per H aggregate both directional searches and DE
    when applicable; repricing is in runtime but not objective evaluation counts.
    Directional disagreement is flagged at 0.1 bp difference between sqrt(loss)
    proxies OR 10% relative loss with absolute gap > 1e-12. These proxies are
    optimizer diagnostics, never labelled actual IV errors.
    """
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or
            np.any(grid <= 0) or np.any(grid > .5) or np.any(np.diff(grid) <= 0)):
        raise ValueError('grid must be finite, strictly increasing and within (0,.5]')
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError('max_nfev must be a positive integer')
    if not np.isfinite(tolerance) or not np.finfo(float).eps < tolerance < 1:
        raise ValueError('tolerance must be between machine epsilon and one')
    if pade.cf_method != 'pade' or adams.cf_method != 'adams':
        raise ValueError('profile requires explicit Pade and independent Adams pricers')
    directions = {}
    for direction, values, seed in [('forward', grid, config.seed),
                                     ('backward', grid[::-1], config.seed + 1)]:
        sweep, previous = {}, None
        for h in values:
            begin = perf_counter()
            objective = CalibrationObjective(quotes, pade, fixed_config(h, config))
            attempts, global_status = [], None
            if previous is None:
                de = differential_evolution(objective, objective.config.active_bounds,
                    seed=seed, popsize=config.de_popsize, maxiter=config.de_maxiter,
                    tol=config.de_tol, polish=False, workers=1, updating='immediate')
                global_status = dict(seed=seed, success=bool(de.success), message=str(de.message),
                                     objective=float(de.fun), nfev=int(de.nfev), nit=int(de.nit))
                starts = [de.x, np.mean(objective.config.active_bounds, axis=1)]
            else:
                starts = [RoughHestonParams(**previous['params']).to_vector()[1:]]
            for start in starts:
                attempts.append(_local(objective, start, max_nfev, tolerance))
            selected = _select(attempts)
            previous = selected
            sweep[float(h)] = dict(selected=selected, attempts=attempts, global_search=global_status,
                objective_evaluations=objective.evaluations, failed_evaluations=len(objective.failures),
                failure_examples=[asdict(f) for f in objective.failures[:3]],
                runtime_seconds=perf_counter()-begin)
            if progress:
                progress(f'{direction:8s} H={h:.4f} loss={selected["objective"]:.6g} success={selected["success"]}')
        directions[direction] = sweep
    rows = []
    for h in grid:
        begin = perf_counter()
        f, b = (directions[d][float(h)] for d in ('forward', 'backward'))
        chosen = _select([f['selected'], b['selected']])
        params = RoughHestonParams(**chosen['params'])
        objective = CalibrationObjective(quotes, pade, fixed_config(h, config))
        p = _reprice(objective, params, pade)
        a = _reprice(objective, params, adams, np.asarray(p.prices) if p.prices else None)
        fl, bl = f['selected']['objective'], b['selected']['objective']
        gap = abs(fl-bl)
        material = (abs(np.sqrt(fl)-np.sqrt(bl))*1e4 > .1 or
                    (gap > 1e-12 and gap > .1*max(fl, bl)))
        row = dict(H=float(h), params=chosen['params'], objective=chosen['objective'],
            optimizer_success=chosen['success'], optimizer_status=chosen['status'],
            optimizer_message=chosen['message'], selected_direction='forward' if chosen is f['selected'] else 'backward',
            objective_evaluations=f['objective_evaluations']+b['objective_evaluations'],
            runtime_seconds=f['runtime_seconds']+b['runtime_seconds']+perf_counter()-begin,
            forward=f, backward=b, directional_objective_gap=gap,
            directional_disagreement=bool(material),
            max_pade_adams_price_difference=a.max_calibration_price_difference)
        for name, result in [('pade', p), ('adams', a)]:
            row.update({f'{name}_{key}': value for key, value in financial_metrics(result, quotes).items()})
        rows.append(row)
        if progress:
            progress(f'validate H={h:.4f} Adams IV RMSE={row["adams_iv_rmse_bps"]} bp')
    minimum = min(r['objective'] for r in rows)
    for row in rows:
        row['delta_objective'] = row['objective']-minimum
        row['relative_excess_objective'] = row['delta_objective']/minimum if minimum > 1e-12 else None
    return dict(metadata=dict(h_grid=grid.tolist(), calibration_config=asdict(config),
        effective_bounds=[list(pair) for pair in fixed_config(float(grid[0]),
            fixed_config(float(grid[-1]), config)).bounds],
        optimizer=dict(method='two endpoint DE + bounded least-squares, bidirectional continuation',
            max_nfev=max_nfev, tolerance=tolerance, residual_multiplier=1e4,
            seeds=[config.seed, config.seed+1], relative_objective_floor=1e-12,
            selection='lowest converged local endpoint; lowest valid if none converged',
            disagreement_rule='sqrt(loss) proxy gap > 0.1 bp OR loss gap > 1e-12 and >10% of larger loss'),
        pricing=dict(pade=asdict(pade), adams=asdict(adams))), profile=rows)
