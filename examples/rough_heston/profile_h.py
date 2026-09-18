"""Phase 6: reproducible fixed-H profile of the Phase-5 Adams synthetic surface.

Run with PYTHONPATH=src and BLAS/Numba thread limits; see phase6_profile.md.
Default: DE (40 generations, population multiplier 8) at both endpoints,
independent midpoint local checks there, then two bounded least-squares sweeps.
No true nuisance parameters are used as optimizer starting points. See the
profile module for selection, failure and directional-disagreement policies.
Matplotlib is optional and only imported with --plot-dir; use Agg headlessly.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
from volcal.rough_heston.calibrator import CalibrationConfig
from volcal.rough_heston.calibrator.profile import h_grid, profile_h
from volcal.rough_heston.calibrator.synthetic import TRUE_PARAMS, synthetic_quotes
from volcal.rough_heston.pricer import (RoughHestonPricer, PadeConfig, AdamsConfig,
    SinhConfig, RefinementConfig, FourierRefinementConfig, refine_prices)


def summarize(result):
    rows = result['profile']
    best = min(rows, key=lambda r: r['objective'])
    valid = [r for r in rows if r['adams_iv_rmse_bps'] is not None]
    thresholds = {}
    for metric, threshold in [('adams_iv_rmse_bps', .5), ('adams_iv_rmse_bps', 1.),
                              ('adams_max_iv_error_bps', 1.)]:
        thresholds[f'{metric} <= {threshold}'] = [r['H'] for r in rows
            if r[metric] is not None and r[metric] <= threshold]
    truth = next((r for r in rows if abs(r['H']-TRUE_PARAMS.H) < 1e-12), None)
    return dict(best_objective_H=best['H'],
        minimum_adams_iv_rmse_H=min(valid, key=lambda r: r['adams_iv_rmse_bps'])['H'] if valid else None,
        true_H=None if truth is None else {k: truth[k] for k in
            ('H', 'objective', 'pade_iv_rmse_bps', 'adams_iv_rmse_bps', 'adams_max_iv_error_bps')},
        threshold_H_sets=thresholds,
        directional_disagreement_H=[r['H'] for r in rows if r['directional_disagreement']],
        unconverged_H=[r['H'] for r in rows if not r['optimizer_success']])


def print_summary(result):
    def fmt(value):
        return 'missing' if value is None else f'{value:.6g}'
    print('H      objective      delta objective  Pade IV RMSE bp  Adams IV RMSE bp  Adams max IV bp  direction gap')
    for r in result['profile']:
        print(f'{r["H"]:.3f}  ' + '  '.join(f'{fmt(r[k]):>14s}' for k in
            ('objective', 'delta_objective', 'pade_iv_rmse_bps', 'adams_iv_rmse_bps',
             'adams_max_iv_error_bps', 'directional_objective_gap')))
    print(json.dumps(result['summary'], indent=2, allow_nan=False))


def plots(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = result['profile']
    h = [r['H'] for r in rows]
    for key, label in [('objective', 'Profile objective (vega-normalized price MSE)'),
                       ('adams_iv_rmse_bps', 'Adams IV RMSE (vol bp)')]:
        fig, ax = plt.subplots()
        ax.plot(h, [r[key] for r in rows], 'o-')
        ax.axvline(TRUE_PARAMS.H, color='gray', linestyle='--', label='True H')
        ax.set(xlabel='H', ylabel=label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f'{key}.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    for ax, name in zip(axes.flat, ('kappa', 'theta', 'sigma', 'v0', 'rho')):
        ax.plot(h, [r['params'][name] for r in rows], 'o-')
        ax.axvline(TRUE_PARAMS.H, color='gray', linestyle='--')
        ax.set(xlabel='H', ylabel=name)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / 'parameter_paths.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in [('h-min', .02), ('h-max', .30), ('h-step', .01)]:
        parser.add_argument('--'+name, type=float, default=value)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--de-maxiter', type=int, default=40)
    parser.add_argument('--de-popsize', type=int, default=8)
    parser.add_argument('--max-nfev', type=int, default=200)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--adams-steps', type=int, default=1600)
    parser.add_argument('--output', type=Path, default=Path('/tmp/rough_heston_profile.json'))
    parser.add_argument('--plot-dir', type=Path)
    args = parser.parse_args()
    try:
        grid = h_grid(args.h_min, args.h_max, args.h_step)
        config = CalibrationConfig(seed=args.seed, de_maxiter=args.de_maxiter, de_popsize=args.de_popsize)
        if args.max_nfev < 1 or not np.finfo(float).eps < args.tolerance < 1:
            raise ValueError('max-nfev must be positive and epsilon < tolerance < 1')
        adams_config = AdamsConfig(args.adams_steps)
    except ValueError as exc:
        parser.error(str(exc))
    sinh = SinhConfig(spacing=.05, nodes=120)
    pade = RoughHestonPricer('pade', PadeConfig(4), sinh)
    adams = RoughHestonPricer('adams', adams_config, sinh)
    quotes = synthetic_quotes(adams)
    finer = synthetic_quotes(RoughHestonPricer('adams', AdamsConfig(2*args.adams_steps), sinh))
    preflight = []
    for (t, s, r, q), indices in quotes.groups():
        idx = list(indices)
        check = refine_prices(pade, T=t, K=np.asarray(quotes.K)[idx],
            option_params=(s, r, q), option_type=np.asarray(quotes.option_type)[idx],
            rough_heston_params=TRUE_PARAMS,
            config=RefinementConfig(atol=2e-6, rtol=1e-6,
                fourier=FourierRefinementConfig(max_nodes=2400, truncation_increment=20)))
        change = float(max(abs(np.asarray(check.final_prices)-check.history[0].prices)))
        if not check.converged or change > 2e-5:
            raise RuntimeError('fixed pricing grid failed synthetic preflight')
        preflight.append(dict(T=t, converged=check.converged, max_price_change=change))
    result = profile_h(quotes, pade, adams, grid, config=config,
        max_nfev=args.max_nfev, tolerance=args.tolerance,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True))
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=repo, text=True).strip())
    result['metadata'].update(source_git_revision=revision, tracked_worktree_dirty=dirty,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), truth=asdict(TRUE_PARAMS),
        quote_source='Adams', quotes=asdict(quotes), fixed_grid_preflight=preflight,
        adams_reference_doubled_steps_max_price_change=float(np.max(np.abs(
            np.asarray(finer.market_price)-quotes.market_price))),
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        command=sys.argv)
    result['summary'] = summarize(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print_summary(result)
    print(f'JSON: {args.output}')
    if args.plot_dir:
        plots(result, args.plot_dir)


if __name__ == '__main__':
    main()
