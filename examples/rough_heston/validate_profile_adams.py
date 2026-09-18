"""Supplemental fixed-H Adams local validation of a saved Padé profile.

Uses stored quotes/configuration verbatim; never runs a global optimizer or
updates the input profile. Each independent restart samples the nuisance box.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy

from volcal.rough_heston.calibrator.config import CalibrationConfig, ObjectiveConfig
from volcal.rough_heston.calibrator.loss import CalibrationObjective, PreparedQuotes
from volcal.rough_heston.calibrator.pipeline import _reprice
from volcal.rough_heston.calibrator.profile import _local, financial_metrics, fixed_config
from volcal.rough_heston.params import RoughHestonParams
from volcal.rough_heston.pricer import AdamsConfig, RoughHestonPricer, SinhConfig


SELECTED_H = (.06, .12, .19, .20, .21, .27, .30)
RESTART_H = (.12, .20, .27)


def validate(source, *, max_nfev=200, tolerance=1e-9, checkpoint=None):
    metadata = source['metadata']
    quotes = PreparedQuotes(**metadata['quotes'])
    controls = dict(metadata['calibration_config'])
    controls['objective'] = ObjectiveConfig(**controls['objective'])
    config = CalibrationConfig(**controls)
    pricing = metadata['pricing']['adams']
    adams = RoughHestonPricer('adams', AdamsConfig(**pricing['cf_config']),
                            SinhConfig(**pricing['integration_config']))
    rows = {round(row['H'], 12): row for row in source['profile']}
    if any(h not in rows for h in SELECTED_H):
        raise ValueError('input profile must contain every selected H')
    result = dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        method='fixed-H bounded local Adams least squares; no global search',
        source_metadata=metadata, max_nfev=max_nfev, tolerance=tolerance,
        residual_multiplier=1e4, restart_seeds=[601, 602, 603, 604],
        restart_distribution='independent uniform nuisance-bound draws, SeedSequence([seed, round(100*H)])',
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)),
        points=[])
    for h in SELECTED_H:
        fixed = fixed_config(h, config)
        bounds = np.asarray(fixed.active_bounds)
        starts = [('pade_profile', None, RoughHestonParams(**rows[h]['params']).to_vector()[1:])]
        if h in RESTART_H:
            for seed in result['metadata']['restart_seeds']:
                rng = np.random.default_rng(np.random.SeedSequence([seed, round(100*h)]))
                starts.append(('independent_restart', seed,
                               rng.uniform(bounds[:, 0], bounds[:, 1])))
        point = dict(H=h, attempts=[])
        result['points'].append(point)
        for kind, seed, start in starts:
            objective = CalibrationObjective(quotes, adams, fixed)
            before = financial_metrics(_reprice(objective, fixed.parameters(start), adams), quotes)
            endpoint = _local(objective, start, max_nfev, tolerance)
            after = financial_metrics(_reprice(objective, RoughHestonParams(**endpoint['params']), adams), quotes)
            point['attempts'].append(dict(kind=kind, seed=seed,
                initial_params=asdict(fixed.parameters(start)), pre_adams=before,
                post_adams=after, optimizer=endpoint,
                failed_evaluations=len(objective.failures),
                failure_examples=[asdict(f) for f in objective.failures[:3]]))
            print(f'H={h:.2f} {kind} seed={seed}: loss {before["objective"]:.8g} -> '
                  f'{after["objective"]:.8g}, IV RMSE {after["iv_rmse_bps"]} bp, '
                  f'success={endpoint["success"]}', flush=True)
            if checkpoint is not None:
                checkpoint(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path('/tmp/rough_heston_profile.json'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/rough_heston_profile_adams_local.json'))
    args = parser.parse_args()
    if args.profile.resolve() == args.output.resolve() or args.output.exists():
        parser.error('output must be a new file, distinct from the input profile')
    raw = args.profile.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(result):
        result['metadata'].update(source_profile=str(args.profile.resolve()),
                                  source_sha256=hashlib.sha256(raw).hexdigest())
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    validate(json.loads(raw), checkpoint=checkpoint)


if __name__ == '__main__':
    main()
