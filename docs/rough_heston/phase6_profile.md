# Phase 6: profile calibration of H

`examples/rough_heston/profile_h.py` profiles the Phase-5 sparse synthetic
surface with truth `(H,kappa,theta,sigma,v0,rho) =
(0.20,1.5,0.04,0.4,0.04,-0.7)`. The default grid has 29 points from 0.02 to
0.30 inclusive. Calls and puts at the same strikes are parity-linked: twelve
quotes supply only six independent prices at two maturities.

## Reproduction

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python examples/rough_heston/profile_h.py \
  --output /tmp/rough_heston_profile.json \
  > /tmp/rough_heston_profile_summary.txt \
  2> /tmp/rough_heston_profile_progress.txt
```

Use `--h-min`, `--h-max`, `--h-step` for a different grid. The range must contain
an integral number of steps; endpoints are included. H must lie in `(0,0.5]`.
A single-point grid is allowed. Optional controls: `--seed`, `--de-maxiter`,
`--de-popsize`, `--max-nfev`, `--tolerance`, `--adams-steps`.
Invalid grids and optimizer controls are rejected before quote generation.

`--plot-dir /tmp/rough_heston_profile_plots` writes objective, Adams IV RMSE,
and five parameter-path plots with the true H marked. This requires separately
installed matplotlib; it is not a runtime dependency. The numerical experiment,
JSON and text summary do not require plotting. Plotting uses the Agg backend.

## Reuse and optimization

The shared synthetic helper is moved without numerical changes from the Phase-5
example to `calibrator/synthetic.py`. The Phase-5 example continues to import it.
`CalibrationObjective.residuals` exposes the exact same residuals, price checks,
vega floor, penalties and evaluation ledger as the scalar objective. Existing
DE/L-BFGS-B calibration behavior is unchanged. Both residual and scalar paths
use the existing grouped pricing function. Post-fit financial checks reuse the
Phase-5 repricing routine.

At each endpoint, DE uses population multiplier 8 and 40 generations, with
seeds 42 (low H) and 43 (high H), serial immediate updating and no polishing.
Both its result and an independent nuisance-bounds midpoint receive bounded
trust-region least-squares refinement. Forward and backward continuation then
use only the previous solution from their own direction. The two sweeps do not
share a start. The truth is never supplied as an optimizer start.

Only `(kappa,theta,sigma,v0,rho)` enter either optimizer. Their bounds are exactly
Phase 5's `[(.3,3),(.01,.10),(.1,.8),(.01,.10),(-.9,-.1)]`. The inactive H bound
is extended from Phase 5's 0.03 lower bound to accommodate fixed H=0.02. Other
custom grids similarly extend only the inactive H interval as necessary.

Least squares uses bounds-width parameter scaling, residual multiplier 10,000,
`ftol=xtol=gtol=1e-9` and `max_nfev=200`. Multiplying every residual by a constant
does not change the minimizer. Reported objectives remain the original mean
squared, vega-normalized price residual, not SciPy's scaled half-sum of squares.
Actual objective evaluation counts include finite-difference probes, endpoint
checks and DE, which SciPy's local `nfev` can omit.

Within each H, select the lowest objective among converged local endpoints.
If none converged, retain the lowest valid endpoint and explicitly report
failure to converge. Finite-difference probes in the objective ledger are not
presented as converged solutions. Raw local attempts and global statuses are
retained. DE may exhaust its budget while local refinement converges; both
statuses remain visible. This is a numerical profile estimate, not a certified
global minimum. Opposite sweeps agreeing is useful but cannot exclude a shared
local basin.

## Output and interpretation

JSON retains both directions, parameters, statuses/messages, objective evaluation
counts, numerical failure examples and runtimes. It separately reports Padé and
Adams objective, price RMSE/max error and actual inverted-IV RMSE/MAE/max error.
One volatility basis point is `1e-4` decimal volatility. Missing IV inversions
produce null IV summary metrics and explicit comparison counts, never an RMSE
silently calculated on a subset. Price/financial failures remain explicit.

Every selected fit is independently repriced with Adams-1600. The market is
also Adams-1600, with Adams-3200 reference prices checked at the truth. Padé-4
and Adams share the fixed sinh grid (spacing .05, nodes 120); the truth receives
a Phase-5 Fourier-refinement preflight. This does not certify Adams time/Fourier
convergence at every fitted parameter. Low-H or boundary fits can require more
resolution; use `--adams-steps` for sensitivity experiments. Adams repricing is
validation of the Padé-selected fit, **not an Adams-optimized profile**.

The output reports objective excess over the grid minimum. Relative excess is
null when the minimum is <=1e-12; ratios against near-zero synthetic losses
are not informative. The summary reports H sets satisfying Adams IV RMSE <=0.5
and <=1.0 bp and Adams maximum IV error <=1.0 bp, rather than assuming those
sets are contiguous intervals. True-H statistics are null if H=0.20 is absent.

Directional loss gaps are separate from objective excess. Disagreement is
flagged when the difference of square-root-loss proxies exceeds 0.1 vol bp,
or when the loss gap exceeds both 1e-12 and 10% of the larger loss. These
square-root proxies are solely optimizer diagnostics, not measured IV RMSE.
Inspect convergence flags and raw directional solutions even below these
heuristic thresholds.

Metadata records revision, tracked-tree dirty state, timestamp, truth, quotes,
pricing controls, optimizer settings, effective bounds, seeds and versions.
Generated JSON/plots remain outside the repository. No results imply structural
non-identifiability: conclusions are conditional on this sparse surface, tested
H grid, nuisance bounds, approximation, numerical resolution and optimization.

## Validation

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 .venv/bin/python -m pytest -q
git diff --check
```

Focused tests cover exact decimal-grid endpoints/true H, invalid grids,
five-dimensional optimization, scalar/residual equivalence and failure policy,
selection/convergence semantics, missing-IV reporting, and a three-point smoke
profile with reduced budgets. No test requires a unique minimum at true H.

Recorded regression validation: **541 passed** (189.94 seconds), using the
virtualenv command above; `git diff --check` passed. The focused suite has 15
tests. Its three-point numerical smoke case uses 80 Fourier nodes, 400 Adams
steps, zero DE generations and two local function evaluations per attempt;
those intentionally reduced settings test wiring and finite diagnostics, not
optimization accuracy or numerical convergence.

## Recorded default experiment (2026-09-18)

The exact reproduction command above ran against source revision `2edf742`
with these Phase-6 changes uncommitted. The 29 selected fits and both sweeps'
local endpoints all converged; there were zero failed objective evaluations.
Both endpoint DE searches exhausted their budgets, after which local refinement
converged. All Padé/Adams financial checks passed, with 12 actual IV comparisons
per method and H. The searches used 5,659 objective evaluations; summed per-H
search/validation runtime was 636.65 seconds on this machine (quote generation
and preflight excluded, other tests running concurrently).

| H | Padé objective | Padé IV RMSE bp | Adams IV RMSE bp | Adams max IV error bp |
|---:|---:|---:|---:|---:|
| 0.02 | 1.23746e-11 | 0.03518 | 1.87286 | 4.16869 |
| 0.10 | 5.93913e-11 | 0.07707 | 0.60218 | 1.34472 |
| 0.12 | 4.36730e-11 | 0.06609 | 0.47613 | 1.07621 |
| 0.19 | 3.06116e-12 | 0.01750 | **0.22348** | 0.39136 |
| **0.20** | **9.22623e-13** | **0.00961** | **0.22550** | **0.35553** |
| 0.21 | **2.35513e-14** | 0.00153 | 0.24057 | 0.39223 |
| 0.27 | 2.49929e-11 | 0.04999 | 0.49900 | 0.71476 |
| 0.30 | 6.00878e-11 | 0.07752 | 0.68937 | 0.95793 |

The raw objective minimum is at H=0.21; minimum Adams IV RMSE among these
**Padé-fitted parameters** is at H=0.19. Every Padé IV RMSE is below 0.091 bp.
Relative objective excesses are deliberately null since the minimum is nearly
zero. At true H, Padé price RMSE is `3.01760e-5`; Adams price RMSE is
`5.31948e-4` and maximum price error `1.08385e-3` (spot-price units).

The following sets are contiguous **on the sampled grid**, not confidence
intervals or claims about unsampled H values:

| Adams criterion | Sampled H range, step 0.01 |
|---|---|
| IV RMSE <=0.5 bp | 0.12–0.27 |
| IV RMSE <=1.0 bp | 0.06–0.30 |
| Maximum IV error <=1.0 bp | 0.13–0.30 |

The largest forward/backward objective gap was `1.59891e-20`; no points were
flagged for material disagreement. The largest absolute nuisance-parameter
difference between directions was `1.45650e-7`. Agreement is strong evidence
against directional hysteresis in this run, but not proof of global optimality.

The selected parameter paths exhibit compensation in this experiment:

| H | kappa | theta | sigma | v0 | rho |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.350765 | 0.038360 | 0.233250 | 0.040285 | -0.704704 |
| 0.10 | 0.807703 | 0.039524 | 0.301169 | 0.040179 | -0.700971 |
| 0.20 | 1.498957 | 0.039998 | 0.400219 | 0.040005 | -0.700079 |
| 0.30 | 2.347211 | 0.040234 | 0.518728 | 0.039805 | -0.699816 |

Across all 29 selected points, kappa, theta, sigma and rho increase monotonically,
while v0 decreases. These are observed fitted paths, not a theoretical
compensation law or a relation guaranteed for other quote surfaces.

The profile is **not mathematically flat**: there is a resolved minimum near
true H. Nevertheless, economically small errors persist over a broad H range,
even under independent Adams repricing. This supports **practical weak
identification at the stated error tolerances on this sparse synthetic surface**.
Earlier free-H variation can therefore involve optimizer sensitivity on a
shallow objective; a resolved minimum and weak practical identification are
compatible. Padé approximation materially changes the apparent error scale and
minimum, particularly at low H, and must not be treated as equivalent to Adams.
The evidence does not establish structural non-identifiability.

### Numerical resolution and artifacts

Truth preflight Fourier price changes were `2.15384e-9` and `1.42109e-14` at
T=0.25/0.75. Adams truth prices changed by at most `4.61241e-7` on doubling
1600 to 3200 time steps.

As an additional post-fit check, all 29 **stored fitted parameters** were
repriced at Adams-3200 against the original, unchanged Adams-1600 market
quotes. Every financial check and IV inversion passed. The largest
1600-to-3200 price change was `2.98886e-6`; IV RMSE changed by at most
`0.000996 bp`, and maximum IV error by at most `0.002417 bp`. All three
threshold H sets were unchanged. The minimum repriced IV RMSE remained at
H=0.19 (`0.223668 bp`). At true H the refined IV RMSE was `0.225636 bp`,
with maximum IV error `0.355592 bp`. These observed changes are much smaller
than the Padé/Adams discrepancy; they are empirical stabilization checks,
not rigorous error bounds. Full Fourier refinement at every fit remains outside
this experiment. Supplemental local Adams optimization is reported separately
below; it does not replace this Padé-optimized profile.

Local generated artifacts (not committed):

- `/tmp/rough_heston_profile.json`: full metadata and all directional results.
- `/tmp/rough_heston_profile_summary.txt`: complete table and threshold sets.
- `/tmp/rough_heston_profile_plots/`: three headless PNG plots, generated using
  the example's `plots` function on the saved JSON.
- `/tmp/rough_heston_profile_adams_refinement.json`: additional Adams-3200 checks.
- `/tmp/check_rough_profile_resolution.py`: the supplemental check script; run
  with `OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python`.

## Supplemental Adams-refined validation points

The original experiment above is **(i) a Padé-optimized profile with Adams
repricing at unchanged fitted parameters**. The supplemental experiment is
**(ii) Adams-refined validation points**: at H in
`{0.06, 0.12, 0.19, 0.20, 0.21, 0.27, 0.30}`, initialize from the stored
Padé-profile optimum and locally optimize only the five nuisance parameters
using the Adams objective, with H fixed. This is not a global Adams search or
a replacement profiling methodology.

The separate `examples/rough_heston/validate_profile_adams.py` reads the original
stored quotes, calibration bounds, objective settings, and Adams-1600/sinh
configuration. It uses the existing bounded least-squares helper with
`max_nfev=200`, `ftol=xtol=gtol=1e-9`, bounds-width scaling and residual
multiplier 10,000. Objectives are reported unscaled; IV metrics come from
actual IV inversions against the unchanged market quotes. No truth parameters
are provided to the optimizer.

At H=0.12, 0.20 and 0.27, four additional local fits start independently from
uniform draws over the full five-dimensional nuisance box. Seeds are
601–604, each combined with `round(100*H)` through NumPy `SeedSequence`.
These starts do not use continuation, the Padé fit, or another restart's
endpoint. All starts and returned endpoints, convergence diagnostics, failed
evaluation counts, and pre/post Adams objective and IV RMSE/MAE/max error are
retained separately. Agreement can test for alternative basins reached by these
starts, but cannot certify a global minimum.

Reproduce after the original profile command (the output must be a new file):

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python examples/rough_heston/validate_profile_adams.py \
  --profile /tmp/rough_heston_profile.json \
  --output /tmp/rough_heston_profile_adams_local.json \
  > /tmp/rough_heston_profile_adams_local.txt 2>&1
```

The separate JSON includes the input profile's SHA-256 and original metadata;
the input and previous repricing artifacts are preserved. Each completed
attempt is checkpointed so that incomplete runs are visible.

### Recorded local validation (2026-09-18)

All 19 attempts (seven Padé-seeded fits plus twelve independent restarts)
converged. There were zero failed objective evaluations; pre/post financial
checks passed and all 12 IV comparisons were available for every attempt.
The local searches used 1,108 objective evaluations and 919.87 seconds summed
optimizer runtime (excluding pre/post repricing; tests ran concurrently).

Each entry below is **before → after Adams local refinement**, starting from
the original Padé optimum. Both objectives use Adams; all IV errors are in vol bp.

| H | Adams objective, pre → post | IV RMSE, pre → post | IV MAE, pre → post | Max IV error, pre → post |
|---:|---:|---:|---:|---:|
| 0.06 | 9.66178e-09 → 6.02291e-11 | 0.983876 → 0.0776076 | 0.658997 → 0.0614927 | 2.1757 → 0.151349 |
| 0.12 | 2.26476e-09 → 2.15592e-11 | 0.476128 → 0.046432 | 0.309978 → 0.0366906 | 1.07621 → 0.0906179 |
| 0.19 | 4.99333e-10 → 3.79512e-13 | 0.223483 → 0.00616046 | 0.18107 → 0.00485081 | 0.391359 → 0.0120338 |
| 0.20 | 5.08425e-10 → 1.35669e-30 | 0.225497 → 1.23907e-11 | 0.199117 → 9.69132e-12 | 0.355526 → 1.9984e-11 |
| 0.21 | 5.787e-10 → 3.93713e-13 | 0.240569 → 0.00627466 | 0.21957 → 0.00493527 | 0.39223 → 0.0122602 |
| 0.27 | 2.48996e-09 → 2.17128e-11 | 0.498996 → 0.046597 | 0.444644 → 0.0365222 | 0.714761 → 0.0911215 |
| 0.30 | 4.75229e-09 → 4.72407e-11 | 0.689365 → 0.0687317 | 0.62475 → 0.0537683 | 0.957927 → 0.134463 |

Independent restart results (four per H; parameter spread includes the
Padé-seeded endpoint and is the largest coordinate range in raw parameter units):

| H | Restart Adams objective range | Restart IV RMSE range, bp | Largest parameter spread |
|---:|---:|---:|---:|
| 0.12 | 2.15592175139e-11–2.15592175166e-11 | 0.0464319610572–0.0464319610615 | 1.05e-7 |
| 0.20 | 3.67e-31–1.84e-30 | 8.05e-12–1.66e-11 | 1.01e-12 |
| 0.27 | 2.17128401804e-11–2.17128401884e-11 | 0.0465969807639–0.0465969807709 | 6.35e-8 |

The JSON records pre/post objective, actual IV RMSE/MAE/max error, initial and
final parameters, and optimizer diagnostics for **every** restart, not only
its best result. No alternative endpoint was found by these independent starts.
This reduces concern that continuation alone produced the observed branch,
but finite local restarts cannot exclude other basins. These are Adams restarts;
they do not certify global optimality of the original Padé profile.

At true H=0.20, Adams refinement recovers the generating nuisance parameters
and effectively zero error at this shared numerical resolution. The original
0.2255 bp Adams-repriced RMSE at the Padé optimum is therefore attributable to
using the Padé optimization objective in this experiment, rather than an
irreducible model fitting error. The selected Adams-refined minimum is at
H=0.20, whereas the original Padé objective minimum was at H=0.21 and its
best unchanged-parameter Adams repricing was at H=0.19.

Nevertheless, all seven tested H values have Adams-refined IV RMSE below
0.078 bp and maximum IV error below 0.152 bp. Small errors away from true H
persist when Adams is used during optimization: practical weak identification
on this sparse synthetic surface is not merely a Padé artifact. This is
compatible with a resolved minimum at true H; it does not establish structural
non-identifiability or a continuous H interval from seven selected points.

These new endpoints use Adams-1600 and the original fixed Fourier grid.
The earlier Adams-3200 stabilization check applies to the **original Padé
endpoints**, not these Adams-refined endpoints. Exact recovery at true H uses
the same numerical pricer as the synthetic market and does not certify
continuous-time accuracy. No full Adams profile, global search, or new
Fourier/time-resolution study is implied.

Regression validation after adding the separate validation script: **541 passed**
(204.03 seconds). `git diff --check` passed. The original profile JSON hash was
verified unchanged. Existing Phase-6 implementation and results are preserved;
no commit was made.
