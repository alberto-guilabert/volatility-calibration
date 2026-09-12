# Phase 2: rough-Heston Padé validation

## Baseline and scope

Started on clean `feat/rough-heston`, HEAD `cb1b7fe`, Python 3.10.11,
Numba 0.61.2, llvmlite 0.44.0. Re-ran the baseline: 212 tests passed.
The modified Adams solver and all Phase-1 numerical/interface files are
unchanged; only package exports were extended. No pricing, contour bounds,
refinement, calibration, market-data, Heston or SABR changes were made.

## API and construction

`from volcal.rough_heston import rough_heston_cf_pade`

`rough_heston_cf_pade(u, T, params, *, order=4)` supports integer orders
2, 3, 4, 5. Order 4 is the recommended/default general-purpose approximation:
it is better conditioned, faster, and more accurate than order 5 on the
reported benchmark vector. Order 5 may provide better accuracy in some
regions, but is unsupported at alpha=2/3. No automatic fallback between Padé
orders or to Adams occurs.

`params` is the immutable Phase-1 `RoughHestonParams`, ordered
(H, kappa, theta, sigma, v0, rho); alpha=H+1/2. Scalar input returns Python
complex; a one-dimensional vector returns a complex128 array, including an
empty vector. Inputs are immutable; repeated results are deterministic.
Order 6, booleans, floats used as orders, invalid maturity, nonfinite frequency
and multidimensional frequency inputs are rejected explicitly.

The order-specific modules contain the expanded legacy coefficient formulas.
The generic module evaluates their rational functions in y=t**alpha and
integrates kappa*theta*h + v0*R(u,h), preserving the Phase-1 CF convention.
Fixed 128-point Gauss-Legendre **time** quadrature uses t=T*x**4; there is no
Fourier/pricing integration. An independent adaptive time integral agrees to
2e-11 in tests. No Adams fallback, parameter perturbation, clipping of CF/path
values, adaptive refinement, caching or import-time quadrature is used.

T=0 returns 1 before coefficient evaluation. At u=0 and u=-i, R(u,0)=0, so
h=0 is the exact zero-initial-value Riccati solution and phi=1. The latter
uses the same normalized-stock martingale convention as Phase 1. Exact
shortcuts, continuity near both frequencies, real-axis conjugate symmetry,
modulus bounds on the test grid, and continuity as H approaches .5 all pass.
A finite Padé approximation is not guaranteed to be a positive-definite CF
on arbitrary grids or accurate outside the tested domain.

## Independent coefficient validation

Write P(y)=sum(p_j*y**j), Q(y)=1+sum(q_j*y**j). Small-time coefficients are
obtained independently by substituting h=sum(b_j*t**(j*alpha)) into the
fractional Riccati equation and convolving its quadratic term. The intended
formal large-time coefficients g_j are obtained from the stable Riccati root
and the inverse-power fractional-derivative recurrence, independently of the
expanded q_j expressions. Tests check the coefficients of P-Q*b through
y**n and the first n coefficients of reversed(P)-reversed(Q)*g. Thus each
order is validated against its own matching equations, never another order.

For n=2,3,4,5, the approximation matches b_1,...,b_n and g_0,...,g_(n-1).
These are **formal inverse-power matching conditions**, not a proof that a
complete large-time expansion contains only those powers, nor a uniform
accuracy theorem for the nonlinear fractional equation.

Grid: H=.05,.1,.2,.3,.49,.499999,.5; u=.001,1,10,2-.5i.
The maximum residual is scaled by the largest polynomial/convolution
coefficient or 1. Additional tests cover reciprocal-gamma zeros at alpha=2/3
(orders 2–4) and alpha=3/4 (all orders).

| Order | Maximum scaled matching residual |
|---|---:|
| 2 | 2.30e-15 |
| 3 | 3.45e-15 |
| 4 | 9.83e-14 |
| 5 | 3.25e-12 |

Both small-time and formal large-time residual assertions pass for every
supported order on this grid. Denominator conditioning is tested separately.
The approach follows the two-point construction described by
[Gatheral and Radoicic](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3191578);
legacy outputs are not used as numerical ground truth.

## Full complex CF errors

Reference grid: H=.05,.2,.49,.5; T=.01,.5,2;
(rho,sigma)=(-.7,.4),(.7,.8),(0,.1); kappa=1.5, theta=v0=.04;
u=.1,1,3,10,1-.5i,3-.5i. This is 216 complex comparisons per order.
Adams uses 3,200 steps; every 1,600-to-3,200 change is asserted below 5e-6.
Relative error is abs(Pade-Adams)/abs(Adams), meaningful on this grid.
Classical comparisons use the existing Phase-1 independent analytical
Heston oracle, itself tested against DOP853, for all nine H=.5 configurations.

| Order | Max abs vs Adams | Max relative vs Adams | Max abs vs analytical Heston at H=.5 |
|---|---:|---:|---:|
| 2 | .0494705160 | .151970621 | .0494704873 |
| 3 | .0179538829 | .0594021564 | .0179538527 |
| 4 | .00729811499 | .0250660688 | .00729808410 |
| 5 | .00316924235 | .0111320677 | .00316921111 |

H=.5 is supported as a finite-order approximation, **not exact analytical
Heston**. Its inverse-power tail coefficients vanish, since the classical
stable solution approaches its root exponentially. Orders 3 and 4 required
explicit zero evaluation at alpha=1; the legacy directly evaluated undefined
gamma ratios there. Near-H=.5 results approach the exact-alpha=1 Padé result
continuously. The nonzero errors above persist as H approaches .5 and are
approximation error, not an Adams or Heston bug.

## Singularities, conditioning and legacy changes

The original expanded p_j and q_j formulas passed matching validation; no
polynomial signs or terms were corrected or algebraically simplified.
Unused b/g/root calculations were removed. Two evaluation changes are explicit:

1. Inverse gamma factors use `rgamma` at exact zeros. Current SciPy returns
   NaN for gamma at negative integers, so literal division by gamma would
   spuriously reject otherwise finite matching coefficients (notably order 4
   at alpha=2/3 and order 5 at alpha=3/4). This evaluates the mathematical
   reciprocal directly, without changing alpha or polynomial formulas.
2. At alpha=1 all used g_j for j>0 are exactly zero, including the missing
   legacy branches in orders 3 and 4. The exponentially decaying classical
   solution justifies this limit. The Heston tests above validate the result.

**Unrepaired discrepancy:** order 5's formal g_4 recurrence is singular at
alpha=2/3. For params=(2/3-.5,1.5,.04,.4,.04,-.7), u=1,
g_3=0.00044183+0.000175343i is nonzero, but the g_4 expression multiplies it
by Gamma(1-3*alpha)/Gamma(1-4*alpha), whose numerator diverges. This is not
fixed by a reciprocal-gamma zero. An assertion-based regression isolates it;
the public CF raises FloatingPointError. The legacy wrapper's epsilon
mutation would conceal this issue and has not been migrated. No replacement
asymptotic ansatz or reconstruction was attempted in this phase.

A scan of 180 (H,rho,sigma,u) cases per order over 0<=t<=5 includes
H=.01,.05,.1,1/6±1e-6,2/3-.5,.2,.3,.49,.5;
(rho,sigma)=(-.9,.8),(.9,.8),(0,.1);
u=.001,1,10,50,1-.5i,10-.5i. No rational poles were found in finite cases.
All 18 exact-alpha=2/3 cases for order 5 raised nonfinite-coefficient errors.
Minimum sampled abs(Q)/sum(abs(q_j)*y**j) was .8183, .7776, .2030,
.003258 for orders 2–5, respectively. Order 5 becomes poorly conditioned
near the singular alpha and should not be assumed reliable there.

Production checks examine denominator roots over the entire time interval,
including between quadrature nodes, and reject roots within 1e-10 of the
scaled interval. Evaluated denominators within 1e-12 of their term magnitude
sum also raise. Errors include order, u, T, params, and denominator magnitude
for pole failures. Synthetic exact and near-pole tests isolate this detection;
nonfinite coefficient and invalid-configuration tests also pass. This finite
scan is not a certification of all parameter/moment domains.

## Performance after correctness tests

Same 64 real frequencies linspace(.1,20,64), T=.5,
params=(.1,1.5,.04,.4,.04,-.7); reference Adams 3,200 steps.
Median of five warm calls, compilation excluded, NUMBA_NUM_THREADS=2 and
OPENBLAS_NUM_THREADS=1. Padé timings include coefficient construction,
whole-interval pole checks and time quadrature. Machine-dependent results:

| Method | Median ms | Speedup vs Adams 1600 | Max abs error | Max relative error |
|---|---:|---:|---:|---:|
| Adams 1600 | 237.51 | 1 | 4.19e-7 | 2.05e-6 |
| Padé 2 | 23.74 | 10.0 | .0128225 | .0642535 |
| Padé 3 | 25.72 | 9.23 | .00155798 | .00780703 |
| Padé 4 | 29.32 | 8.10 | .000563655 | .00282447 |
| Padé 5 | 41.16 | 5.77 | .000793585 | .00397664 |

Order 5 is less accurate than order 4 here. No pointwise monotonic-order
assertion is made and performance is not a correctness condition.
Run `PYTHONPATH=src NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/rough_heston/validate_pade.py`
for benchmark and scan JSON. Full-complex statistics are printed by
`pytest tests/rough_heston/test_pade_vs_adams.py -q -s`.

## Order 6 recommendation

Keep it private/unmigrated. Legacy h66 uses the same g recurrence through g_5
and a separate 597-line expanded aux_h66 module. The additional gamma factors
inherit the unresolved asymptotic issues and add further singular cases.
If pursued later, reconstruct the rational coefficients from explicit
matching equations, independently validate residuals and Adams errors, and
compare those results to the legacy expressions. Do not promote the legacy
module solely because lower orders work. Inspection did not establish that
order 6's expanded polynomial formulas are wrong or justify dropping it.

## Files and verification

Modified: the two rough_heston package `__init__.py` files (exports only).
Created: `characteristic_function/pade.py`; `pade_coefficients/__init__.py`,
`h22.py`, `h33.py`, `h44.py`, `h55.py`; three requested pytest files;
`examples/rough_heston/validate_pade.py`; this report.
Adams, Riccati helpers, params, and all legacy files are unchanged.
No staging or commits were performed.

Final verification after changing the default to order 4: rough-Heston tests
**385 passed** (18.66 s); full pytest discovery **385 passed** (18.73 s). `git diff --check` passed after removing
trailing blank lines in the export files. New files were also checked for
trailing whitespace. Git tracked diff: 2 export files; 11 new files, all
untracked. No changes are staged.
