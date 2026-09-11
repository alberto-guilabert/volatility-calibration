"""CF invariants and an independent classical-Heston reference.

The analytic oracle solves the classical affine Riccati equation using the
decaying exponential branch; it imports no volcal Heston or legacy code.
It is checked separately against a high-accuracy DOP853 integration of the
classical ODE. At H=0.5, (kappa,theta,sigma,v0,rho) map identically, with sigma
the CIR diffusion coefficient (not sigma/kappa). Carry/log-spot phases are zero
because this is the CF of log(S_T/F_T).

Formula: D=(b-d)/sigma**2*(1-exp(-d*T))/(1-g*exp(-d*T)),
C=kappa*theta/sigma**2*((b-d)*T-2*log((1-g*exp(-d*T))/(1-g))),
b=kappa-i*rho*sigma*u, d=sqrt(b**2+sigma**2*(u**2+i*u)), g=(b-d)/(b+d).
This follows directly from D'=a+b_ode*D+c*D**2 and C'=kappa*theta*D.

Tolerances: algebraic identities 1e-14; independently checked analytic oracle
2e-11; Adams/analytic comparison 2e-7 at M=800 on the declared test domain.
The latter is a discretization budget, not a machine-precision identity.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from volcal.rough_heston import RoughHestonParams, rough_heston_cf_adams as cf


PARAMS = RoughHestonParams(0.1, 1.5, 0.04, 0.4, 0.04, -0.7)
FREQUENCIES = np.array([0.1, 1, 3, 10, 1 - 0.5j, 3 - 0.5j])


def classical_heston_cf(u, T, params):
    u = np.atleast_1d(np.asarray(u, dtype=complex))
    out = np.ones(u.shape, dtype=complex)
    mask = (u != 0) & (u != -1j)
    z = u[mask]
    b = params.kappa - 1j*params.rho*params.sigma*z
    d = np.sqrt(b**2 + params.sigma**2*(z**2 + 1j*z))
    d = np.where(d.real < 0, -d, d)
    g = (b-d)/(b+d)
    e = np.exp(-d*T)
    D = (b-d)/params.sigma**2*(-np.expm1(-d*T))/(1-g*e)
    C = params.kappa*params.theta/params.sigma**2 * (
        (b-d)*T - 2*(np.log1p(-g*e)-np.log1p(-g))
    )
    out[mask] = np.exp(C + D*params.v0)
    return out


@pytest.mark.parametrize("T", [0.01, 0.5, 2.0])
@pytest.mark.parametrize("rho", [-0.7, 0.7])
def test_classical_analytic_oracle_against_independent_ode(T, rho):
    p = replace(PARAMS, H=0.5, rho=rho)
    exact = classical_heston_cf(FREQUENCIES, T, p)
    for u, expected in zip(FREQUENCIES, exact):
        def ode(t, y):
            D, C = y
            return [-(u*u + 1j*u)/2 + (1j*u*p.rho*p.sigma-p.kappa)*D
                    + p.sigma*p.sigma*D*D/2, p.kappa*p.theta*D]
        solution = solve_ivp(ode, (0, T), [0j, 0j], method="DOP853",
                             rtol=2e-12, atol=2e-14)
        assert solution.success, solution.message
        D, C = solution.y[:, -1]
        np.testing.assert_allclose(expected, np.exp(C + p.v0*D), rtol=0, atol=2e-11)


@pytest.mark.parametrize("H", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("rho", [-1.0, -0.7, 0.7, 1.0])
def test_exact_normalization_and_martingale(H, rho):
    p = replace(PARAMS, H=H, rho=rho)
    np.testing.assert_array_equal(cf([0, -1j], 2, p), [1 + 0j, 1 + 0j])
    # Nearby frequencies must approach the identity as well; the exact shortcut
    # must not hide a discontinuity in the numerical solver.
    near_martingale = cf(np.array([-1j-1e-7, -1j+1e-7]), 0.5, p)
    np.testing.assert_allclose(near_martingale, 1, rtol=0, atol=2e-8)


def test_zero_maturity_for_arbitrary_frequencies():
    np.testing.assert_array_equal(cf([0, 1, -1j, 2 - 0.5j, 100 + 2j], 0, PARAMS),
                                  np.ones(5))
    assert cf(3j, 0, PARAMS) == 1 + 0j


@pytest.mark.parametrize("H", [0.05, 0.2, 0.5])
def test_conjugate_symmetry_and_modulus_on_real_axis(H):
    p = replace(PARAMS, H=H)
    u = np.linspace(0, 20, 41)
    values = cf(u, 0.5, p)
    np.testing.assert_allclose(cf(-u, 0.5, p), values.conjugate(), rtol=0, atol=1e-14)
    assert np.all(abs(values) <= 1 + 1e-14)


def test_continuity_near_origin():
    errors = [abs(cf(eps, 0.5, PARAMS)-1) for eps in (1e-3, 1e-5, 1e-7)]
    assert errors[1] < 0.02*errors[0]
    assert errors[2] < 0.02*errors[1]
    assert errors[-1] < 2e-9


@pytest.mark.parametrize("H", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("T", [1/365, 0.1, 2.0])
@pytest.mark.parametrize("rho", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("sigma", [0.1, 0.4, 0.8])
def test_representative_grid_is_finite(H, T, rho, sigma):
    p = replace(PARAMS, H=H, rho=rho, sigma=sigma)
    # Stay within the normalized stock's moment strip, Im(u) in [-1,0].
    u = np.array([-5, 0, 5, -2-0.5j, 2-0.5j, 0.1-0.99j, 3-0.1j])
    assert np.all(np.isfinite(cf(u, T, p, time_steps=800)))


@pytest.mark.parametrize("T", [0.01, 0.5, 2.0])
def test_h_half_matches_classical_heston(T):
    p = replace(PARAMS, H=0.5)
    expected = classical_heston_cf(FREQUENCIES, T, p)
    values = [cf(FREQUENCIES, T, p, time_steps=n) for n in (200, 400, 800)]
    errors = [np.max(abs(value-expected)) for value in values]
    assert errors[1] < 0.4*errors[0], errors
    assert errors[2] < 0.4*errors[1], errors
    np.testing.assert_allclose(values[-1], expected, rtol=0, atol=2e-7)


def test_approach_to_classical_heston_limit():
    expected = classical_heston_cf(FREQUENCIES, 0.5, replace(PARAMS, H=0.5))
    errors = [np.max(abs(cf(FREQUENCIES, 0.5, replace(PARAMS, H=H), time_steps=800)
                         - expected)) for H in (0.45, 0.49, 0.499, 0.5)]
    assert all(b < a for a, b in zip(errors, errors[1:])), errors


def test_zero_variance_process():
    p = replace(PARAMS, theta=0, v0=0)
    np.testing.assert_array_equal(cf(FREQUENCIES, 0.5, p), np.ones(len(FREQUENCIES)))
