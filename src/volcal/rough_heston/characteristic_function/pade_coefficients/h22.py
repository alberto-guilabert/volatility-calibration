"""Legacy (2,2) expanded coefficient formulas, retained for audit.

Unused terms were removed; reciprocal gamma uses rgamma at its exact zeros.
The expanded numerator/denominator expressions are unchanged.
At alpha=1 all inverse-power asymptotic
coefficients vanish because the classical Riccati solution converges
exponentially (on its stable branch). No alpha perturbation is used.
"""

import numpy as np
from scipy.special import gamma, rgamma


def coefficients(u, params):
    """Return ascending numerator/denominator coefficients (internal API)."""
    alpha = params.alpha
    kappa, sigma, rho = params.kappa, params.sigma, params.rho
    kappa_tilde = kappa/sigma - 1j*rho*u
    # A: raíz cuadrada compleja
    A = np.sqrt(u*(u+1j) + kappa_tilde**2)
    # Fijamos la rama de la raíz: Re(A) >= 0 para estabilidad numérica
    A = np.where(np.real(A) < 0, -A, A)
    rm = kappa/sigma - 1j*rho*u - A

    # Coeficientes de la expansión de h para t -> 0
    b1 = -u*(u+1j)/(2*gamma(1+alpha))
    b2 = -gamma(1+alpha)/gamma(1+2*alpha)*kappa_tilde*sigma*b1

    # Coeficientes de la expansión de h para t -> inf
    g0 = rm/sigma
    g1 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/A * (1*rgamma(1-alpha)) * (g0/sigma)

    # Coeficientes del denominador de Padé (2,2)
    q1 = (b1*g0-b2*g1)/(g0**2 + b1*g1)
    q2 = (b1**2+b2*g0)/(g0**2 + b1*g1)

    # Coeficientes del numerador de Padé (2,2)
    p1 = b1
    p2 = b2 + b1*q1

    return (np.array([0j, p1, p2], dtype=np.complex128),
            np.array([1+0j, q1, q2], dtype=np.complex128))
