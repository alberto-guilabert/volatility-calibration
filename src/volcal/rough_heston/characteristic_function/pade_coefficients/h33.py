"""Legacy (3,3) expanded coefficient formulas, retained for audit.

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
    b3 = (-b2*kappa_tilde*sigma+sigma**2*b1**2/2)*gamma(1+2*alpha)/gamma(1+3*alpha)

    # Coeficientes de la expansión de h para t -> inf
    g0 = rm/sigma
    g1 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/A * (1*rgamma(1-alpha)) * (g0/sigma)
    g2 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/(A*sigma)*(gamma(1-alpha)*rgamma(1-2*alpha)*g1-1/2*sigma**2*g1*g1)


    den = g0**3+2*b1*g0*g1-b2*g1**2+b1**2*g2+b2*g0*g2

    # Coeficientes del denominador de Padé (3,3)
    q1 = (b1*g0**2+b1**2*g1-b2*g0*g1+b3*g1**2-b1*b2*g2-b3*g0*g2)/den
    q2 = (b1**2*g0+b2*g0**2-b1*b2*g1-b3*g0*g1+b2**2*g2-b1*b3*g2)/den
    q3 = (b1**3+2*b1*b2*g0+b3*g0**2-b2**2*g1+b1*b3*g1)/den

    # Coeficientes del numerador de Padé (3,3)
    p1 = b1
    p2 = (b1**2*g0**2+b2*g0**3+b1**3*g1+b1*b2*g0*g1-b2**2*g1**2+b1*b3*g1**2+b2**2*g0*g2-b1*b3*g0*g2)/den
    p3 = g0*q3

    return (np.array([0j, p1, p2, p3], dtype=np.complex128),
            np.array([1+0j, q1, q2, q3], dtype=np.complex128))
