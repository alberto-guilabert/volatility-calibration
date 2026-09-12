"""Legacy (5,5) expanded coefficient formulas, retained for audit.

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
    b4 = (-b3*kappa_tilde*sigma+sigma**2*b1*b2)*gamma(1+3*alpha)/gamma(1+4*alpha)
    b5 = (-b4*kappa_tilde*sigma+sigma**2*(1/2*b2*b2+b1*b3))*gamma(1+4*alpha)/gamma(1+5*alpha)

    # Coeficientes de la expansión de h para t -> inf
    g0 = rm/sigma
    g1 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/A * (1*rgamma(1-alpha)) * (g0/sigma)
    g2 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/(A*sigma)*(gamma(1-alpha)*rgamma(1-2*alpha)*g1-1/2*sigma**2*g1*g1)
    g3 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/(A*sigma)*(gamma(1-2*alpha)*rgamma(1-3*alpha)*g2-sigma**2*g1*g2)
    g4 = np.zeros_like(g0, dtype=np.complex128) if alpha == 1.0 else -1/(A*sigma)*(gamma(1-3*alpha)*rgamma(1-4*alpha)*g3-sigma**2*(1/2*g2*g2+g1*g3))

    den = (-g0**5 - 4*b1*g0**3*g1 - 3*b1**2*g0*g1**2 + 3*b2*g0**2*g1**2 + 2*b1*b2*g1**3 -
              2*b3*g0*g1**3 + b4*g1**4 - 3*b1**2*g0**2*g2 - 3*b2*g0**3*g2 - 2*b1**3*g1*g2 +
              2*b1*b2*g0*g1*g2 + 4*b3*g0**2*g1*g2 - b2**2*g1**2*g2 - 2*b1*b3*g1**2*g2 -
              3*b4*g0*g1**2*g2 + b1**2*b2*g2**2 - 2*b2**2*g0*g2**2 + 4*b1*b3*g0*g2**2 +
              b4*g0**2*g2**2 + 2*b2*b3*g1*g2**2 - 2*b1*b4*g1*g2**2 - b3**2*g2**3 +
              b2*b4*g2**3 - 2*b1**3*g0*g3 - 4*b1*b2*g0**2*g3 - 2*b3*g0**3*g3 +
              2*b1**2*b2*g1*g3 + 4*b2**2*g0*g1*g3 + 2*b4*g0**2*g1*g3 - 2*b2*b3*g1**2*g3 +
              2*b1*b4*g1**2*g3 - 2*b1*b2**2*g2*g3 + 2*b1**2*b3*g2*g3 - 2*b2*b3*g0*g2*g3 +
              2*b1*b4*g0*g2*g3 + 2*b3**2*g1*g2*g3 - 2*b2*b4*g1*g2*g3 + b2**3*g3**2 -
              2*b1*b2*b3*g3**2 + b1**2*b4*g3**2 - b3**2*g0*g3**2 + b2*b4*g0*g3**2 -
              b1**4*g4 - 3*b1**2*b2*g0*g4 - b2**2*g0**2*g4 - 2*b1*b3*g0**2*g4 - b4*g0**3*g4 +
              2*b1*b2**2*g1*g4 - 2*b1**2*b3*g1*g4 + 2*b2*b3*g0*g1*g4 - 2*b1*b4*g0*g1*g4 -
              b3**2*g1**2*g4 + b2*b4*g1**2*g4 - b2**3*g2*g4 + 2*b1*b2*b3*g2*g4 -
              b1**2*b4*g2*g4 + b3**2*g0*g2*g4 - b2*b4*g0*g2*g4)

    # Coeficientes del denominador de Padé (5,5)
    q1 = (-(b1*g0**4) - 3*b1**2*g0**2*g1 + b2*g0**3*g1 - b1**3*g1**2 + 4*b1*b2*g0*g1**2 - \
             b3*g0**2*g1**2 - b2**2*g1**3 - 2*b1*b3*g1**3 + b4*g0*g1**3 - b5*g1**4 - \
             2*b1**3*g0*g2 - b1*b2*g0**2*g2 + b3*g0**3*g2 + 4*b1**2*b2*g1*g2 + \
             2*b1*b3*g0*g1*g2 - 2*b4*g0**2*g1*g2 + 2*b2*b3*g1**2*g2 + b1*b4*g1**2*g2 + \
             3*b5*g0*g1**2*g2 - 2*b1*b2**2*g2**2 + b1**2*b3*g2**2 - 2*b1*b4*g0*g2**2 - \
             b5*g0**2*g2**2 - b3**2*g1*g2**2 - b2*b4*g1*g2**2 + 2*b1*b5*g1*g2**2 + \
             b3*b4*g2**3 - b2*b5*g2**3 - b1**4*g3 - b1**2*b2*g0*g3 + b2**2*g0**2*g3 + \
             b4*g0**3*g3 - 2*b1**2*b3*g1*g3 - 4*b2*b3*g0*g1*g3 - 2*b5*g0**2*g1*g3 + \
             b3**2*g1**2*g3 + b2*b4*g1**2*g3 - 2*b1*b5*g1**2*g3 + b2**3*g2*g3 - \
             b1**2*b4*g2*g3 + b3**2*g0*g2*g3 + b2*b4*g0*g2*g3 - 2*b1*b5*g0*g2*g3 - \
             2*b3*b4*g1*g2*g3 + 2*b2*b5*g1*g2*g3 - b2**2*b3*g3**2 + b1*b3**2*g3**2 + \
             b1*b2*b4*g3**2 - b1**2*b5*g3**2 + b3*b4*g0*g3**2 - b2*b5*g0*g3**2 + \
             b1**3*b2*g4 + 2*b1*b2**2*g0*g4 + b1**2*b3*g0*g4 + 2*b2*b3*g0**2*g4 + \
             b1*b4*g0**2*g4 + b5*g0**3*g4 - b2**3*g1*g4 + b1**2*b4*g1*g4 - b3**2*g0*g1*g4 - \
             b2*b4*g0*g1*g4 + 2*b1*b5*g0*g1*g4 + b3*b4*g1**2*g4 - b2*b5*g1**2*g4 + \
             b2**2*b3*g2*g4 - b1*b3**2*g2*g4 - b1*b2*b4*g2*g4 + b1**2*b5*g2*g4 - \
             b3*b4*g0*g2*g4 + b2*b5*g0*g2*g4)/den
    q2 = (-(b1**2*g0**3) - b2*g0**4 - 2*b1**3*g0*g1 - b1*b2*g0**2*g1 + b3*g0**3*g1 + \
             2*b1**2*b2*g1**2 + b2**2*g0*g1**2 - b4*g0**2*g1**2 + b1*b4*g1**3 + \
             b5*g0*g1**3 - b1**4*g2 - b1**2*b2*g0*g2 - 2*b2**2*g0**2*g2 + 3*b1*b3*g0**2*g2 + \
             b4*g0**3*g2 - 2*b1*b2**2*g1*g2 - 4*b1*b4*g0*g1*g2 - 2*b5*g0**2*g1*g2 - \
             b2*b4*g1**2*g2 + b1*b5*g1**2*g2 + 2*b1*b2*b3*g2**2 - 2*b1**2*b4*g2**2 - \
             b3**2*g0*g2**2 + 3*b2*b4*g0*g2**2 - 2*b1*b5*g0*g2**2 + b3*b4*g1*g2**2 - \
             b2*b5*g1*g2**2 - b4**2*g2**3 + b3*b5*g2**3 + b1**3*b2*g3 + 3*b1**2*b3*g0*g3 + \
             3*b1*b4*g0**2*g3 + b5*g0**3*g3 + b2**3*g1*g3 - 2*b1*b2*b3*g1*g3 + \
             b1**2*b4*g1*g3 + b3**2*g0*g1*g3 - b2*b4*g0*g1*g3 - b3*b4*g1**2*g3 + \
             b2*b5*g1**2*g3 - b2**2*b3*g2*g3 - b1*b3**2*g2*g3 + 3*b1*b2*b4*g2*g3 - \
             b1**2*b5*g2*g3 - b3*b4*g0*g2*g3 + b2*b5*g0*g2*g3 + 2*b4**2*g1*g2*g3 - \
             2*b3*b5*g1*g2*g3 + b2*b3**2*g3**2 - b2**2*b4*g3**2 - b1*b3*b4*g3**2 + \
             b1*b2*b5*g3**2 - b4**2*g0*g3**2 + b3*b5*g0*g3**2 - b1**2*b2**2*g4 + \
             b1**3*b3*g4 - b2**3*g0*g4 + b1**2*b4*g0*g4 - b2*b4*g0**2*g4 + b1*b5*g0**2*g4 + \
             b2**2*b3*g1*g4 + b1*b3**2*g1*g4 - 3*b1*b2*b4*g1*g4 + b1**2*b5*g1*g4 + \
             b3*b4*g0*g1*g4 - b2*b5*g0*g1*g4 - b4**2*g1**2*g4 + b3*b5*g1**2*g4 - \
             b2*b3**2*g2*g4 + b2**2*b4*g2*g4 + b1*b3*b4*g2*g4 - b1*b2*b5*g2*g4 + \
             b4**2*g0*g2*g4 - b3*b5*g0*g2*g4)/den
    q3 = (-(b1**3*g0**2) - 2*b1*b2*g0**3 - b3*g0**4 - b1**4*g1 - b1**2*b2*g0*g1 + \
             2*b2**2*g0**2*g1 - b1*b3*g0**2*g1 + b4*g0**3*g1 + b1*b2**2*g1**2 - \
             2*b1**2*b3*g1**2 - 2*b2*b3*g0*g1**2 - b5*g0**2*g1**2 + b2*b4*g1**3 - \
             b1*b5*g1**3 + b1**3*b2*g2 + 3*b1**2*b3*g0*g2 + 3*b1*b4*g0**2*g2 + b5*g0**3*g2 + \
             2*b3**2*g0*g1*g2 - 2*b2*b4*g0*g1*g2 - b3*b4*g1**2*g2 + b2*b5*g1**2*g2 - \
             b1*b3**2*g2**2 + b1*b2*b4*g2**2 - b3*b4*g0*g2**2 + b2*b5*g0*g2**2 + \
             b4**2*g1*g2**2 - b3*b5*g1*g2**2 - b1**2*b2**2*g3 + b1**3*b3*g3 + b2**3*g0*g3 - \
             4*b1*b2*b3*g0*g3 + 3*b1**2*b4*g0*g3 - 2*b3**2*g0**2*g3 + b2*b4*g0**2*g3 + \
             b1*b5*g0**2*g3 - b2**2*b3*g1*g3 + 3*b1*b3**2*g1*g3 - b1*b2*b4*g1*g3 - \
             b1**2*b5*g1*g3 + 3*b3*b4*g0*g1*g3 - 3*b2*b5*g0*g1*g3 - b4**2*g1**2*g3 + \
             b3*b5*g1**2*g3 + b2*b3**2*g2*g3 - b2**2*b4*g2*g3 - b1*b3*b4*g2*g3 + \
             b1*b2*b5*g2*g3 - b4**2*g0*g2*g3 + b3*b5*g0*g2*g3 - b3**3*g3**2 + \
             2*b2*b3*b4*g3**2 - b1*b4**2*g3**2 - b2**2*b5*g3**2 + b1*b3*b5*g3**2 + \
             b1*b2**3*g4 - 2*b1**2*b2*b3*g4 + b1**3*b4*g4 + b2**2*b3*g0*g4 - \
             2*b1*b3**2*g0*g4 + b1**2*b5*g0*g4 - b3*b4*g0**2*g4 + b2*b5*g0**2*g4 - \
             b2*b3**2*g1*g4 + b2**2*b4*g1*g4 + b1*b3*b4*g1*g4 - b1*b2*b5*g1*g4 + \
             b4**2*g0*g1*g4 - b3*b5*g0*g1*g4 + b3**3*g2*g4 - 2*b2*b3*b4*g2*g4 + \
             b1*b4**2*g2*g4 + b2**2*b5*g2*g4 - b1*b3*b5*g2*g4)/den
    q4 = (-(b1**4*g0) - 3*b1**2*b2*g0**2 - b2**2*g0**3 - 2*b1*b3*g0**3 - b4*g0**4 + \
             b1**3*b2*g1 + 4*b1*b2**2*g0*g1 - b1**2*b3*g0*g1 + 4*b2*b3*g0**2*g1 - \
             b1*b4*g0**2*g1 + b5*g0**3*g1 - b2**3*g1**2 + b1**2*b4*g1**2 - \
             2*b3**2*g0*g1**2 + 2*b1*b5*g0*g1**2 + b3*b4*g1**3 - b2*b5*g1**3 - \
             b1**2*b2**2*g2 + b1**3*b3*g2 - 2*b2**3*g0*g2 + 2*b1*b2*b3*g0*g2 + \
             b3**2*g0**2*g2 - 2*b2*b4*g0**2*g2 + b1*b5*g0**2*g2 + 2*b2**2*b3*g1*g2 - \
             4*b1*b2*b4*g1*g2 + 2*b1**2*b5*g1*g2 - b4**2*g1**2*g2 + b3*b5*g1**2*g2 - \
             b2*b3**2*g2**2 + b2**2*b4*g2**2 + b1*b3*b4*g2**2 - b1*b2*b5*g2**2 + \
             b4**2*g0*g2**2 - b3*b5*g0*g2**2 + b1*b2**3*g3 - 2*b1**2*b2*b3*g3 + \
             b1**3*b4*g3 + b2**2*b3*g0*g3 - 2*b1*b3**2*g0*g3 + b1**2*b5*g0*g3 - \
             b3*b4*g0**2*g3 + b2*b5*g0**2*g3 - b2*b3**2*g1*g3 + b2**2*b4*g1*g3 + \
             b1*b3*b4*g1*g3 - b1*b2*b5*g1*g3 + b4**2*g0*g1*g3 - b3*b5*g0*g1*g3 + \
             b3**3*g2*g3 - 2*b2*b3*b4*g2*g3 + b1*b4**2*g2*g3 + b2**2*b5*g2*g3 - \
             b1*b3*b5*g2*g3 - b2**4*g4 + 3*b1*b2**2*b3*g4 - b1**2*b3**2*g4 - \
             2*b1**2*b2*b4*g4 + b1**3*b5*g4 + 2*b2*b3**2*g0*g4 - 2*b2**2*b4*g0*g4 - \
             2*b1*b3*b4*g0*g4 + 2*b1*b2*b5*g0*g4 - b4**2*g0**2*g4 + b3*b5*g0**2*g4 - \
             b3**3*g1*g4 + 2*b2*b3*b4*g1*g4 - b1*b4**2*g1*g4 - b2**2*b5*g1*g4 + \
             b1*b3*b5*g1*g4)/den
    q5 = (-b1**5 - 4*b1**3*b2*g0 - 3*b1*b2**2*g0**2 - 3*b1**2*b3*g0**2 - 2*b2*b3*g0**3 - \
             2*b1*b4*g0**3 - b5*g0**4 + 3*b1**2*b2**2*g1 - 3*b1**3*b3*g1 + 2*b2**3*g0*g1 + \
             2*b1*b2*b3*g0*g1 - 4*b1**2*b4*g0*g1 + b3**2*g0**2*g1 + 2*b2*b4*g0**2*g1 - \
             3*b1*b5*g0**2*g1 - b2**2*b3*g1**2 - 2*b1*b3**2*g1**2 + 4*b1*b2*b4*g1**2 - \
             b1**2*b5*g1**2 - 2*b3*b4*g0*g1**2 + 2*b2*b5*g0*g1**2 + b4**2*g1**3 - \
             b3*b5*g1**3 - 2*b1*b2**3*g2 + 4*b1**2*b2*b3*g2 - 2*b1**3*b4*g2 - \
             2*b2**2*b3*g0*g2 + 4*b1*b3**2*g0*g2 - 2*b1**2*b5*g0*g2 + 2*b3*b4*g0**2*g2 -
             2*b2*b5*g0**2*g2 + 2*b2*b3**2*g1*g2 - 2*b2**2*b4*g1*g2 - 2*b1*b3*b4*g1*g2 + \
             2*b1*b2*b5*g1*g2 - 2*b4**2*g0*g1*g2 + 2*b3*b5*g0*g1*g2 - b3**3*g2**2 + \
             2*b2*b3*b4*g2**2 - b1*b4**2*g2**2 - b2**2*b5*g2**2 + b1*b3*b5*g2**2 + \
             b2**4*g3 - 3*b1*b2**2*b3*g3 + b1**2*b3**2*g3 + 2*b1**2*b2*b4*g3 - b1**3*b5*g3 - \
             2*b2*b3**2*g0*g3 + 2*b2**2*b4*g0*g3 + 2*b1*b3*b4*g0*g3 - 2*b1*b2*b5*g0*g3 + \
             b4**2*g0**2*g3 - b3*b5*g0**2*g3 + b3**3*g1*g3 - 2*b2*b3*b4*g1*g3 + \
             b1*b4**2*g1*g3 + b2**2*b5*g1*g3 - b1*b3*b5*g1*g3)/den

    # Coeficientes del numerador de Padé (5,5)
    p1 = b1
    p2 = b2 + b1*q1
    p3 = b3 + b1*q2 + b2*q1
    p4 = b4 + b3*q1 + b2*q2 + b1*q3
    p5 = g0*q5

    return (np.array([0j, p1, p2, p3, p4, p5], dtype=np.complex128),
            np.array([1+0j, q1, q2, q3, q4, q5], dtype=np.complex128))
