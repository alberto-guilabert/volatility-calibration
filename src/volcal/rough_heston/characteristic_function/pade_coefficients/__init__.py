"""Order-specific, unsimplified two-point Padé coefficients."""

from . import h22, h33, h44, h55

COEFFICIENTS = {2: h22.coefficients, 3: h33.coefficients,
                4: h44.coefficients, 5: h55.coefficients}
