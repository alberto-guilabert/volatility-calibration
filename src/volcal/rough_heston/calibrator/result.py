"""Immutable calibration and validation snapshots."""
from dataclasses import dataclass
from ..params import RoughHestonParams
from ..pricer import RoughHestonPricer


@dataclass(frozen=True)
class OptimizerStatus:
    success: bool
    status: object
    message: str
    loss: float
    evaluations: int
    iterations: int


@dataclass(frozen=True)
class RepricingResult:
    configuration: RoughHestonPricer
    prices: tuple = ()
    loss: object = None
    diagnostics: tuple = ()  # (original row indices, Diagnostic)
    failure: str = ''
    max_market_price_error: object = None
    max_market_iv_error: object = None
    iv_comparisons: int = 0
    max_calibration_price_difference: object = None

    @property
    def passed(self):
        return not self.failure and all(d.passed for _, d in self.diagnostics)


@dataclass(frozen=True)
class CalibrationResult:
    params: RoughHestonParams
    initial_loss: object
    de_loss: float
    final_loss: float
    runtime: float
    de: OptimizerStatus
    lbfgsb: OptimizerStatus
    total_objective_evaluations: int
    failed_numerical_evaluations: int
    failures: tuple
    objective_convention: str
    pricing_method: str
    numerical_configuration: RoughHestonPricer
    calibration_configuration: object
    fixed_H: object
    mean_objective_seconds: float
    final_repricing: RepricingResult
    refinement: tuple  # (original row indices, RefinementResult)
    adams_validation: object
    selected_from: str


class CalibrationError(RuntimeError):
    """No valid objective evaluation exists; a penalty is not a fitted price."""
    def __init__(self, failures, evaluations):
        self.failures = tuple(failures)
        self.evaluations = evaluations
        super().__init__('calibration produced no valid pricing evaluation')
