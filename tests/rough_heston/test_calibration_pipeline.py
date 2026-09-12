from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from volcal.rough_heston.calibrator import calibrate, CalibrationConfig, CalibrationError
from volcal.rough_heston.pricer import RoughHestonPricer
from test_calibration_loss import quotes


def test_no_valid_prices_is_error(monkeypatch):
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.full(len(kw['K']),np.nan))
    with pytest.raises(CalibrationError) as exc:
        calibrate(quotes(),RoughHestonPricer('pade'),CalibrationConfig(de_popsize=1,de_maxiter=0,lbfgsb_maxiter=0))
    assert len(exc.value.failures) == exc.value.evaluations


def test_fixed_evaluations_diagnostics_and_status(monkeypatch):
    methods=[]
    def price(self,**kw):
        methods.append(self.cf_method)
        p=kw['rough_heston_params']
        if p.sigma>.5: raise FloatingPointError('test rational pole')
        return np.full(len(kw['K']),100*p.v0)
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',price)
    import volcal.rough_heston.calibrator.pipeline as pipeline
    monkeypatch.setattr(pipeline,'refine_prices',lambda *a,**kw: pytest.fail('refinement during optimization'))
    result=calibrate(quotes(),RoughHestonPricer('pade'),CalibrationConfig(de_popsize=2,de_maxiter=0,lbfgsb_maxiter=10))
    assert result.failed_numerical_evaluations > 0
    assert result.failed_numerical_evaluations == len(result.failures)
    assert result.total_objective_evaluations == 1+result.de.evaluations+result.lbfgsb.evaluations
    assert set(methods)=={'pade'}
    assert not result.de.success and result.de.message
    assert result.final_repricing.prices and result.refinement==()
    assert result.final_repricing.loss==result.final_loss
    with pytest.raises(FrozenInstanceError): result.params=None


def test_adams_supported(monkeypatch):
    seen=[]
    def price(self,**kw):
        seen.append(self.cf_method)
        return np.full(len(kw['K']),6.)
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',price)
    result=calibrate(quotes(),RoughHestonPricer(),CalibrationConfig(de_popsize=1,de_maxiter=0,lbfgsb_maxiter=0))
    assert result.pricing_method=='adams' and set(seen)=={'adams'}


def test_penalty_smaller_than_loss_cannot_be_selected(monkeypatch):
    def price(self,**kw):
        if kw['rough_heston_params'].sigma>.5: raise FloatingPointError('bad')
        return np.full(len(kw['K']),50.)
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',price)
    result=calibrate(quotes(),RoughHestonPricer('pade'),CalibrationConfig(de_popsize=2,de_maxiter=0,lbfgsb_maxiter=0,failure_penalty=1e-20))
    assert result.failed_numerical_evaluations
    assert result.final_loss>1e-20
    assert result.params.sigma<=.5
    assert result.final_repricing.prices==(50.,50.)


def test_seed_reproducibility(monkeypatch):
    def price(self,**kw):
        p=kw['rough_heston_params']
        return np.full(len(kw['K']),100*p.v0+p.theta+p.H+p.sigma)
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',price)
    config=CalibrationConfig(de_popsize=2,de_maxiter=1,lbfgsb_maxiter=5,seed=71)
    a=calibrate(quotes(),RoughHestonPricer('pade'),config)
    b=calibrate(quotes(),RoughHestonPricer('pade'),config)
    assert a.params==b.params and a.final_loss==b.final_loss
    assert a.total_objective_evaluations==b.total_objective_evaluations
    assert a.de==b.de and a.lbfgsb==b.lbfgsb


def test_failed_final_repricing_has_no_prices(monkeypatch):
    import volcal.rough_heston.calibrator.pipeline as pipeline
    from volcal.rough_heston import RoughHestonParams
    from volcal.rough_heston.calibrator import CalibrationObjective
    objective=CalibrationObjective(quotes(),RoughHestonPricer('pade'))
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',lambda self,**kw: np.full(len(kw['K']),np.nan))
    result=pipeline._reprice(objective,RoughHestonParams(.2,1.5,.04,.4,.04,-.7),objective.pricer)
    assert not result.passed and result.failure and result.prices==() and result.loss is None


def test_post_validation_does_not_change_optimum(monkeypatch):
    import volcal.rough_heston.calibrator.pipeline as pipeline
    from volcal.rough_heston.pricer import RefinementConfig
    stages=[]
    def price(self,**kw):
        stages.append(self.cf_method)
        return np.full(len(kw['K']),6.+(self.cf_method=='adams'))
    def refine(*a,**kw):
        stages.append('refine')
        return 'refinement sentinel'
    monkeypatch.setattr(RoughHestonPricer,'vanilla_price',price)
    monkeypatch.setattr(pipeline,'refine_prices',refine)
    result=calibrate(quotes(),RoughHestonPricer('pade'),
        CalibrationConfig(de_popsize=1,de_maxiter=0,lbfgsb_maxiter=0),
        refinement_config=RefinementConfig(),adams_validation_pricer=RoughHestonPricer())
    assert stages[-4:]==['refine','refine','adams','adams']
    assert result.adams_validation.max_calibration_price_difference==1.
    assert result.final_repricing.prices==(6.,6.)
    assert result.pricing_method=='pade'
