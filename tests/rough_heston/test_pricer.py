from dataclasses import replace
import numpy as np
import pytest
from volcal.rough_heston import RoughHestonParams
from volcal.rough_heston.pricer import AdamsConfig, PadeConfig, RoughHestonPricer, SinhConfig

PARAMS=RoughHestonParams(.2,1.5,.04,.4,.04,-.7)


@pytest.mark.parametrize('method', ['adams','pade'])
def test_financial_identities(method):
    pricer=RoughHestonPricer(method)
    K=np.linspace(50,200,61)
    args=dict(T=.5,K=K,option_params=(100,.04,.015),rough_heston_params=PARAMS)
    c=pricer.vanilla_price(**args)
    p=pricer.vanilla_price(**args,option_type='put')
    ds=100*np.exp(-.015*.5);dk=K*np.exp(-.04*.5)
    parity=max(abs(c-p-(ds-dk)))
    lower=min(np.min(c-np.maximum(ds-dk,0)),np.min(p-np.maximum(dk-ds,0)))
    slope=np.max(np.diff(c));convex=np.min(np.diff(c,2))
    print(method,'parity',parity,'lower margin',lower,'max call increment',slope,'min second difference',convex)
    assert parity<3e-14
    assert lower>-2e-7
    assert np.all(c<=ds+2e-7) and np.all(p<=dk+2e-7)
    assert slope<2e-7 and convex>-2e-7
    scaled=pricer.vanilla_price(**dict(args,K=K*3,option_params=(300,.04,.015)))
    np.testing.assert_allclose(scaled,3*c,atol=3e-12,rtol=0)
    mixed=pricer.vanilla_price(**args,option_type=np.where(K<100,'put','call'))
    np.testing.assert_allclose(mixed,np.where(K<100,p,c),atol=0,rtol=0)
    scalar=pricer.vanilla_price(**dict(args,K=100.))
    assert isinstance(scalar,float)
    assert abs(scalar-c[20])<1e-12


@pytest.mark.parametrize('method',['adams','pade'])
def test_expiry_and_short_maturity(method):
    pricer=RoughHestonPricer(method,integration_config=SinhConfig(spacing=.005,nodes=1800))
    args=dict(K=[50,100,200],option_params=(100,.04,.015),rough_heston_params=PARAMS)
    np.testing.assert_array_equal(pricer.vanilla_price(T=0,**args),[50,0,0])
    np.testing.assert_array_equal(pricer.vanilla_price(T=0,**args,option_type='put'),[0,0,100])
    c=pricer.vanilla_price(T=.001,**args)
    intrinsic=np.maximum(100*np.exp(-.015*.001)-np.array(args['K'])*np.exp(-.04*.001),0)
    assert np.all(c>=intrinsic-2e-7)
    assert np.all(c<=100*np.exp(-.015*.001))
    assert 0<c[1]<1


@pytest.mark.parametrize('kwargs',[{'T':-1},{'T':[1]},{'K':0},{'K':[]},{'K':[[100]]},
    {'K':1j},{'K':np.nan},{'option_params':(0,0,0)}, {'option_params':(100,0)},
    {'option_type':'CALL'},{'option_type':['call','put']}])
def test_invalid_inputs(kwargs):
    args=dict(T=1,K=100,option_params=(100,0,0),rough_heston_params=PARAMS)
    args.update(kwargs)
    with pytest.raises(ValueError): RoughHestonPricer().vanilla_price(**args)


def test_config_and_failure_propagation():
    default=RoughHestonPricer()
    assert default.cf_method=='adams'
    assert default.cf_config==AdamsConfig()
    assert default==RoughHestonPricer(cf_method='adams')
    pade=RoughHestonPricer(cf_method='pade')
    assert pade.cf_method=='pade'
    assert pade.cf_config==PadeConfig(order=4)
    with pytest.raises(ValueError): RoughHestonPricer('other')
    with pytest.raises(TypeError): RoughHestonPricer('adams',PadeConfig())
    with pytest.raises(ValueError): PadeConfig(6)
    with pytest.raises(ValueError): AdamsConfig(0)
    args=dict(T=.5,K=100,option_params=(100,0,0),rough_heston_params=replace(PARAMS,H=1/6))
    with pytest.raises(FloatingPointError,match='order=5'):
        RoughHestonPricer('pade',PadeConfig(5)).vanilla_price(**args)
    args.update(T=2,rough_heston_params=replace(PARAMS,H=.1,sigma=.3))
    with pytest.raises(FloatingPointError,match='Modified Adams'):
        RoughHestonPricer('adams',AdamsConfig(400)).vanilla_price(**args)


@pytest.mark.parametrize('method',['adams','pade'])
def test_bent_contour_agrees_with_horizontal(method):
    args=dict(T=.5,K=[80,100,120],option_params=(100,.04,.015),rough_heston_params=PARAMS)
    flat=RoughHestonPricer(method).vanilla_price(**args)
    for omega in [-.05,.05]:
        bent=RoughHestonPricer(method,integration_config=SinhConfig(omega=omega)).vanilla_price(**args)
        np.testing.assert_allclose(bent,flat,atol=1e-8,rtol=0)
