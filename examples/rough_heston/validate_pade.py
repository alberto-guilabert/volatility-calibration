"""Run with PYTHONPATH=src:tests/rough_heston.

Use NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 for the recorded benchmark.
No pricing, calibration, or automatic accuracy refinement is performed.
"""
from dataclasses import replace
from time import perf_counter
import json

import numpy as np
from volcal.rough_heston import RoughHestonParams, rough_heston_cf_adams as adams, rough_heston_cf_pade as pade
from volcal.rough_heston.characteristic_function.pade import _checked_coefficients


def main():
    p=RoughHestonParams(.1,1.5,.04,.4,.04,-.7)
    u=np.linspace(.1,20,64);T=.5
    ref=adams(u,T,p,time_steps=3200)
    methods={'Adams 1600':lambda:adams(u,T,p,time_steps=1600)}
    methods.update({f'Pade {n}':lambda n=n:pade(u,T,p,order=n) for n in range(2,6)})
    output={}
    for name,fn in methods.items():
        value=fn()  # warm up; exclude compilation
        times=[]
        for _ in range(5):
            start=perf_counter();fn();times.append(perf_counter()-start)
        output[name]={'median_seconds':float(np.median(times)),
                      'max_abs_error':float(max(abs(value-ref))),
                      'max_rel_error':float(max(abs(value-ref)/abs(ref)))}
    scan={}
    for n in range(2,6):
        failures=[];minimum=1.;count=0
        for H in [.01,.05,.1,1/6-1e-6,2/3-.5,1/6+1e-6,.2,.3,.49,.5]:
            for rho,sigma in [(-.9,.8),(.9,.8),(0,.1)]:
                params=replace(p,H=H,rho=rho,sigma=sigma)
                for z in [.001,1,10,50,1-.5j,10-.5j]:
                    count+=1
                    try:
                        _,q=_checked_coefficients(z,5,params,n)
                        y=np.linspace(0,5**params.alpha,501)
                        condition=abs(np.polynomial.polynomial.polyval(y,q))/np.polynomial.polynomial.polyval(y,abs(q))
                        minimum=min(minimum,float(min(condition)))
                    except FloatingPointError as exc:
                        failures.append(str(exc))
        scan[n]={'cases':count,'failures':failures,'minimum_scaled_denominator':minimum}
    print(json.dumps({'benchmark':output,'scan':scan},indent=2))


if __name__=='__main__':main()
