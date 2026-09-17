"""Response coefficient normalization, sharing, partition and serialization."""
import numpy as np
import jax
import pytest
from desilike import build,get_params
from desilike.base import share_params
from desilike.theories.galaxy_clustering import (
 DensitySplitResponsePowerSpectrumBasis as Basis,
 DensitySplitMatterPowerSpectrumMultipoles as Cross,
 DensitySplitPairPowerSpectrumMultipoles as Pair,
 WindowedDensitySplitResponseBasis as Windowed,
 response_residual_templates)


def kernel():
 return Basis(k=[.025,.055,.095],z=.5,ells=(0,2,4),engine='eisenstein_hu',
              nq=8,nx=8,nphi=4,nklin=128,nmu=12)


def test_polynomial_exact_and_jit():
 obj=kernel();values=np.asarray(build(obj)({}))
 assert values.shape==(21,6,3,3)
 for x in (np.zeros(5),np.array([.02,.1,-.01,-.2,-.3])):
  np.testing.assert_allclose(obj.evaluate(x),obj.assemble(x[:3],x[3:]),atol=1.e-8,rtol=1.e-10)
 clone=jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(obj)))
 np.testing.assert_allclose(clone.evaluate(np.ones(5)),obj.evaluate(np.ones(5)))
 assert np.isfinite(jax.jacfwd(obj.evaluate)(np.zeros(5))).all()


def test_parameter_sharing_and_counts():
 b=kernel();kw=dict(k=b.k,z=b.z,ells=b.ells,kernels=b,model='response')
 x=Cross(quantiles=(1,2,4,5),**kw);p=Pair(pairs=tuple((a,c) for i,a in enumerate((1,2,4,5)) for c in (1,2,4,5)[i:]),**kw)
 share_params([x,p])
 for name in ('c1q1','c2q5','a0','a2','a4','beta0','beta2'):assert getattr(x,name) is getattr(p,name)
 for name in Basis.parameter_names:
  state=getattr(x,name).prior.__getstate__();assert state['loc']==0 and state['scale']==1
 # Default cosmology also varies tau/mass, so count model parameters separately.
 names=get_params(p).select(varied=True).names()
 assert sum(n.startswith(('c1q','c2q','s0q','s2q','s2parallelq')) or n in Basis.parameter_names for n in names)==43
 assert x.kernels is p.kernels


def test_q3_partition_and_residual_templates():
 b=kernel();fractions=np.array([.1,.2,.3,.15,.25])
 pairs=tuple((a,c) for a in range(1,6) for c in range(a,6))
 p=Pair(k=b.k,z=b.z,ells=b.ells,kernels=b,model='response',pairs=pairs,quantile_fractions=fractions)
 x=Cross(k=b.k,z=b.z,ells=b.ells,kernels=b,model='response',quantiles=(1,2,3,4,5),quantile_fractions=fractions)
 share_params([x,p]);rng=np.random.default_rng(4)
 params={n:rng.normal() for n in get_params(p).names() if n.startswith(('c1q','c2q','s0q','s2q','s2parallelq'))}
 y=np.asarray(build(p)(params));z=np.asarray(build(x)({name:value for name,value in params.items() if name in get_params(x)}))
 np.testing.assert_allclose(np.einsum('q,qlk->lk',fractions,z),0.,atol=1.e-10)
 matrix=np.zeros((5,5,3,3))
 for (a,c),power in zip(pairs,y):matrix[a-1,c-1]=matrix[c-1,a-1]=power
 np.testing.assert_allclose(np.einsum('q,qrlk->rlk',fractions,matrix),0.,atol=1.e-9)
 assert not any('q3' in n for n in get_params(p).names())
 t=np.asarray(response_residual_templates([.1],(0,2,4)))[:,:,0]
 np.testing.assert_allclose(t,[[1,0,0],[1,0,0],[1/3,2/3,0]])
 with pytest.raises(ValueError,match='AP unity'):kernel().__class__(k=[.03],ap=True)


def test_window_and_chebyshev_roundtrip(tmp_path):
 from desilike.emulators import Emulator, CalculatorEmulator, Space
 b=kernel();matrix=np.arange(81).reshape(9,9)/100
 w=Windowed(b,matrix,k=b.k)
 graph=build(w);answer=np.asarray(graph({}))
 np.testing.assert_allclose(answer.reshape(21,6,-1),np.asarray(b.coefficients).reshape(21,6,-1)@matrix.T)
 emulator=Emulator(w, Space(bounds={'h': (float(get_params(w)['h'].value)-.01, float(get_params(w)['h'].value)+.01)}), budget=1).train()
 path=tmp_path/'response.h5';emulator.write(path)
 restored=CalculatorEmulator.read(path).to_calculator()
 p=Cross(k=b.k,z=b.z,ells=b.ells,kernels=restored,model='response',quantiles=(1,))
 actual=np.asarray(build(p)({'a0':.1,'beta0':-.3,'c2q1':2.}))
 truth=Cross(k=b.k,z=b.z,ells=b.ells,kernels=w,model='response',quantiles=(1,))
 expected=np.asarray(build(truth)({'a0':.1,'beta0':-.3,'c2q1':2.}))
 np.testing.assert_allclose(actual,expected,rtol=1.e-9,atol=1.e-7)


def test_windowed_residuals_and_restricted_pairs(tmp_path):
    from desilike.emulators import Emulator, CalculatorEmulator, Space
    b=kernel();matrix=np.eye(9)+np.random.default_rng(9).normal(size=(9,9))*.02
    projected=Windowed(b,matrix,k=b.k)
    options=dict(k=b.k,z=b.z,ells=b.ells,model='response',pairs=((1,1),(1,2)))
    before=Pair(kernels=b,**options)
    after=Pair(kernels=projected,**options)
    names=get_params(after).names()
    assert 's0q2q2' not in names and 's2q2q2' not in names
    assert after.residual_pairs==((1,1),(1,2))
    point=dict(c1q1=-2.,c1q2=-1.,c2q1=4.,c2q2=-1.,beta0=.2,
               s0q1q1=1000.,s0q1q2=-500.,s2q1q1=100.,s2parallelq1q2=300.)
    expected=np.asarray(build(before)(point)).reshape(2,-1)@matrix.T
    np.testing.assert_allclose(build(after)(point).reshape(2,-1),expected,atol=1.e-8,rtol=1.e-12)
    path=tmp_path/'windowed.h5'
    Emulator(projected, Space(bounds={'h': (float(get_params(projected)['h'].value)-.01, float(get_params(projected)['h'].value)+.01)}), budget=0).train().write(path)
    restored=CalculatorEmulator.read(path).to_calculator()
    emulated=Pair(kernels=restored,**options)
    np.testing.assert_allclose(build(emulated)(point).reshape(2,-1),expected,atol=1.e-8,rtol=1.e-12)
    # With only a raw hexadecapole all three residual templates vanish.
    high=Basis(k=b.k,z=b.z,ells=(4,),engine='eisenstein_hu',nq=8,nx=8,nphi=4,nklin=128,nmu=12)
    only=Pair(k=high.k,z=high.z,ells=high.ells,kernels=high,model='response',pairs=((1,1),))
    assert all(not p.varied for p in only.residual_params)
