"""Independent normalization and perturbative matching of response completion."""
import numpy as np
import jax
import jax.numpy as jnp
from desilike import build
from desilike.theories.galaxy_clustering import DensitySplitResponsePowerSpectrumKernels
from desilike.theories.galaxy_clustering.density_split_matter import (
    _quadratic_propagator, _quadratic_spectra, _loop_quadrature, _matter_loop_nodes, _matter_one_loop)


def test_response_cyclic_decomposition():
    # Compare unfurled bispectrum integration with independent folded Gram terms.
    grid=np.geomspace(1.e-7,3.,8192);power=grid*np.exp(-(grid/.2)**2)
    q=_loop_quadrature(240,96,32,1.e-5,2.)
    nodes=_matter_loop_nodes(240,96,32)
    k,mu=jnp.array([.02,.08]),jnp.array([.3,.8]);f=.8
    gamma=np.asarray(_quadratic_propagator(k,mu,grid,power,f,10.,q))
    old=np.asarray(_quadratic_spectra(k,mu,grid,power,f,10.,q))
    gram=np.asarray(_matter_one_loop(k,mu,grid,power,f,nodes,1.e-5,2.,radius=10.))
    pl=np.exp(np.interp(np.log(k),np.log(grid),np.log(np.maximum(power,1.e-300))))
    np.testing.assert_allclose(gamma*(1+f*mu**2)*pl+gram[3],old[0],rtol=3.e-4,atol=1.e-12)
    np.testing.assert_allclose(gram[4],old[1],rtol=3.e-4,atol=1.e-12)
    assert np.all(gram[0]*gram[4]-gram[3]**2>=-1.e-15)


def make_synthetic():
    obj=object.__new__(DensitySplitResponsePowerSpectrumKernels)
    obj.components=jnp.array([2.,1.3,-.1,.3,.2,.03,.07,.2,.8,.6,.4,.3])[:,None,None]
    obj.projection=jnp.ones((1,1))
    return obj


def test_matched_expansion_and_shared_response():
    obj=make_synthetic();original=obj.components
    def at_amplitude(amplitude):
        scale=jnp.array([amplitude,1.,amplitude,amplitude,amplitude**2,
            amplitude**2,amplitude**2,amplitude,1.,1.,1.,1.])
        obj.components=original*scale[:,None,None]
        return obj.assemble()[:,0,0]-obj.assemble(truncated=True)[:,0,0]
    large=np.asarray(at_amplitude(1.e-3));small=np.asarray(at_amplitude(5.e-4))
    np.testing.assert_allclose(large,8*small,rtol=.01,atol=1.e-14)
    obj.components=original
    p=np.asarray(obj.assemble(response=(.2,-.1)))[:,0,0]
    np.testing.assert_allclose(p[[1,3,4]], [.8*p[0],.64*p[0],.8*p[2]])
    assert p[0]*p[5]>=p[2]**2
    derivative=jax.jacfwd(lambda beta:obj.assemble(response=beta))(jnp.zeros(2))
    assert np.isfinite(derivative).all()


def test_exact_factor_and_regulator_dependence():
    obj=make_synthetic();pl,z1,gm,g2,mm,m2,two,d,w,*_=np.asarray(obj.components[:,0,0])
    p=np.asarray(obj.assemble())[:,0,0]
    np.testing.assert_allclose(p[2],np.exp(-2*d)*((z1+gm+d*z1)*g2*pl+m2))
    np.testing.assert_allclose(p[5],np.exp(-2*d)*(g2*g2*pl+two))
    for damping in (.5,1.,2.):
        p=np.asarray(obj.assemble(damping=damping))[:,0,0]
        assert p[0]*p[5]>=p[2]**2


def test_calculator_jit_roundtrip():
    obj=DensitySplitResponsePowerSpectrumKernels(k=[.04],ells=(0,2,4),engine='eisenstein_hu',
        nq=16,nx=12,nphi=8,nklin=256,nmu=12)
    graph=build(obj)
    values=np.asarray(jax.jit(graph)({}))
    leaves,tree=jax.tree_util.tree_flatten(obj)
    clone=jax.tree_util.tree_unflatten(tree,leaves)
    np.testing.assert_allclose(clone.assemble()[1:],values,atol=1.e-8)
    assert np.isfinite(jax.jacfwd(lambda h:graph({'h':h}))(.67)).all()


def test_real_space_long_mode_response():
    # For PL=q exp[-(q/a)^2], expanding W(|k-q|) cancels the odd soft pole.
    # The finite radial integral is analytic and includes the raw O2 factor 1/2.
    radius,a=10.,.2
    coefficient=radius**2+a**-2
    expected=(34./21.-2*radius**2/(3*coefficient))/(4*np.pi**2*coefficient**2)
    grid=np.geomspace(1.e-9,3.,32768);power=grid*np.exp(-(grid/a)**2)
    quadrature=_loop_quadrature(240,96,4,1.e-7,2.)
    result=_quadratic_propagator(jnp.array([1.e-5]),jnp.array([.4]),grid,power,0.,radius,quadrature)
    np.testing.assert_allclose(result,expected,rtol=1.e-5)


def test_intrinsic_angular_reconstruction():
    kwargs=dict(k=[.02,.08],ells=(0,2,4,6,8,10,12,14,16),engine='eisenstein_hu',
                nq=16,nx=12,nphi=16,nklin=256,nmu=16)
    fast=DensitySplitResponsePowerSpectrumKernels(**kwargs)
    direct=DensitySplitResponsePowerSpectrumKernels(**kwargs,intrinsic_nmu=0)
    a=np.asarray(jax.jit(build(fast))({}));b=np.asarray(jax.jit(build(direct))({}))
    np.testing.assert_allclose(a,b,rtol=1.e-8,atol=1.e-8)
    np.testing.assert_allclose(fast.with_counterterms([.01,.1,-.01]),
        fast.assemble(matter=[.01,.1,-.01])[1:])


def test_folps_blocked_chebyshev_roundtrip(tmp_path):
    import subprocess
    import sys
    source = r"""
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.lower().startswith('folps'):raise ImportError('FOLPS blocked')
sys.meta_path.insert(0,Block())
import numpy as np
from desilike import build
from desilike.emulators import Emulator, CalculatorEmulator, Space
from desilike import get_params
from desilike.theories.galaxy_clustering import DensitySplitResponsePowerSpectrumKernels
obj=DensitySplitResponsePowerSpectrumKernels(k=[.04],ells=(0,2,4),engine='eisenstein_hu',nq=8,nx=8,nphi=4,nklin=128,nmu=12)
graph=build(obj);em=Emulator(obj, Space(bounds={'h': (float(get_params(obj)['h'].value)-.01, float(get_params(obj)['h'].value)+.01)}), budget=1).train();em.write(sys.argv[1])
clone=CalculatorEmulator.read(sys.argv[1]).to_calculator()
np.testing.assert_allclose(build(clone, output=lambda: clone.kernels)({}),graph({}),rtol=1e-10,atol=1e-8)
np.testing.assert_allclose(clone.assemble([.01,.1,-.01],[.1,-.2]),obj.assemble([.01,.1,-.01],[.1,-.2]),rtol=1e-10,atol=1e-8)
print('FOLPS-blocked Chebyshev and exact response assembly passed')
"""
    subprocess.run([sys.executable,"-c",source,str(tmp_path/"response.h5")],check=True,timeout=180)
