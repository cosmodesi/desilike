"""Independent one-loop matter checks, without an external PT implementation."""
import itertools
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.integrate import quad
from desilike import build
from desilike.theories.galaxy_clustering import (
    DensitySplitPowerSpectrumKernels, DensitySplitOneLoopPowerSpectrumKernels,
)
from desilike.theories.galaxy_clustering.density_split_matter import (
    _collapsed_matter_z3, _matter_one_loop, _matter_loop_nodes,
)


def independent_z3(a,b,c,f):
    def fg2(a,b):
        aa,bb,ab=a@a,b@b,a@b
        return 5/7+ab/2*(1/aa+1/bb)+2/7*ab**2/(aa*bb),3/7+ab/2*(1/aa+1/bb)+4/7*ab**2/(aa*bb)
    def alpha(a,b): return (a+b)@a/(a@a)
    def beta(a,b): return ((a+b)@(a+b))*(a@b)/(2*(a@a)*(b@b))
    ff=gg=0.
    for u,v,w in itertools.permutations((a,b,c)):
        f2,g2=fg2(v,w);_,g12=fg2(u,v)
        ff+=(7*alpha(u,v+w)*f2+2*beta(u,v+w)*g2+g12*(7*alpha(u+v,w)+2*beta(u+v,w)))/108
        gg+=(3*alpha(u,v+w)*f2+6*beta(u,v+w)*g2+g12*(3*alpha(u+v,w)+6*beta(u+v,w)))/108
    k=a+b+c;kz=k[2];answer=ff+f*kz*kz/(k@k)*gg
    for u,v,w in ((a,b,c),(b,a,c),(c,a,b)):
        f2,g2=fg2(v,w);v2=(v+w)[2]/((v+w)@(v+w))*g2;v1=u[2]/(u@u)
        answer+=(f*kz*(v1*f2+v2)+f*f*kz*kz*v1*v2)/3
        answer+=f*f*kz*kz/6*v[2]*w[2]/((v@v)*(w@w))
    return answer+f**3*kz**3/6*np.prod([v[2]/(v@v) for v in (a,b,c)])


def test_collapsed_limit_permutation_and_real_space():
    k=np.array([.08,.02,.05]);q=np.array([.17,-.11,.07])
    for f in (0.,.8):
        value=float(_collapsed_matter_z3(k,q,f))
        np.testing.assert_allclose(value,_collapsed_matter_z3(k,-q,f),rtol=1.e-13)
        for direction in (np.array([1.,.2,-.3]),np.array([-.2,1.,.4])):
            # Average opposite approaches removes the first-order displacement.
            limit=np.mean([independent_z3(k,q,-q+sign*1.e-4*direction,f) for sign in (-1,1)])
            np.testing.assert_allclose(value,limit,rtol=2.e-5,atol=1.e-9)


def test_real_space_integrals_and_amplitude_scaling():
    grid=np.geomspace(1.e-7,3.,8192);power=grid*np.exp(-(grid/.2)**2)
    nodes=_matter_loop_nodes(256,96,8)
    run=jax.jit(lambda amplitude:_matter_one_loop(jnp.array([.08]),jnp.array([.4]),grid,amplitude*power,0.,nodes,1.e-5,2.))
    measured=np.array(run(1.))[:,0]
    def pk(q): return q*np.exp(-(q/.2)**2)
    k=.08
    def integrand(r):
        if abs(r-1)<1.e-10: bracket=-88.
        else: bracket=12/r**2-158+100*r*r-42*r**4+3/r**3*(r*r-1)**3*(7*r*r+2)*np.log(abs((1+r)/(1-r)))
        return pk(k*r)*bracket
    p13=k**3*pk(k)/(252*(2*np.pi)**2)*sum(quad(integrand,a,b,epsabs=1.e-11,limit=200)[0]
                                                     for a,b in ((1.e-5/k,1.),(1.,2/k)))
    x,wx=np.polynomial.legendre.leggauss(256)
    def radial(q):
        p=np.sqrt(k*k+q*q-2*k*q*x);angle=(k*x-q)/p
        f2=5/7+.5*angle*(q/p+p/q)+2/7*angle**2
        return q*q*pk(q)*np.sum(wx*2*f2*f2*pk(p)*((p>=1.e-5)&(p<=2.)))/(2*np.pi)**2
    p22=sum(quad(radial,a,b,epsabs=1.e-12,limit=200)[0] for a,b in ((1.e-5,k),(k,2.)))
    np.testing.assert_allclose(measured[:2],[p22,p13],rtol=3.e-4)
    np.testing.assert_allclose(measured[2],sum(measured[:2]),rtol=1.e-12)
    np.testing.assert_allclose(run(2.),4*run(1.),rtol=1.e-12)
    assert np.isfinite(np.array(jax.jacfwd(run)(1.))).all()


def test_calculator_smoothing_counterterms_and_serialization():
    options=dict(k=np.array([.03,.08]),z=.5,ells=(0,2,4,6,8),engine='eisenstein_hu',
                 nq=16,nx=12,nphi=8,nklin=256)
    obj=DensitySplitOneLoopPowerSpectrumKernels(**options)
    run=build(obj,output=lambda:(obj.kernels,obj.matter_pieces,obj.counterterms,obj.with_counterterms(jnp.array([1.,2.,3.]))))
    basis,matter,ct,eft=map(np.asarray,jax.jit(run)({}))
    old=np.asarray(build(DensitySplitPowerSpectrumKernels(**options,qmin=1.e-5,qmax=10.))({}))
    w=np.exp(-.5*(options['k']*10)**2)
    np.testing.assert_allclose(basis[[1,3,4]],old[[1,3,4]],rtol=1.e-12,atol=1.e-9)
    np.testing.assert_allclose(basis[0],w*sum(matter),rtol=1.e-6)
    np.testing.assert_allclose(basis[2],w*w*sum(matter),rtol=1.e-6)
    np.testing.assert_allclose(ct[:,1],ct[:,0]*w,rtol=1.e-12,atol=1.e-10)
    np.testing.assert_allclose(ct[:,2],ct[:,0]*w*w,rtol=1.e-12,atol=1.e-10)
    np.testing.assert_allclose(eft[0],basis[0]+np.einsum('a,alk->lk',[1,2,3],ct[:,1]),rtol=1.e-12)
    leaves,tree=jax.tree_util.tree_flatten(obj)
    clone=jax.tree_util.tree_unflatten(tree,leaves)
    np.testing.assert_allclose(clone.with_counterterms(jnp.zeros(3)),basis)
    with pytest.raises(ValueError,match='only operator_order'):
        DensitySplitOneLoopPowerSpectrumKernels([.1],operator_order=3)


def test_ap_identity_and_finite_cosmology_derivative():
    options=dict(k=np.array([.04]),z=.5,ells=(0,2,4),engine='eisenstein_hu',
                 nq=8,nx=8,nphi=4,nklin=128)
    plain=jax.jit(build(DensitySplitOneLoopPowerSpectrumKernels(**options)))
    mapped=jax.jit(build(DensitySplitOneLoopPowerSpectrumKernels(**options,ap=True,nmu=12)))
    # The AP path uses a distinct log interpolation grid, as in the existing
    # calculator; test identity to its interpolation accuracy rather than zero.
    np.testing.assert_allclose(mapped({}),plain({}),rtol=2.e-3,atol=1.e-6)
    derivative=jax.jacfwd(lambda h:mapped({'h':h}))(jnp.array(.67))
    assert np.isfinite(np.asarray(derivative)).all()


def test_soft_cutoff_and_signed_loop_values():
    grid=np.geomspace(1.e-8,4.,8192);power=grid*np.exp(-(grid/.2)**2)
    nodes=_matter_loop_nodes(240,80,16)
    def run(qmin):
        return np.asarray(_matter_one_loop(jnp.array([.03,.08]),jnp.array([.1,.8]),grid,power,.8,nodes,qmin,2.))
    low,lower=run(1.e-5),run(1.e-6)
    assert (low[1]<0.).all()  # P13 must remain signed.
    np.testing.assert_allclose(low,lower,rtol=3.e-4,atol=1.e-12)


def test_exact_collinear_limit():
    k=jnp.array([.03,.02,.05]);mu2=k[2]**2/jnp.sum(k*k)
    for f in (0.,.8):
        for ratio in (-2.,-1.,1.,2.):
            value=_collapsed_matter_z3(k,ratio*k,f)
            np.testing.assert_allclose(value,-(1+f*mu2)**3/(6*ratio**2),rtol=1.e-13)
            assert np.isfinite(np.asarray(jax.grad(lambda growth:_collapsed_matter_z3(k,ratio*k,growth))(f)))


def test_chebyshev_roundtrip_without_folps(tmp_path):
    import subprocess
    import sys
    source = r"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.lower().startswith('folps'):
            raise ImportError('FOLPS blocked')
sys.meta_path.insert(0, Block())
import numpy as np
from desilike import build
from desilike.emulators import Emulator, CalculatorEmulator, Space
from desilike import get_params
from desilike.theories.galaxy_clustering import DensitySplitOneLoopPowerSpectrumKernels
obj = DensitySplitOneLoopPowerSpectrumKernels(k=[.04], ells=(0,2,4),
    engine='eisenstein_hu', nq=8, nx=8, nphi=4, nklin=128)
graph = build(obj)
emulator = Emulator(obj, Space(bounds={'h': (float(get_params(obj)['h'].value)-.01, float(get_params(obj)['h'].value)+.01)}), budget=1).train()
emulator.write(sys.argv[1])
restored = CalculatorEmulator.read(sys.argv[1]).to_calculator()
np.testing.assert_allclose(build(restored, output=lambda: restored.kernels)({}), graph({}), rtol=1.e-10, atol=1.e-8)
np.testing.assert_allclose(restored.with_counterterms([.01,.1,-.01]),
    obj.with_counterterms([.01,.1,-.01]), rtol=1.e-10, atol=1.e-8)
"""
    subprocess.run([sys.executable, '-c', source, str(tmp_path/'one_loop.h5')],
                   check=True, timeout=180)
