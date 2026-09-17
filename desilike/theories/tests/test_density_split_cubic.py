"""Independent cubic contractions and modern calculator regression tests."""
from itertools import permutations, combinations
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desilike import Calculator, Parameter, build, get_params
from desilike.base import share_params
from desilike.theories.galaxy_clustering import (
    DensitySplitPowerSpectrumKernels as Kernels,
    DensitySplitMatterPowerSpectrumMultipoles as Cross,
    DensitySplitPairPowerSpectrumMultipoles as Pair,
)
from desilike.theories.galaxy_clustering.density_split_matter import (
    _fg3, _matter_z3, _matter_trispectrum, _sobol_normals,
    _trispectrum_integral, _cubic_convolutions, _loop_quadrature,
    _interpolate_poles,
)


def fg(vectors):
    """Unsymmetrized SPT recursion, independent NumPy implementation."""
    if len(vectors) == 1:
        return 1., 1.
    n = len(vectors)
    density, velocity = 0., 0.
    for m in range(1, n):
        a, b = np.sum(vectors[:m], axis=0), np.sum(vectors[m:], axis=0)
        fa, ga = fg(vectors[:m]); fb, gb = fg(vectors[m:])
        alpha = np.dot(a+b, a)/np.dot(a,a)
        beta = np.dot(a+b,a+b)*np.dot(a,b)/(2*np.dot(a,a)*np.dot(b,b))
        density += ga*((2*n+1)*alpha*fb + 2*beta*gb)
        velocity += ga*(3*alpha*fb + 2*n*beta*gb)
    return np.array([density, velocity])/((2*n+3)*(n-1))


def mapping_z3(vectors, f):
    """Coefficient of three distinct plane waves in (1+delta) exp(-ik_z u)."""
    def multiply(left, right):
        product = {}
        for a, va in left.items():
            for b, vb in right.items():
                if a & b == 0:
                    product[a | b] = product.get(a | b, 0.) + va*vb
        return product

    delta, velocity = {0: 1.}, {}
    for n in (1,2,3):
        for subset in combinations(range(3), n):
            fsum, gsum = np.sum([fg(vectors[list(order)]) for order in permutations(subset)], axis=0)
            mask = sum(1 << i for i in subset)
            momentum = vectors[list(subset)].sum(axis=0)
            delta[mask] = fsum
            velocity[mask] = f*momentum[2]/np.dot(momentum,momentum)*gsum
    kz = vectors.sum(axis=0)[2]
    result = delta.get(7, 0.)
    term = delta
    for n in (1,2,3):
        term = multiply(term, velocity)
        result += kz**n / (1,2,6)[n-1] * term.get(7, 0.)
    return result/6.


def test_z3_recursion_mapping_and_symmetry():
    vectors = np.array([[.04,.01,-.02], [.01,-.06,.03], [-.07,.02,.09]])
    expected = np.mean([fg(vectors[list(order)]) for order in permutations(range(3))], axis=0)
    np.testing.assert_allclose(_fg3(*vectors), expected, rtol=1.e-12)
    for growth in (0., .8):
        reference = mapping_z3(vectors, growth)
        for order in permutations(range(3)):
            np.testing.assert_allclose(_matter_z3(*vectors[list(order)], growth), reference, rtol=1.e-12)
    np.testing.assert_allclose(_matter_z3(*vectors, 0.), expected[0])


def test_trispectrum_permutations_amplitude_and_soft_limits():
    grid = np.geomspace(1.e-9, 5., 512)
    power = grid*np.exp(-grid**2)
    run = jax.jit(_matter_trispectrum)
    for soft in (1., 1.e-3, 1.e-6):
        vectors = np.array([[.04,.01,-.02], [.01,-.06,.03], [-.07,.02,.09]])
        vectors[0] *= soft
        vectors = np.vstack([vectors, -vectors.sum(axis=0)])
        reference = run(vectors, grid, power, .8)
        assert np.isfinite(reference)
        for order in permutations(range(4)):
            np.testing.assert_allclose(run(vectors[list(order)], grid, power, .8), reference, rtol=1.e-8, atol=1.e-10)
        np.testing.assert_allclose(run(vectors, grid, 2*power, .8), 8*reference, rtol=1.e-10)
    a,b = vectors[1:3]
    collapsed = np.array([a,-a,b,-b])
    assert np.isfinite(run(collapsed, grid, power, .8))


def test_trispectrum_independent_connected_wick_trees():
    """Enumerate Gaussian contractions without assuming the 6 and 4 prefactors."""
    def pairings(slots):
        if not slots:
            yield []
        else:
            for j in range(1,len(slots)):
                for tail in pairings(slots[1:j]+slots[j+1:]):
                    yield [(slots[0],slots[j])]+tail

    vectors=np.array([[.04,.01,-.02],[.01,-.06,.03],[-.07,.02,.09],[.02,.03,-.10]])
    np.testing.assert_allclose(vectors.sum(axis=0),0.,atol=1.e-15)
    for growth in (0.,.8):
        result=0.
        for orders in set(permutations((3,1,1,1))) | set(permutations((2,2,1,1))):
            slots=[(vertex,i) for vertex,n in enumerate(orders) for i in range(n)]
            for edges in pairings(slots):
                if any(a[0]==b[0] for a,b in edges):
                    continue
                adjacency={i:set() for i in range(4)}
                for a,b in edges:
                    adjacency[a[0]].add(b[0]);adjacency[b[0]].add(a[0])
                reached={0}
                for _ in range(3):
                    reached |= set.union(*(adjacency[i] for i in reached))
                if len(reached)!=4:
                    continue
                legs={i:[] for i in range(4)}
                for a,b in edges:
                    reached={a[0]}
                    for _ in range(3):
                        reached |= set.union(*(adjacency[i] for i in reached))-{b[0]}
                    momentum=vectors[list(reached)].sum(axis=0)
                    legs[a[0]].append(momentum);legs[b[0]].append(-momentum)
                contribution=1.
                for incoming in legs.values():
                    incoming=np.asarray(incoming)
                    if len(incoming)==1:
                        a=incoming[0];kernel=1+growth*a[2]**2/np.dot(a,a)
                    elif len(incoming)==2:
                        a,b=incoming;total=a+b
                        f2,g2=np.mean([fg(incoming),fg(incoming[::-1])],axis=0)
                        ua=a[2]/np.dot(a,a);ub=b[2]/np.dot(b,b)
                        kernel=f2+growth*total[2]**2/np.dot(total,total)*g2+.5*growth*total[2]*(ua+ub)+.5*growth**2*total[2]**2*ua*ub
                    else:
                        kernel=mapping_z3(incoming,growth)
                    contribution*=kernel
                result+=contribution
        grid=np.geomspace(1.e-8,2.,128)
        np.testing.assert_allclose(_matter_trispectrum(vectors,grid,np.ones_like(grid),growth),result,rtol=1.e-12)


def test_gaussian_importance_normalization_and_derivative():
    # A constant trispectrum integrates to the Gaussian window volume / 6.
    from unittest.mock import patch
    normals = jnp.asarray(_sobol_normals(6,42,16))
    grid = jnp.geomspace(1.e-6,5.,64)
    k = np.array([.01,.08,.12]); radius = 10.
    with patch('desilike.theories.galaxy_clustering.density_split_matter._matter_trispectrum',
               side_effect=lambda vectors, grid, power, f: jnp.ones(vectors[0].shape[:-1])*power[0]):
        run = jax.jit(lambda amplitude: _trispectrum_integral(k,.4,grid,amplitude*jnp.ones_like(grid),.8,radius,normals))
        expected = np.exp(-(k*radius)**2/6.)/(6*3**1.5*(2*np.pi)**3*radius**6)
        np.testing.assert_allclose(run(1.), expected, rtol=1.e-12)
        np.testing.assert_allclose(jax.jacfwd(run)(1.), expected, rtol=1.e-12)


def test_gaussian_triple_convolution_and_signed_interpolation():
    radius = 10.; amplitude = 1.e4
    grid = np.geomspace(1.e-9,8.,1024)
    inner = np.geomspace(1.e-8,6.,8192)
    p22 = .5*amplitude**2*(np.pi/(2*radius**2))**1.5/(2*np.pi)**3*np.exp(-.5*(inner*radius)**2)
    poles = np.zeros((2,5,len(inner))); poles[1,0] = p22
    k = np.array([.001,.05,.12])
    actual = _cubic_convolutions(k,.6,grid,np.full_like(grid,amplitude),0.,radius,
                                _loop_quadrature(160,48,16,1.e-7,5.),inner,poles)[1]/3.
    expected = amplitude**3/6. * np.pi**3/(3**1.5*radius**6*(2*np.pi)**6)*np.exp(-(k*radius)**2/3.)
    np.testing.assert_allclose(actual,expected,rtol=2.e-5)
    signed = _interpolate_poles(np.array([1.,2.,4.]),0.,np.array([1.,4.]),jnp.array([[-1.,1.]]),(0,))
    np.testing.assert_allclose(signed,[-1.,0.,1.],atol=1.e-14)
    assert np.isnan(_interpolate_poles(.5,0.,np.array([1.,4.]),jnp.array([[-1.,1.]]),(0,)))


class SmoothCosmology(Calculator):
    def __init__(self):
        self.amplitude = Parameter('amplitude', value=1., prior=dict(limits=(.5,2.)))
    def add_requirements(self, requirements):
        pass
    def __call__(self):
        self.A = self.amplitude.value
        return self.A
    def get_fourier(self):
        return self
    def pk(self, *, k, **kwargs):
        return self.A*1.e4*jnp.exp(-2.*jnp.asarray(k)**2)
    def sigma8_z(self, *, of, **kwargs):
        return (.7 if of == 'theta_cb' else 1.)*jnp.sqrt(self.A)
    def tree_flatten(self):
        return [self.A], None
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj=object.__new__(cls); obj.A=children[0]; return obj


def small_kernels(order=3, **kwargs):
    options = dict(k=np.array([.02,.06]), z=.5, ells=(0,2,4), cosmo=SmoothCosmology(),
                   operator_order=order,nq=24,nx=12,nphi=16,nmu=7,
                   cubic_options=dict(log2_nodes=6,chunk_size=32,nk=2,ninner=24))
    options.update(kwargs)
    return Kernels(**options)


def test_cubic_calculator_nested_response_sharing_and_partition():
    kernels = small_kernels()
    kwargs = dict(k=kernels.k,z=.5,ells=kernels.ells,model='cubic',kernels=kernels)
    cross = Cross(quantiles=(1,2,3,4,5),quantile_fractions=(.1,.2,.3,.15,.25),**kwargs)
    pair = Pair(pairs=((1,1),(1,5),(5,5)),**kwargs)
    share_params([cross,pair])
    assert cross.c3q1 is pair.c3q1
    run = build(cross)
    values = dict(c3q1=2.,c3q2=-1.,c3q4=3.,c3q5=4.)
    result = np.asarray(run(values))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.einsum('q,qlk->lk',cross.quantile_fractions,result),0.,atol=1.e-10)
    cubic = np.asarray(build(kernels)())
    quadratic = np.asarray(build(small_kernels(order=2))())
    np.testing.assert_allclose(cubic[[0,1,3,4,6]],quadratic,rtol=1.e-12,atol=1.e-10)
    # Building the shared kernel supersedes the earlier cross graph.
    zero = np.asarray(build(cross)({name:0. for name in values}))
    assert np.isfinite(zero).all()
    names = [p.name for p in get_params(cross)]
    assert 'c3q3' not in names and sum(name.startswith('c3q') for name in names)==4
    assert not any(name.lower().startswith('folps') for name in sys.modules)


def test_cubic_configuration_validation():
    with pytest.raises(ValueError,match='operator_order'):
        Kernels(k=[.02],operator_order=4)
    with pytest.raises(ValueError,match='must match'):
        build(Cross(k=[.02,.06],z=.5,ells=(0,2,4),model='quadratic',kernels=small_kernels()))
    with pytest.raises(ValueError,match='fractions'):
        Cross(model='cubic',quantile_fractions=[0.,.2,.2,.3,.3])


def test_intrinsic_cubic_angular_termination():
    calculator=small_kernels(ells=tuple(range(0,17,2)),nmu=9)
    basis=np.asarray(build(calculator)())
    assert np.max(np.abs(basis[:,7:])) < 1.e-7
    assert np.max(np.abs(basis[:,6])) > 1.e-6
    real=small_kernels(ells=(0,),rsd=False)
    assert np.isfinite(build(real)()).all()
