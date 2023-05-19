import os

import numpy as np
from desilike import Parameter, ParameterPrior, ParameterArray, setup_logging
from desilike.samples import Chain, plotting, diagnostics


def get_chain(params, nwalkers=4, size=4000, seed=42):
    rng = np.random.RandomState(seed=seed)
    ndim = len(params)
    mean = np.zeros(ndim, dtype='f8')
    cov = np.diag(np.linspace(1., 25., ndim))
    cov += 0.1  # off-diagonal
    invcov = np.linalg.inv(cov)
    array = rng.multivariate_normal(mean, cov, size=(size, nwalkers))
    diff = array - mean
    logposterior = -0.5 * np.sum(diff.dot(invcov) * diff, axis=-1)
    chain = Chain(list(np.moveaxis(array, -1, 0)) + [logposterior], params=params + ['logposterior'])
    for iparam, param in enumerate(chain.params(derived=False)):
        param.update(fixed=False, value=mean[iparam])
    return mean, cov, chain


def test_misc():

    chain_dir = '_chains'
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    mean, cov, chain = get_chain(params, nwalkers=10)

    chain['like.a'].param.update(latex='a', prior=ParameterPrior(limits=(-10., 10.)))
    assert isinstance(list(chain), list)
    pb = chain['like.b'].param
    pb.update(prior=ParameterPrior(dist='norm', loc=1.))
    pb = Parameter.from_state(pb.__getstate__())
    chain['logposterior'] = np.zeros(chain.shape, dtype='f8')
    fn = os.path.join(chain_dir, 'chain.npy')
    chain.save(fn)
    base_fn = os.path.join(chain_dir, 'chain')
    chain.write_getdist(base_fn, ichain=0)
    chain2 = Chain.read_getdist(base_fn)
    chain.to_getdist()
    chain.interval('like.a')
    chain2 = chain.deepcopy()
    chain['like.a'] += 1
    chain2['like.a'].param.update(latex='answer')
    assert np.allclose(chain2['like.a'], chain['like.a'] - 1)
    assert chain2['like.a'].param.latex() != chain['like.a'].param.latex()
    size = chain2.size * 2
    chain2.extend(chain2)
    assert chain2.size == size
    assert chain == chain
    print(chain.choice(index=[0, 1], return_type=None).shape, chain.shape)
    assert chain.choice(index=[0, 1], return_type=None).shape[0] == 2
    chain.bcast(chain)
    chain.sendrecv(chain, source=0, dest=0)
    chain['like.a'].param.update(fixed=False)
    assert not chain[4:10]['like.a'].param.fixed
    assert not chain.concatenate(chain, chain)['like.a'].param.fixed
    assert np.all(np.array(chain.match(chain)[0]) == np.array(np.unravel_index(np.arange(chain.size), shape=chain.shape)))


def test_stats():
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    mean, cov, chain = get_chain(params, nwalkers=4, size=4000)
    assert chain.ravel().shape == (16000,)
    assert chain.shape == (4000, 4)

    try:
        from emcee import autocorr
        ref = autocorr.integrated_time(chain['like.a'].ravel(), quiet=True)
        assert np.allclose(diagnostics.integrated_autocorrelation_time(chain, params='like.a'), ref)
        assert len(diagnostics.integrated_autocorrelation_time(chain, params=['like.a'] * 2)) == 2
    except ImportError:
        pass

    chains = [chain] + [get_chain(params, seed=seed)[-1] for seed in range(44, 54)]
    assert np.allclose(diagnostics.gelman_rubin(chains, 'like.a', method='diag'), diagnostics.gelman_rubin(chains, 'like.a', method='eigen'))
    assert np.ndim(diagnostics.gelman_rubin(chains, 'like.a', method='eigen')) == 0
    assert diagnostics.gelman_rubin(chains, ['like.a'], method='eigen').shape == (1,)
    assert np.ndim(diagnostics.integrated_autocorrelation_time(chains, 'like.a')) == 0
    assert diagnostics.geweke(chains, params=['like.a'] * 2, first=0.25, last=0.75).shape == (2, len(chains))
    print(chain.to_stats(tablefmt='latex_raw'))


def test_bcast():
    from desilike.parameter import ParameterArray
    import mpytools as mpy

    mpicomm = mpy.COMM_WORLD

    array = mpy.array(np.ones(5))
    print(mpicomm.rank, type(mpy.bcast(array, mpiroot=0, mpicomm=mpicomm)))

    array = ParameterArray(np.ones(5), Parameter('a'))
    print(mpicomm.rank, type(mpy.bcast(array, mpiroot=0, mpicomm=mpicomm)))


def test_plot():

    chain_dir = '_chains'
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    chains = [get_chain(params, seed=ii)[-1] for ii in range(4)]
    plotting.plot_triangle(chains[0], fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_triangle(chains[0], params='like.*', fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_triangle([chains[0], chains[1].select(name=params[1:])], params=params, fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_trace(chains[0], fn=os.path.join(chain_dir, 'trace.png'))
    plotting.plot_autocorrelation_time(chains[0], fn=os.path.join(chain_dir, 'autocorrelation_time.png'))
    plotting.plot_gelman_rubin(chains, fn=os.path.join(chain_dir, 'gelman_rubin.png'))
    plotting.plot_geweke(chains, fn=os.path.join(chain_dir, 'geweke.png'))


def test_solved():
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    chain = get_chain(params, size=1000, nwalkers=5)[-1]
    chain['like.a'].param.update(derived='.auto')
    chain['like.b'].param.update(derived='.auto')
    array = np.zeros(chain.shape + (4,), dtype='f8')
    array[..., 1] = -0.3
    array[..., 3] = -0.3
    chain.set(ParameterArray(array, param='loglikelihood', derivs=[(), ('like.a',) * 2, ('like.a', 'like.b'), ('like.b',) * 2]))
    chain['logprior'] = chain['loglikelihood']

    assert np.allclose(chain.median('like.a'), np.median(chain['like.a']), atol=1.)
    print(chain.sample_solved().to_stats(tablefmt='pretty'))
    print(chain.to_stats(tablefmt='pretty'))


def test_cholesky():
    ndim = 4
    cov = np.random.uniform(size=(ndim, ndim))
    cov += 2. * np.eye(ndim)
    L = np.linalg.cholesky(cov)
    noise = np.random.standard_normal(size=(ndim, 1000))
    values = np.sum(noise[None, ...] * L[:, :, None], axis=1)
    print(cov)
    print(np.cov(values))


if __name__ == '__main__':

    setup_logging()

    test_bcast()
    test_misc()
    test_stats()
    test_plot()
    test_solved()
    # test_cholesky()
