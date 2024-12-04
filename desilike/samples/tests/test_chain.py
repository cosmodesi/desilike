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
    chain = Chain(list(np.moveaxis(array, -1, 0)) + [logposterior], params=params + ['logposterior'], loglikelihood='LRG.loglikelihood')
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
    chain.set_derived('derived.a', chain['like.a'] * 2.)
    for ff in ['chain.npy', 'chain.npz']:
        fn = os.path.join(chain_dir, ff)
        chain.save(fn)
        chain2 = Chain.load(fn)
        assert chain2.params() == chain.params()
        assert chain2 == chain

    base_fn = os.path.join(chain_dir, 'chain')
    chain.write_getdist(base_fn, ichain=0)
    chain2 = Chain.read_getdist(base_fn, concatenate=True)
    chain3 = Chain.from_getdist(chain.to_getdist())
    for chain2 in [chain2, chain3]:
        for param in chain2.params():
            assert np.allclose(chain2[param], chain[param].ravel())
    base_fn = os.path.join(chain_dir, 'chain_multiple')
    Chain.write_getdist([chain, chain], base_fn)
    chain2 = Chain.read_getdist(base_fn)
    assert len(chain2) == 2
    for chain2 in chain2:
        for param in chain2.params():
            assert np.allclose(chain2[param], chain[param].ravel())
    chain2 = Chain.from_getdist(Chain.to_getdist([chain, chain]))
    assert len(chain2) == 2
    for chain2 in chain2:
        for param in chain2.params():
            assert np.allclose(chain2[param], chain[param].ravel())
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
    assert chain.concatenate(chain, chain)._loglikelihood == 'LRG.loglikelihood'
    assert np.all(np.array(chain.match(chain)[0]) == np.array(np.unravel_index(np.arange(chain.size), shape=chain.shape)))
    print(chain.mean('like.a'), chain.std('like.a'))


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
    print(chain.to_stats(tablefmt='list_latex'))
    assert isinstance(chain.to_stats(tablefmt='list')[0], list)


def test_bcast():
    from desilike.parameter import ParameterArray
    import mpytools as mpy

    mpicomm = mpy.COMM_WORLD

    array = mpy.array(np.ones(5))
    print(mpicomm.rank, type(mpy.bcast(array, mpiroot=0, mpicomm=mpicomm)))

    array = ParameterArray(np.ones(5), Parameter('a'))
    print(mpicomm.rank, type(mpy.bcast(array, mpiroot=0, mpicomm=mpicomm)))


def test_plot():
    from matplotlib import pyplot as plt

    chain_dir = '_chains'
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    chains = [get_chain(params, seed=ii)[-1] for ii in range(4)]
    plotting.plot_triangle(chains[0], fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_triangle(chains[0], params=chains[0].params(varied=True), fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_triangle(chains[0], params='like.*', fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_triangle([chains[0], chains[1].select(name=params[1:])], params=params, fn=os.path.join(chain_dir, 'triangle.png'))
    plotting.plot_trace(chains[0], fn=os.path.join(chain_dir, 'trace.png'))
    plotting.plot_autocorrelation_time(chains[0], fn=os.path.join(chain_dir, 'autocorrelation_time.png'))
    plotting.plot_gelman_rubin(chains, fn=os.path.join(chain_dir, 'gelman_rubin.png'))
    plotting.plot_gelman_rubin(chains[0], nsplits=8, fn=os.path.join(chain_dir, 'gelman_rubin.png'))
    plotting.plot_geweke(chains, fn=os.path.join(chain_dir, 'geweke.png'))
    plt.close('all')

    from desilike.samples import Profiles, ParameterBestFit, ParameterContours, ParameterCovariance, Samples, utils
    chain = chains[0]
    params = chain.params(name=params)
    profiles1 = Profiles()
    profiles1.set(start=Samples([[chain.mean(param)] for param in params], params=params))
    profiles1.set(bestfit=ParameterBestFit([[chain.mean(param)] for param in params], params=params))
    profiles1.set(covariance=ParameterCovariance(chain.covariance(params=params), params=params))
    #plotting.plot_triangle([chain, profiles1], labels=['chain', 'profiles'], show=True)

    profile, contours = [], []
    for param in params:
        x = np.linspace(np.min(chain[param]), np.max(chain[param]), 100)
        profile.append(np.column_stack([x, -0.5 * (x - chain.mean(param))**2 / chain.std(param)**2]))
    profiles2 = Profiles()
    profiles2.set(profile=Samples(profile, params=params))
    cls = [1, 2]
    radii = np.sqrt([utils.nsigmas_to_deltachi2(cl, ddof=2) for cl in cls])
    t = np.linspace(0., 2. * np.pi, 1000, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    contours = ParameterContours()
    for i1, param1 in enumerate(params):
        for param2 in params[:i1]:
            mean = chain.mean([param1, param2])
            cov = chain.covariance(params=[param1, param2])
            sigx2, sigy2, sigxy = cov[0, 0], cov[1, 1], cov[0, 1]
            for cl, radius in zip(cls, radii):
                a = radius * np.sqrt(0.5 * (sigx2 + sigy2) + np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
                b = radius * np.sqrt(0.5 * (sigx2 + sigy2) - np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
                th = 0.5 * np.arctan2(2. * sigxy, sigx2 - sigy2)
                x1 = mean[0] + a * ct * np.cos(th) - b * st * np.sin(th)
                x2 = mean[1] + a * ct * np.sin(th) + b * st * np.cos(th)
                x1, x2 = (np.concatenate([xx, xx[:1]], axis=0) for xx in (x1, x2))
                contours.update({cl: [(ParameterArray(x1, param1), ParameterArray(x2, param2))]})
    profiles2.set(contour=contours)

    #g = plotting.plot_triangle([chain], contour_colors=['C0'], filled=True, show=False)
    #plotting.plot_triangle_contours([profiles1, profiles2], figsize=(5, 5), filled=False, colors=['C1', 'C2'], fig=g.subplots)
    #plotting.add_legend(labels=['chain', 'profiles1', 'profiles2'], colors=['C0', 'C1', 'C2'], loc='upper right')

    plotting.plot_triangle([chain, profiles1, profiles2], labels=['chain', 'profiles1', 'profiles2'],
                           contour_colors=['C0', 'C1', 'C2'], filled=[True, False, False], markers={'like.a': 0., 'like.d': 0.}, show=True)
    plotting.plot_triangle_contours([profiles1, profiles2], labels=['profiles1', 'profiles2'],
                                    colors=['C1', 'C2'], filled=[True, False], truths={'like.a': 0., 'like.d': 0.}, show=True)


def test_solved():
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    chain = get_chain(params, size=1000, nwalkers=5)[-1]
    chain['like.a'].param.update(derived='.auto')
    chain['like.b'].param.update(derived='.auto')
    array = np.zeros(chain.shape + (4,), dtype='f8')
    array[..., 1] = -0.3
    array[..., 3] = -0.3
    chain.set(ParameterArray(array, param='LRG.loglikelihood', derivs=[(), ('like.a',) * 2, ('like.a', 'like.b'), ('like.b',) * 2]))
    chain['logprior'] = chain['LRG.loglikelihood']

    print(chain.median('like.a'), np.median(chain['like.a']))
    assert np.allclose(chain.median('like.a'), np.median(chain['like.a']), atol=1.)
    print(chain.to_stats(tablefmt='pretty'))
    print(chain.sample_solved().to_stats(tablefmt='pretty'))
    chain['like.c'].param.update(derived='.auto')
    print(chain.sample_solved().to_stats(tablefmt='pretty'))


def test_cholesky():
    ndim = 4
    cov = np.random.uniform(size=(ndim, ndim))
    cov += 2. * np.eye(ndim)
    L = np.linalg.cholesky(cov)
    noise = np.random.standard_normal(size=(ndim, 1000))
    values = np.sum(noise[None, ...] * L[:, :, None], axis=1)
    print(cov)
    print(np.cov(values))


def test_pickle():
    chain_dir = '_chains'
    mean, cov, chain = get_chain(['like.a', 'like.b', 'like.c', 'like.d'], nwalkers=10)
    fn = os.path.join(chain_dir, 'chain.npy')
    chain.save(fn)

    import sys
    sys.modules['desilike'] = None
    np.load(fn, allow_pickle=True)


if __name__ == '__main__':

    setup_logging()

    test_misc()
    test_plot()
    test_bcast()
    test_stats()
    test_solved()
    # test_cholesky()
    test_pickle()
