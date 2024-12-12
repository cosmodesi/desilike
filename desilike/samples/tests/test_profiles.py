import os

import numpy as np
from desilike import setup_logging
from desilike.samples import Profiles, Samples, ParameterBestFit, ParameterCovariance, ParameterContours, ParameterGrid, plotting


def get_profiles(params):
    rng = np.random.RandomState()
    profiles = Profiles()
    profiles.set(start=Samples([0. for param in params], params=params))
    params = profiles.start.params()
    for param in params: param.update(fixed=False)
    profiles.set(bestfit=ParameterBestFit([rng.normal(0., 0.1, size=1) for param in params] + [-0.5], params=params + ['logposterior'], loglikelihood='LRG.loglikelihood'))
    profiles.set(error=Samples([[0.5] for param in params], params=params))
    profiles.set(covariance=ParameterCovariance(np.eye(len(params)), params=params))
    profiles.set(interval=Samples([(-0.5, 0.5) for param in params], params=params))
    x = np.linspace(-1., 1., 101)
    profiles.set(profile=Samples([[x, 1. + x**2] for param in params], params=params))
    t = np.linspace(0., 2. * np.pi, 21)
    params2 = [(param1, param2) for i1, param1 in enumerate(params) for param2 in params[:i1 + 1]]
    profiles.set(contour=ParameterContours({1: [(np.cos(t), np.sin(t)) for param in params2]}, params=params2))
    grid = np.meshgrid(*(np.linspace(0., 0.1, 3),) * (len(params) + 1), indexing='ij')
    profiles.set(grid=ParameterGrid(grid, params=params + ['logposterior']))
    return profiles


def test_misc():
    profiles_dir = '_profiles'
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = [get_profiles(params) for i in range(5)]
    profiles = Profiles.concatenate(*profiles)
    assert profiles.bestfit._loglikelihood == 'LRG.loglikelihood'
    assert profiles.bestfit.shape == profiles.bestfit['logposterior'].shape == (5,)
    assert profiles.contour[1]['params.b', 'params.a'][::-1] == profiles.contour[1]['params.a', 'params.b']
    profiles.set(contour=profiles.contour.interpolate(size=42))
    assert profiles.contour[1]['params.b', 'params.a'][0].size == 42
    fn = os.path.join(profiles_dir, 'profile.npy')
    profiles.save(fn)
    profiles2 = profiles.load(fn)
    assert profiles2 == profiles
    profiles.bcast(profiles)
    choice = profiles.choice()
    assert choice.bestfit.shape == (1,)
    print(choice.error.shape, choice.error)
    assert choice.error.shape == (1,)
    assert profiles.choice(index=[0, 1]).bestfit.shape == (2,)
    del profiles.error
    profiles.bcast(profiles)
    profiles.profile.choice()
    profiles.grid.choice()


def test_stats():
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = get_profiles(params)
    print(profiles.to_stats(tablefmt='latex_raw'))
    print(profiles.to_stats(tablefmt='pretty'))
    print(profiles.to_stats(tablefmt='list_latex'))
    assert isinstance(profiles.to_stats(tablefmt='list')[0], list)


def test_plot():
    profiles_dir = '_profiles'
    params = ['like.a', 'like.b', 'like.c', 'like.d']
    profiles = [get_profiles(params)] * 2
    plotting.plot_aligned_stacked(profiles, fn=os.path.join(profiles_dir, 'aligned.png'))

    profiles = [get_profiles(params)] * 2
    plotting.plot_profile(profiles, fn=os.path.join(profiles_dir, 'profile.png'))
    plotting.plot_profile_comparison(profiles[0], profiles[1], fn=os.path.join(profiles_dir, 'profile_comparison.png'))


def test_mpi():
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = get_profiles(params)
    profiles.bestfit.attrs.update(ndof=10, chi2=10.)
    profiles2 = get_profiles(params)
    profiles2.update(profiles)

    from desilike import mpi
    mpicomm = mpi.COMM_WORLD
    profiles = Profiles.bcast(profiles, mpiroot=0)
    if mpicomm.rank == 0:
        print(profiles.bestfit.attrs)


def test_pickle():
    profiles_dir = '_profiles'
    profiles = get_profiles(['like.a', 'like.b', 'like.c', 'like.d'])
    fn = os.path.join(profiles_dir, 'profile.npy')
    profiles.save(fn)

    import sys
    sys.modules['desilike'] = None
    np.load(fn, allow_pickle=True)



if __name__ == '__main__':

    setup_logging()

    test_misc()
    test_stats()
    test_plot()
    test_mpi()
    test_pickle()
