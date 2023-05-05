import os

import numpy as np
from desilike import setup_logging
from desilike.samples import Profiles, Samples, ParameterBestFit, ParameterCovariance, ParameterContours, plotting


def get_profiles(params):
    rng = np.random.RandomState()
    profiles = Profiles()
    profiles.set(start=Samples([0. for param in params], params=params))
    params = profiles.start.params()
    for param in params: param.update(fixed=False)
    profiles.set(bestfit=ParameterBestFit([rng.normal(0., 0.1) for param in params] + [-0.5], params=params + ['logposterior']))
    profiles.set(error=Samples([0.5 for param in params], params=params))
    profiles.set(covariance=ParameterCovariance(np.eye(len(params)), params=params))
    profiles.set(interval=Samples([(-0.5, 0.5) for param in params], params=params))
    x = np.linspace(-1., 1., 101)
    profiles.set(profile=Samples([[x, 1. + x**2] for param in params], params=params))
    t = np.linspace(0., 2. * np.pi, 101)
    params2 = [(param1, param2) for param1 in params for param2 in params]
    profiles.set(contour=ParameterContours([(np.cos(t), np.sin(t)) for param in params2], params=params2))
    return profiles


def test_misc():
    profiles_dir = '_profiles'
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = Profiles.concatenate(*[get_profiles(params) for i in range(5)])
    assert profiles.bestfit.shape == profiles.bestfit['logposterior'].shape == (5,)
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


def test_stats():
    params = ['params.a', 'params.b', 'params.c', 'params.d']
    profiles = get_profiles(params)
    print(profiles.to_stats(tablefmt='latex_raw'))
    print(profiles.to_stats(tablefmt='pretty'))


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


if __name__ == '__main__':

    setup_logging()

    test_misc()
    test_stats()
    test_plot()
    test_mpi()
