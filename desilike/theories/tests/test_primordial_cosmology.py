import numpy as np

from desilike import setup_logging
from desilike.theories import Cosmoprimo


def test_omegak():
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    ax = plt.gca()
    fiducial = DESI()
    size = 5
    values = np.linspace(-0.1, 0.1, size)
    cmap = plt.get_cmap('jet', len(values))

    for i, Omega_k in enumerate(values):
        cosmo = fiducial.clone(Omega_k=Omega_k)
        pk = cosmo.get_fourier().pk_interpolator(of='delta_cb').to_1d(z=1.)
        print(pk.k.min(), pk.k.max(), pk.pk.shape)
        ax.loglog(pk.k, pk.pk, color=cmap(i), label=r'$\Omega_k = {:.2f}$'.format(Omega_k))
    ax.legend()
    plt.savefig('pklin_omegak.png')
    plt.show()


def test_parameterization():

    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    cosmo()
    cosmo = Cosmoprimo(fiducial='DESI', engine='camb')
    del cosmo.params['h']
    cosmo.params['theta_MC_100'].update(prior={'limits': [0.99, 1.01]})
    cosmo()
    bak = {'h': cosmo.h, 'omega_cdm': cosmo['omega_cdm'], 'omega_b': cosmo['omega_b']}
    cosmo(theta_MC_100=1.)
    assert not np.allclose(cosmo.h, bak['h'])
    assert np.allclose(cosmo.Omega0_cdm * cosmo.h**2, bak['omega_cdm'])
    assert np.allclose(cosmo.Omega0_b * cosmo.h**2, bak['omega_b'])

    cosmo.runtime_info.pipeline._set_speed()

    cosmo = Cosmoprimo(fiducial='DESI', engine='isitgr')
    cosmo.init.params['Q0'] = dict()
    cosmo(Q0=0.1)

    cosmo = Cosmoprimo(fiducial='DESI', engine='mgcamb', MG_flag=1)
    cosmo.init.params['sigma0'] = dict()
    cosmo(sigma0=0.1)

    cosmo = Cosmoprimo()
    cosmo()
    del cosmo.params['N_eff']
    cosmo(logA=3.)
    assert np.allclose(cosmo['N_eff'], 3.044)
    #print(cosmo.init.params['omega_b'].latex(), cosmo.init.params['tau_reio'].latex())

    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo()
    N_eff = cosmo['N_eff']
    cosmo(m_ncdm=0.)
    assert np.allclose(cosmo['N_eff'], N_eff)

    cosmo(Omega_k=0.1)
    assert np.allclose(cosmo['Omega_k'], 0.1)

    cosmo.init.params['sigma8_m'] = {'derived': True}
    _, derived = cosmo(return_derived=True)
    print(derived)
    print(derived['sigma8_m'])


def test_cosmoprimo():
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
    cosmo = Cosmoprimo(engine='class')
    template = DirectPowerSpectrumTemplate(z=0.5, cosmo=cosmo, with_now='wallish2018')
    template()
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    theory()


if __name__ == '__main__':

    setup_logging()
    test_omegak()
    test_parameterization()
    test_cosmoprimo()