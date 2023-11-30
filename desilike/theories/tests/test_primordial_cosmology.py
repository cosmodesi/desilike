import numpy as np

from desilike import setup_logging
from desilike.theories import Cosmoprimo


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


if __name__ == '__main__':

    setup_logging()
    test_parameterization()