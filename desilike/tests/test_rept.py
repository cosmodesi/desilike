"""velocileptors REPT in two stages (REPTVelocileptorsPTSpectrum2Poles): the JAX stage 2 against velocileptors' own multipole tables, and the
amplitude homogeneity of the stage-1 tables that REPTEmulator routes on. Needs velocileptors and cosmoprimo (class)."""
from types import SimpleNamespace

import numpy as np
import pytest


velocileptors = pytest.importorskip('velocileptors')


def _rept(amplitude=1.):
    """velocileptors' REPT and desilike's pt run on the same linear power (DESI fiducial, z = 0.5), the pt's stage 1 called on a bare
    template holding only what it reads."""
    from scipy.signal import savgol_filter
    from cosmoprimo.fiducial import DESI
    from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT
    from desilike.theories.galaxy_clustering.full_shape import REPTVelocileptorsPTSpectrum2Poles
    cosmo = DESI(engine='class')
    k = np.geomspace(1e-4, 2., 500)
    pk = amplitude * cosmo.get_fourier().pk_interpolator(of='delta_cb')(k, z=0.5)
    # any smooth no-wiggle power will do, the same for both (the pt passes the template's): log-space Savitzky-Golay, exact in the amplitude
    pknow = np.exp(savgol_filter(np.log(pk), 51, 3))
    options = dict(rbao=110, beyond_gauss=True, one_loop=True, shear=True, cutoff=20, jn=5, N=4000, extrap_min=-4, extrap_max=3,
                   import_wisdom=False)
    rept = REPT(k, pk, pnw=pknow, kmin=0.0005, kmax=0.5, nk=200, **options)
    pt = object.__new__(REPTVelocileptorsPTSpectrum2Poles)
    pt.k, pt.ells, pt.nmu, pt._options = np.array([0.0005, 0.5]), (0, 2, 4), 4, options
    pt.template = SimpleNamespace(k=k, pk_dd=pk, pknow_dd=pknow, fk=0.8 * np.ones_like(k), f=0.8, qpar=1., qper=1., sigma8=0.8, fsigma8=0.64,
                                  sigma8_fid=0.8)
    REPTVelocileptorsPTSpectrum2Poles.__call__(pt)
    return rept, pt


def test_rept_stage2_matches_velocileptors():
    from scipy.interpolate import interp1d
    rept, pt = _rept()
    k = np.linspace(0.02, 0.2, 19)
    f = 0.8 * (1. - 0.01 * rept.kv)  # a scale-dependent f, as desilike passes it
    pt.f, pt.table['fk_over_f'] = 0.8, f / 0.8
    for qpar, qper in [(1., 1.), (1.02, 0.97)]:
        reference = rept.compute_redshift_space_power_multipoles_tables(f, apar=qpar, aperp=qper, ngauss=4)[1:]
        pt.qpar, pt.qper = qpar, qper
        new = np.asarray(pt.bias_table(k))
        scale = np.abs(interp1d(rept.kv, reference[0], kind='cubic', axis=0)(k)).max(axis=0)  # each column's monopole size
        for ill in range(3):
            ref = interp1d(rept.kv, reference[ill], kind='cubic', axis=0)(k)
            assert np.all(np.abs(new[ill] - ref).max(axis=0) <= 1e-3 * scale), (qpar, qper, ill)


def test_rept_stage1_amplitude_degrees():
    from desilike.theories.galaxy_clustering.full_shape import REPTEmulator
    tables = [_rept(amplitude)[1].table for amplitude in (1., 1.21)]
    for name in tables[0]:
        degree = REPTEmulator._table_scaling(name)[0]
        first, second = np.atleast_2d(tables[0][name]), np.atleast_2d(tables[1][name])
        if name in ('kv', 'fk_over_f'):
            assert np.allclose(first, second); continue
        rows = range(1, first.shape[0]) if name.split('_')[0] in ('pk', 'vk', 's0', 's2') else range(first.shape[0])  # row 0: degree-1 remainder
        for row in rows:
            # relative to the row's amplitude (measured <= 1e-7, but the wiggle g1, g3 tables, differences of two FFTLog evaluations: <= 1e-3);
            # a wrong degree would be off by 21%
            assert np.abs(second[row] - 1.21**degree * first[row]).max() <= 2e-3 * np.abs(first[row]).max() * 1.21**degree, (name, row)
