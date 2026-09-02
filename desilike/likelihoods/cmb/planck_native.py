"""Native (table-based) Planck low-ell EE likelihood, sroll2 map-making.

Port of cobaya's ``planck_2018_lowl.EE_sroll2`` (https://github.com/CobayaSampler/cobaya), a
per-multipole tabulated probability: for each ell in [2, 29] the table holds ``log P(Dl_EE)``
on a regular grid of ``Dl_EE``, and the likelihood is the sum of the table entries at the
theory's own band powers.

Two reasons this exists rather than :class:`~desilike.likelihoods.cmb.candl.PlanckPR3LowlEESroll2Likelihood`,
which wraps the corresponding ``.clik`` through clipy:

* the SP4A cobaya block declares ``planck_2018_lowl.EE_sroll2``, i.e. THIS implementation, so
  the table is the like-for-like choice for reproducing it;
* the sroll2 ``.clik`` is not obtainable -- its only published source,
  ``web.fe.infn.it/~pagano/low_ell_datasets/sroll2/``, is a dead link (404).

The non-sroll2 low-ell EE needs no such port: its clik (``simall_100x143_offlike5_EE_Aplanck_B.clik``)
loads through clipy, so use ``PlanckPR3LowlEELikelihood`` for that one.

Written with ``jax.numpy`` so the arm traces like the rest of the CMB stack. Note the table
lookup is piecewise constant, hence exactly zero gradient -- fine for a gradient-free sampler,
wrong for anything that differentiates through it.
"""

import os

import numpy as np
import jax.numpy as jnp

from desilike.base import Likelihood
from desilike.parameter import Parameter, VariableCollection


class _BasePlanckLowlEENativeLikelihood(Likelihood):
    r"""Shared setup for the tabulated Planck low-ell EE likelihoods.

    Parameters
    ----------
    data_dir : str, default=None
        Directory holding :attr:`_table_file_name`. Defaults to the path recorded by
        :class:`~desilike.install.Installer`.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``CosmoprimoCosmology(engine='camb')``.
    calib : bool, default=True
        Whether to declare the ``A_planck`` calibration parameter. The calibration error is
        very small compared to the EE uncertainty; with ``calib=False`` it is taken to be 1.
    params : list of Parameter, default=None
        Extra parameters to add alongside the class defaults.
    """
    installer_section = None
    _table_file_name = None
    T0_cmb = 2.7255
    _lmin = 2
    _lmax = 29
    _nsteps = 3000
    _step = 0.0001

    def __init__(self, data_dir=None, cosmo=None, calib=True, params=None):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer().data_dir(self.installer_section)
        table = np.loadtxt(os.path.join(data_dir, self._table_file_name))
        nell = self._lmax - self._lmin + 1
        if table.shape != (self._nsteps, nell):
            raise ValueError(f'{self._table_file_name} has shape {table.shape}, expected '
                             f'{(self._nsteps, nell)}')
        self.table = jnp.asarray(table)

        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(engine='camb')
        self.cosmo = cosmo

        vc = self.propose_params() if calib else VariableCollection([])
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

    @classmethod
    def propose_params(cls):
        # No prior: A_planck is shared with the high-ell arm that declares one (CamSpec /
        # plik-lite). Declaring a second prior here would double-count the calibration.
        return VariableCollection([
            Parameter('A_planck', value=1., prior=None, latex=r'A_{\mathrm{planck}}'),
        ])

    def __post_init__(self, *args, **kwargs):
        self.cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': self._lmax}]})

    @property
    def ndata(self):
        return self._lmax - self._lmin + 1

    def __call__(self):
        harmonic = self.cosmo.get_harmonic()
        cl_ee = harmonic.lensed_cl(ellmax=self._lmax)['ee']
        ells = jnp.arange(self._lmax + 1)
        unit = (self.T0_cmb * 1e6) ** 2
        dl_ee = (ells * (ells + 1) / 2. / np.pi * unit * cl_ee)[self._lmin:self._lmax + 1]

        calib = self.params['A_planck'].value if 'A_planck' in self.params else 1.
        index = (dl_ee / (calib ** 2 * self._step)).astype(int)
        # Out of the tabulated range the likelihood is undefined; reject rather than let the
        # lookup wrap around (numpy's take_along_axis silently accepts negative indices).
        valid = jnp.all((index >= 0) & (index < self._nsteps)) & jnp.all(jnp.isfinite(dl_ee))
        index = jnp.clip(index, 0, self._nsteps - 1)
        self.logpdf = jnp.take_along_axis(self.table, index[None, :], axis=0).sum()
        self.logpdf = jnp.where(valid, self.logpdf, -jnp.inf)
        return self.logpdf

    @classmethod
    def install(cls, installer):
        from desilike.install import exists_path, download, extract
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)
        if installer.reinstall or not exists_path(os.path.join(data_dir, cls._table_file_name)):
            os.makedirs(data_dir, exist_ok=True)
            zip_fn = os.path.join(data_dir, cls._zip_basename)
            download(cls._zip_url, zip_fn)
            extract(zip_fn, data_dir)
            os.remove(zip_fn)
        installer.write({cls.installer_section: {'data_dir': data_dir}})


class PlanckPR3LowlEESroll2NativeLikelihood(_BasePlanckLowlEENativeLikelihood):
    r"""
    Planck low-ell EE likelihood with the sroll2 map-making, tabulated.

    This is the arm the SP4A configuration uses (cobaya's ``planck_2018_lowl.EE_sroll2``).

    Reference
    ---------
    Pagano et al. 2020, https://arxiv.org/abs/1908.09856
    """
    installer_section = 'PlanckPR3LowlEESroll2NativeLikelihood'
    _table_file_name = 'sroll2_prob_table.txt'
    _zip_basename = 'planck_sroll2_lowE.zip'
    _zip_url = ('https://github.com/CobayaSampler/planck_native_data/releases/download/v1/'
                'planck_sroll2_lowE.zip')
