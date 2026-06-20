"""Planck NPIPE (PR4) CamSpec high-ell CMB likelihood."""

import os

import numpy as np
import jax.numpy as jnp

from desilike.base import GaussianLikelihood
from desilike.parameter import Parameter, VariableCollection


class _BasePlanckNPIPECamspecLikelihood(GaussianLikelihood):
    r"""Shared setup for the Planck NPIPE (PR4) CamSpec high-ell likelihood (see
    :class:`TTTEEEHighlPlanckNPIPECamspecLikelihood` and :class:`TTHighlPlanckNPIPECamspecLikelihood`):
    data loading, power-law foreground/calibration model, and ``install()``.

    JAX adaptation of
    https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/base_classes/planck_2018_CamSpec_python.py

    Note
    ----
    The fast Chebyshev-projection foreground marginalization (``proj_order`` in the
    original implementation) is not ported; this likelihood always computes the full chi2.

    ``jax.grad`` through the external ``camb``/``class`` engine's finite-difference
    (``pure_callback``) path is currently unreliable for ``harmonic.lensed_cl``: it can
    raise a ``CosmologyInputError`` deep inside ``cosmoprimo`` when a finite-difference
    step perturbs a parameter (root cause not yet diagnosed). The forward pass (and jit
    thereof) is unaffected.

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory (containing the extracted ``CamSpec_NPIPE`` archive). Defaults to the
        path saved by :class:`~desilike.install.Installer` once the likelihood has been installed.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator. If ``None``, defaults to ``CosmoprimoCosmology(engine='camb', fiducial='DESI')``
        (a Boltzmann engine is required to compute CMB :math:`C_\ell`).
    params : Parameter, VariableCollection, dict, default=None
        Override the default nuisance parameters (calibration and foreground amplitudes/tilts).
    """
    installer_section = 'PlanckNPIPECamspecLikelihood'
    all_cls = ['100x100', '143x143', '217x217', '143x217', 'TE', 'EE']
    select_cls = None  # set by concrete subclasses
    # Fixed CMB temperature used for the dimensionless-Cl -> muK^2 unit conversion
    # (matches the reference implementation; not tied to the cosmology's T_cmb parameter).
    T0_cmb = 2.7255

    @classmethod
    def propose_params(cls):
        params = [
            Parameter('A_planck', value=1., prior=dict(dist='norm', loc=1., scale=0.0025),
                      ref=dict(dist='norm', loc=1., scale=0.002), latex=r'y_\mathrm{cal}'),
            Parameter('cal0', value=1., fixed=True, latex='c_{100}'),
            Parameter('cal2', value=1., fixed=True, latex='c_{217}'),
            Parameter('amp_100', value=0., fixed=True, latex=r'A^{\mathrm{power}}_{100}'),
            Parameter('amp_143', value=10., prior=dict(limits=[0., 50.]),
                      ref=dict(dist='norm', loc=10., scale=1.), latex=r'A^{\mathrm{power}}_{143}'),
            Parameter('amp_217', value=20., prior=dict(limits=[0., 50.]),
                      ref=dict(dist='norm', loc=20., scale=1.), latex=r'A^{\mathrm{power}}_{217}'),
            Parameter('amp_143x217', value=10., prior=dict(limits=[0., 50.]),
                      ref=dict(dist='norm', loc=10., scale=1.), latex=r'A^{\mathrm{power}}_{143\times217}'),
            Parameter('n_100', value=1., fixed=True, latex=r'\gamma^{\mathrm{power}}_{100}'),
            Parameter('n_143', value=1., prior=dict(limits=[0., 5.]),
                      ref=dict(dist='norm', loc=1., scale=0.2), latex=r'\gamma^{\mathrm{power}}_{143}'),
            Parameter('n_217', value=1., prior=dict(limits=[0., 5.]),
                      ref=dict(dist='norm', loc=1., scale=0.2), latex=r'\gamma^{\mathrm{power}}_{217}'),
            Parameter('n_143x217', value=1., prior=dict(limits=[0., 5.]),
                      ref=dict(dist='norm', loc=1., scale=0.2), latex=r'\gamma^{\mathrm{power}}_{143\times217}'),
        ]
        if cls.select_cls is None or 'TE' in cls.select_cls:
            params.append(Parameter('calTE', value=1., prior=dict(dist='norm', loc=1., scale=0.01),
                                     ref=dict(dist='norm', loc=1., scale=0.01), latex='c_{TE}'))
        if cls.select_cls is None or 'EE' in cls.select_cls:
            params.append(Parameter('calEE', value=1., prior=dict(dist='norm', loc=1., scale=0.01),
                                     ref=dict(dist='norm', loc=1., scale=0.01), latex='c_{EE}'))
        return VariableCollection(params)

    def __init__(self, data_dir=None, cosmo=None, params=None):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer().data_dir(self.installer_section), 'CamSpec_NPIPE')
        self._load_data(data_dir)
        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=self.ellmax, non_linear='mead')))
        self.cosmo = cosmo
        self.cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': self.ellmax}]})
        vc = self.propose_params()
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

    def _load_data(self, data_dir):
        input_data = np.loadtxt(os.path.join(data_dir, 'like_NPIPE_12.6_unified_spectra.txt'))
        flatdata, masks, index_ells, all_cls = [], [], {}, []
        with open(os.path.join(data_dir, 'like_NPIPE_12.6_unified_data_ranges.txt'), 'r', encoding='utf-8-sig') as file:
            for iline, line in enumerate(file):
                if not line.strip():
                    continue
                items = line.split()
                cl = items[0]
                all_cls.append(cl)
                elllim = [int(x) for x in items[1:]]
                nells = elllim[1] - elllim[0] + 1
                flatdata.append(input_data[elllim[0]: elllim[1] + 1, iline])
                tmp_ells = np.arange(elllim[0], elllim[1] + 1)
                mask = np.zeros(nells, dtype='?')
                if elllim[1] and nells and cl in self.select_cls:
                    mask[...] = True
                masks.append(mask)
                if mask.any():
                    index_ells[cl] = tmp_ells[mask]
        if all_cls != self.all_cls:
            raise ValueError('Unexpected spectra order in data_ranges file: {}'.format(all_cls))
        mask = np.concatenate(masks)
        nx = len(mask)
        with open(os.path.join(data_dir, 'like_NPIPE_12.6_unified_cov.bin'), 'rb') as file:
            covariance = np.fromfile(file, dtype=np.float32)
        if nx ** 2 != covariance.shape[0]:
            raise ValueError('Covariance size {} does not match expected {}**2'.format(covariance.shape[0], nx))
        self.flatdata = jnp.asarray(np.concatenate(flatdata)[mask])
        covariance = covariance.reshape(nx, nx)[np.ix_(mask, mask)].astype('f8')
        # Inverting the full (~11000x11000) matrix takes ~1 min; cache the result per select_cls.
        precision_fn = os.path.join(data_dir, 'precision_{}.npy'.format('_'.join(self.select_cls)))
        try:
            precision, cached_covariance = np.load(precision_fn, allow_pickle=True)
            if not np.allclose(covariance, cached_covariance):
                raise ValueError
        except Exception:
            precision = np.linalg.inv(covariance)
            np.save(precision_fn, np.array([precision, covariance], dtype=object))
        self.precision = jnp.asarray(precision)
        self.index_ells = index_ells
        self.ellmax = max(max(ell) for ell in self.index_ells.values())
        self.has_foregrounds = any(cl in self.all_cls[:4] for cl in self.index_ells)
        pivot = 1500
        ells = jnp.arange(self.ellmax + 1)
        self._template_foreground_tilt = jnp.log(jnp.maximum(ells, 1) / pivot)
        self._template_foreground_amp = jnp.where(ells >= 1, jnp.ones_like(self._template_foreground_tilt), 0.)
        self._factor = ells * (ells + 1) / 2 / np.pi

    def _get_foregrounds(self):
        names = ['100', '143', '217', '143x217']
        amp = jnp.array([self.params['amp_{}'.format(name)].value for name in names])
        tilt = jnp.array([self.params['n_{}'.format(name)].value for name in names])
        return amp[:, None] * self._template_foreground_amp * jnp.exp(self._template_foreground_tilt * tilt[:, None])

    def _get_cals(self):
        calPlanck = self.params['A_planck'].value ** 2
        calTE = self.params['calTE'].value if 'calTE' in self.params else 1.
        calEE = self.params['calEE'].value if 'calEE' in self.params else 1.
        cal0, cal2 = self.params['cal0'].value, self.params['cal2'].value
        return jnp.array([cal0, 1., cal2, jnp.sqrt(cal2), calTE, calEE]) * calPlanck

    def __call__(self):
        cl = self.cosmo.get_harmonic().lensed_cl(ellmax=self.ellmax)
        unit = (self.T0_cmb * 1e6) ** 2
        cl_tt, cl_te, cl_ee = (self._factor * unit * cl[name] for name in ['tt', 'te', 'ee'])

        cals = self._get_cals()
        if self.has_foregrounds:
            foregrounds = self._get_foregrounds()

        flattheory = []
        for icl, name in enumerate(self.all_cls):
            if name in self.index_ells:
                index = self.index_ells[name]
                if icl <= 3:
                    tmp = cl_tt[index] + foregrounds[icl][index]
                elif icl == 4:
                    tmp = cl_te[index]
                else:
                    tmp = cl_ee[index]
                flattheory.append(tmp / cals[icl])
        self.flattheory = jnp.concatenate(flattheory)
        return super().__call__()

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.reinstall or not exists_path(os.path.join(data_dir, 'CamSpec_NPIPE')):
            zip_base = 'CamSpec_NPIPE.zip'
            url = 'https://github.com/CobayaSampler/planck_native_data/releases/download/v1/{}'.format(zip_base)
            zip_fn = os.path.join(data_dir, zip_base)
            download(url, zip_fn)
            extract(zip_fn, data_dir)

        installer.write({cls.installer_section: {'data_dir': data_dir}})


class TTTEEEHighlPlanckNPIPECamspecLikelihood(_BasePlanckNPIPECamspecLikelihood):
    """
    TT+TE+EE Planck NPIPE (PR4) CamSpec high-ell likelihood.

    Reference
    ---------
    https://arxiv.org/abs/2205.10869
    """
    select_cls = ['143x143', '217x217', '143x217', 'TE', 'EE']


class TTHighlPlanckNPIPECamspecLikelihood(_BasePlanckNPIPECamspecLikelihood):
    """
    TT-only Planck NPIPE (PR4) CamSpec high-ell likelihood.

    Reference
    ---------
    https://arxiv.org/abs/2205.10869
    """
    select_cls = ['143x143', '217x217', '143x217']
