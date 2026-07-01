"""BAO likelihoods."""

import os

import numpy as np

from desilike.likelihoods.base import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering.compressed import BAOCompressionObservable


_TRACER_FILES = {
    'BGS':       ('desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_mean.txt',
                  'desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_cov.txt'),
    'LRG1':      ('desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt',
                  'desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt'),
    'LRG2':      ('desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt',
                  'desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt'),
    'LRG3+ELG1': ('desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_mean.txt',
                  'desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_cov.txt'),
    'ELG2':      ('desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt',
                  'desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt'),
    'QSO':       ('desi_gaussian_bao_QSO_GCcomb_mean.txt',
                  'desi_gaussian_bao_QSO_GCcomb_cov.txt'),
    'Lya':       ('desi_gaussian_bao_Lya_GCcomb_mean.txt',
                  'desi_gaussian_bao_Lya_GCcomb_cov.txt'),
}

# BAOTheory attribute names vs quantity labels in data files.
_QUANTITY_TO_PARAM = {'DM_over_rs': 'DM_over_rd', 'DH_over_rs': 'DH_over_rd', 'DV_over_rs': 'DV_over_rd'}

_GITHUB_BASE = 'https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/'


def _read_mean_file(fn):
    """Return (z_eff, flat_values, param_names) from a BAO mean-values file."""
    z_list, value_list, param_list = [], [], []
    with open(fn) as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            z_list.append(float(parts[0]))
            value_list.append(float(parts[1]))
            quantity = parts[2]
            if quantity not in _QUANTITY_TO_PARAM:
                raise ValueError(f'Unknown quantity {quantity!r} in {fn}')
            param_list.append(_QUANTITY_TO_PARAM[quantity])
    return float(z_list[0]), np.array(value_list, dtype='f8'), param_list


def _read_cov_file(fn, size):
    """Return a (size × size) covariance matrix from a BAO covariance file."""
    return np.loadtxt(fn).reshape(size, size)


class DESIDR2BAOLikelihood(ObservablesGaussianLikelihood):
    r"""DESI DR2 BAO Gaussian likelihood.

    Computes :math:`\log P = -\tfrac{1}{2}\,\mathbf{r}^T C^{-1} \mathbf{r}` where
    :math:`\mathbf{r}` is the residual between measured and predicted
    :math:`D_M/r_d`, :math:`D_H/r_d`, or :math:`D_V/r_d` across the selected
    tracer bins.

    One :class:`~desilike.observables.galaxy_clustering.compressed.BAOCompressionObservable`
    is created per selected tracer bin and joined via
    :class:`~desilike.likelihoods.base.ObservablesGaussianLikelihood` with the
    block-diagonal joint covariance.

    Measurement and covariance files are read from the directory managed by
    :class:`~desilike.install.Installer`.  Install with
    ``Installer()(DESIDR2BAOLikelihood)``.

    Parameters
    ----------
    zbins : list of str, default=None
        Tracer bins to include.  Recognised names (in redshift order):
        ``'BGS'``, ``'LRG1'``, ``'LRG2'``, ``'LRG3+ELG1'``, ``'ELG2'``,
        ``'QSO'``, ``'Lya'``.  ``None`` includes all bins.
    data_dir : str, default=None
        Directory containing the BAO data files.  Defaults to the path saved
        by :class:`~desilike.install.Installer`.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator shared across all tracer bins.
        Defaults to ``CosmoprimoCosmology(fiducial='DESI')``.

    References
    ----------
    DESI 2025  https://arxiv.org/abs/2503.14738
    Data       https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2
    """

    installer_section = 'DESIDR2BAOLikelihood'
    _all_zbins = list(_TRACER_FILES)

    def __init__(self, zbins=None, data_dir=None, cosmo=None):
        if zbins is None:
            zbins = list(self._all_zbins)
        unknown_zbins = [zbin for zbin in zbins if zbin not in _TRACER_FILES]
        if unknown_zbins:
            raise ValueError(f'Unknown zbins {unknown_zbins}. Available: {list(_TRACER_FILES)}')

        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer().data_dir(self.installer_section)

        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(fiducial='DESI')

        # Build one BAOCompressionObservable per tracer bin (no per-obs covariance;
        # ObservablesGaussianLikelihood distributes the joint covariance to each).
        observables = []
        covariance_blocks = []

        for zbin in zbins:
            mean_fn, cov_fn = _TRACER_FILES[zbin]
            z_eff, meas_values, param_names = _read_mean_file(os.path.join(data_dir, mean_fn))
            cov_block = _read_cov_file(os.path.join(data_dir, cov_fn), len(param_names))
            obs = BAOCompressionObservable(
                data=meas_values, parameters=param_names, name=zbin,
                z=z_eff, cosmo=cosmo,
            )
            observables.append(obs)
            covariance_blocks.append(cov_block)

        # Assemble block-diagonal joint covariance (tracers are uncorrelated).
        total_size = sum(block.shape[0] for block in covariance_blocks)
        joint_covariance = np.zeros((total_size, total_size), dtype='f8')
        row_idx = 0
        for block in covariance_blocks:
            block_size = block.shape[0]
            joint_covariance[row_idx:row_idx + block_size, row_idx:row_idx + block_size] = block
            row_idx += block_size

        super().__init__(observables, covariance=joint_covariance)

    def __post_init__(self, observables, covariance=None, scale_covariance=1.,
                      correct_covariance=None, precision=None):
        # _init is set by ObservablesGaussianLikelihood.__init__ (called from our __init__),
        # so __post_init__ receives ObservablesGaussianLikelihood's signature at compile time.
        self.precision = self._precision if self._precision is not None else self.covariance.inv(level=1)

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download

        all_filenames = {fname for mean_fn, cov_fn in _TRACER_FILES.values() for fname in (mean_fn, cov_fn)}
        for filename in sorted(all_filenames):
            target = os.path.join(data_dir, filename)
            if installer.reinstall or not exists_path(target):
                download(_GITHUB_BASE + filename, target)

        installer.write({cls.installer_section: {'data_dir': data_dir}})
