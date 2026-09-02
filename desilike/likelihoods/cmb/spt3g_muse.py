"""SPT-3G 2-year delensed EE + optimal ϕϕ (MUSE) likelihood.

Wrapper around the public ``muse3glike`` release (Ge et al. 2024), the likelihood the
``CMB-SPA`` chains run as ``muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse``.

This is NOT the same thing as
:class:`~desilike.likelihoods.cmb.act_dr6_spt_lensing.ACTDR6SPTLensingLikelihood` with
``variant='spt3g'``, even though both describe the same measurement. That variant reads the
same MUSE band powers repackaged as ``muse_likelihood.npz`` (``d_pp`` agrees with the numbers
here to 7e-16, the band-power windows are identical), but it then evaluates a *plain* Gaussian
in binned ``C_L^kk``, applies a Hartlap factor derived from 796 ACT simulations, and carries a
covariance that differs from the released marginal one by up to a factor 4 in the last two
bins. Measured on identical theory, the two differ by a near-constant ``delta chi2`` of about
0.6-1.0. Use this class to reproduce a chain that declares ``muse3glike``; use the other one
when SPT lensing goes inside the joint ACT+Planck+SPT covariance.

Two conventions inherited from the upstream cobaya plugin, both easy to get wrong:

* the EE leg is the **unlensed** (delensed) EE spectrum in muK^2, while ``ϕϕ`` is the
  dimensionless ``C_L^{ϕϕ}`` -- so EE carries a ``(T0_cmb * 1e6)^2`` factor against
  cosmoprimo's dimensionless output and ``ϕϕ`` carries none;
* both legs are read over ``ell = 1 .. 5000``, the span of the band-power windows. The
  upstream plugin nominally requests ``pp`` only to 3000, but cobaya hands it the full array
  built for the ``ee`` request, so 5000 is what it actually uses.

The likelihood is Gaussian only in the transformed band-power space computed by the release:
each band power is divided by a per-bin scale ``s`` and passed through ``arctanh(x - 1.01)``,
with a volume term folded into the covariance to keep a uniform prior on band powers. The
arctanh domain is exactly the release's stated prior, ``0 < x < 2.01`` times the fiducial, and
outside it this class returns ``-inf`` rather than the upstream ``nan``.

The systematics (calibrations, polarisation angles, band-pass and leakage terms) are always
marginalised over inside the release's covariance, so this likelihood declares no nuisance
parameters.

Reference
---------
Ge et al. 2024 (SPT-3G Collaboration), https://arxiv.org/abs/2411.06000
Data and code: https://lambda.gsfc.nasa.gov/data/suborbital/SPT/muse_3g_like_march_2025.zip
"""

import os

import numpy as np
import jax.numpy as jnp

from desilike.base import GaussianLikelihood
from desilike.parameter import VariableCollection


#: Component names, exactly as the release spells them: they key ``BPWF`` and ``s``, and are
#: what its constructor accepts.
_COMPONENTS = ('ϕϕ', 'EE')

#: The transform the release applies to band powers before the gaussian: arctanh(x - offset).
_TRANSFORM_OFFSET = 1.01


class SPT3G2yrMUSELikelihood(GaussianLikelihood):
    r"""
    SPT-3G 2-year delensed EE + optimal :math:`\phi\phi` likelihood (MUSE).

    Parameters
    ----------
    components : list, str, default=('ϕϕ', 'EE')
        Which legs to use, named as the release names them: ``'ϕϕ'`` selects the lensing
        potential band powers, ``'EE'`` the delensed EE band powers. The chains reproduced by
        ``CMB-SPA`` use ``'ϕϕ'`` alone. Components that are not selected are still marginalised
        over by the release's covariance; selecting fewer legs changes which data enter, not
        whether the systematics are marginalised.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``CosmoprimoCosmology(engine='camb')``.
    filename : str, default=None
        Path to the band-power ``.h5``. Defaults to the file shipped inside ``muse3glike``.
    params : list of Parameter, default=None
        Extra parameters. The likelihood itself declares none.
    """
    installer_section = 'SPT3G2yrMUSELikelihood'
    T0_cmb = 2.7255
    #: Span of the band-power window functions.
    ellmin, ellmax = 1, 5000
    _zip_url = 'https://lambda.gsfc.nasa.gov/data/suborbital/SPT/muse_3g_like_march_2025.zip'

    def __init__(self, components=_COMPONENTS, cosmo=None, filename=None, params=None):
        import muse3glike

        if isinstance(components, str):
            components = [components]
        components = [str(component) for component in components]
        unknown = [component for component in components if component not in _COMPONENTS]
        if unknown:
            raise ValueError(f'unknown component(s) {unknown}; expected any of {list(_COMPONENTS)}')
        if len(set(components)) != len(components):
            raise ValueError(f'repeated component in {components}')
        self.components = components

        like = muse3glike.spt3g_2yr_delensed_ee_optimal_pp_muse(filename=filename, components=components)

        # Take the constants from the release, but hold them in float64: everything below is
        # built with jax.numpy inside muse3glike, and would be float32 if x64 were ever off.
        self._bpwf = {name: jnp.asarray(np.asarray(like.BPWF[name], dtype='f8')) for name in components}
        self._scale = {name: jnp.asarray(np.asarray(like.s[name], dtype='f8')) for name in components}
        self._drop_last = {name: name == 'EE' for name in components}

        ells = np.asarray(like.BPWF['ℓ'])
        if ells[0] != self.ellmin or ells[-1] != self.ellmax:
            raise ValueError(f'band-power windows span ell {ells[0]}..{ells[-1]}, expected '
                             f'{self.ellmin}..{self.ellmax}')

        self.flatdata = jnp.asarray(np.asarray(like.d_transformed_vec, dtype='f8'))
        covariance = np.asarray(like.Σ_transformed, dtype='f8')
        self.precision = jnp.asarray(np.linalg.inv(covariance))

        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(engine='camb')
        self.cosmo = cosmo

        vc = VariableCollection([])
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

    def __post_init__(self, *args, **kwargs):
        requirements = [{'ellmax': self.ellmax}]
        if 'ϕϕ' in self.components:
            self.cosmo.add_requirements({'harmonic.lens_potential_cl': requirements})
        if 'EE' in self.components:
            # Delensed EE: the unlensed spectrum, not the lensed one.
            self.cosmo.add_requirements({'harmonic.unlensed_cl': requirements})

    def __call__(self):
        # C_ell for each selected leg over ell = 1 .. 5000, in the release's units.
        harmonic = self.cosmo.get_harmonic()
        sl = slice(self.ellmin, self.ellmax + 1)
        cl = {}
        if 'ϕϕ' in self.components:
            # C_L^{ϕϕ} is dimensionless on both sides -- no temperature factor here.
            cl['ϕϕ'] = harmonic.lens_potential_cl(ellmax=self.ellmax)['pp'][sl]
        if 'EE' in self.components:
            unit = (self.T0_cmb * 1e6) ** 2
            cl['EE'] = unit * harmonic.unlensed_cl(ellmax=self.ellmax)['ee'][sl]

        flattheory, valid = [], True
        for name in self.components:
            binned = (self._bpwf[name] @ cl[name]) / self._scale[name]
            if self._drop_last[name]:
                binned = binned[:-1]
            # arctanh is defined on (-1, 1), i.e. band powers within (0, 2.01) times fiducial.
            # That IS the release's stated uniform prior on the input spectra; outside it the
            # upstream returns nan, and we reject instead.
            argument = binned - _TRANSFORM_OFFSET
            valid = valid & jnp.all(jnp.abs(argument) < 1.) & jnp.all(jnp.isfinite(binned))
            flattheory.append(jnp.arctanh(jnp.clip(argument, -1. + 1e-12, 1. - 1e-12)))

        self.flattheory = jnp.concatenate(flattheory)
        logpdf = super().__call__()
        self.logpdf = jnp.where(valid, logpdf, -jnp.inf)
        return self.logpdf

    @classmethod
    def install(cls, installer):
        """Install ``muse3glike`` from the LAMBDA archive.

        The zip holds the project one directory down, so pip cannot take the URL directly:
        download, extract, then install the extracted directory. ``no_deps`` because the
        release pins ``numpy < 2`` and ``jax < 0.5``, neither of which it actually needs.
        """
        from desilike.install import download, extract

        try:
            import muse3glike  # noqa: F401
            if not installer.reinstall:
                return
        except ImportError:
            pass

        data_dir = installer.data_dir(cls.installer_section)
        os.makedirs(data_dir, exist_ok=True)
        zip_fn = os.path.join(data_dir, os.path.basename(cls._zip_url))
        download(cls._zip_url, zip_fn)
        extract(zip_fn, data_dir)
        os.remove(zip_fn)
        source_dir = os.path.join(data_dir, 'muse_3g_like_march_2025')
        installer.pip(source_dir, pkgname='muse3glike', no_deps=True,
                      force_reinstall=installer.reinstall)
