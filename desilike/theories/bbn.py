"""Big Bang Nucleosynthesis (BBN) light-element abundances, computed with LINX.

LINX (Light Isotope Nucleosynthesis with JAX; https://github.com/cgiovanetti/LINX,
arXiv:2408.14538) solves the BBN background and nuclear network in JAX, so it is
jit-, vmap- and grad-compatible and runs as a native (non-external) desilike calculator.
"""

import numpy as np
import jax.numpy as jnp

from desilike.base import Calculator
from desilike.parameter import Parameter, VariableCollection


#: Atomic masses [kg] used by CAMB's ``bbn.ypBBN_to_yhe`` (camb 1.6.6).
_M_HE4, _M_H = 6.646479073e-27, 1.673575e-27

#: LINX initial Delta N_eff at which the final N_eff is solved, to invert the relation by spline.
#: 5 nodes reach LINX's own solver noise (~1e-5 in N_eff); more do not improve on it.
DELTA_NEFF_INIT_NODES = (-7.5, -3.75, 0., 3.75, 7.5)


def ypBBN_to_yhe(y_p):
    """Helium nucleon fraction :math:`4 n_{^4\\mathrm{He}} / n_\\mathrm{b}` to mass fraction, as CAMB's ``bbn.ypBBN_to_yhe``."""
    return -y_p * _M_HE4 / (-y_p * _M_HE4 + 4. * y_p * _M_H - 4. * _M_H)


def delta_neff_init_from_final(neff, neff_nodes):
    """LINX initial Delta N_eff giving final ``neff``: natural cubic spline through ``(neff_nodes, DELTA_NEFF_INIT_NODES)``, extrapolated past its ends."""
    import interpax
    return interpax.interp1d(neff, jnp.asarray(neff_nodes), jnp.asarray(DELTA_NEFF_INIT_NODES), method='cubic2', extrap=True)


class LINXAbundances(Calculator):
    r"""Primordial light-element abundances from LINX.

    Takes :math:`\omega_\mathrm{b}` and :math:`N_\mathrm{eff}` from ``cosmo``, and owns the
    neutron lifetime ``tau_n`` and the nuclear-rate nuisances ``q_<reaction>`` (one per
    reaction in the network; :math:`q \sim \mathcal{N}(0, 1)` selects the rate in its
    log-normal uncertainty band).  All are varied by default, with Gaussian priors: the rate
    nuisances carry the theory error on D/H, which is comparable to the observational one, so fixing
    them overstates the precision on :math:`\omega_\mathrm{b}`.  Fix them (``param.update(fixed=True)``)
    for a quick run.

    Outputs (set by ``__call__``, unscaled):

    * ``Y_p``     -- helium-4 nucleon fraction, :math:`4\, n_{^4\mathrm{He}} / n_\mathrm{b}`
      (the convention of observational :math:`Y_\mathrm{P}` determinations);
    * ``D_H``     -- :math:`n_\mathrm{D} / n_\mathrm{H}`;
    * ``He3_H``   -- :math:`(n_{^3\mathrm{He}} + n_\mathrm{T}) / n_\mathrm{H}`;
    * ``Li7_H``   -- :math:`(n_{^7\mathrm{Li}} + n_{^7\mathrm{Be}}) / n_\mathrm{H}`;
    * ``N_eff_bbn`` -- :math:`N_\mathrm{eff}` at the end of the LINX background integration.
    * ``YHe``     -- helium *mass* fraction, from ``Y_p`` with CAMB's ``bbn.ypBBN_to_yhe`` conversion:
      the ``YHe`` input of Boltzmann codes.

    :math:`N_\mathrm{eff}`: LINX injects extra radiation as an initial :math:`\Delta N_\mathrm{eff}`
    at :math:`T = 8.6` MeV, before :math:`e^+e^-` annihilation.  With ``dNeff_convention='final'``,
    ``cosmo['N_eff']`` is the final :math:`N_\mathrm{eff}`; the initial value is obtained by inverting
    the installed LINX's background, solved at ``DELTA_NEFF_INIT_NODES`` at the first build (a few seconds)
    and splined, so ``N_eff_bbn`` equals ``cosmo['N_eff']`` to LINX's solver noise (:math:`\sim 10^{-5}`).
    With ``'init'``, the calculator's own ``Delta_N_eff_init`` parameter goes to LINX directly
    (the parameterization of the LINX scripts and arXiv:2609.12065) and ``cosmo['N_eff']`` is not used.

    Parameters
    ----------
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator providing ``omega_b`` and ``N_eff``.
        If ``None``, defaults to ``CosmoprimoCosmology(fiducial='DESI')``.
    nuclear_net : str, default='key_PRIMAT_2023'
        LINX nuclear network: 'key_PRIMAT_2023', 'key_recommended', 'key_PRIMAT_2018',
        'key_YOF', 'key_PArthENoPE' or 'full_PRIMAT_2023'.
    params : Parameter, VariableCollection, dict, default=None
        Overrides of the default ``tau_n`` / ``q_<reaction>`` (and ``Delta_N_eff_init``) parameters.
    dNeff_convention : str, default='final'
        'final': ``cosmo['N_eff']`` is the final :math:`N_\mathrm{eff}`.
        'init': ``Delta_N_eff_init`` (fixed at 0 by default) is LINX's initial :math:`\Delta N_\mathrm{eff}`.
    linx_kwargs : dict, default=None
        Accuracy settings forwarded to ``linx.abundances.AbundanceModel.__call__``
        (e.g. ``rtol``, ``sampling_nTOp``, ``max_steps``).
    throw : bool, default=True
        Whether LINX raises on ODE solver failure.
    """
    @classmethod
    def install(cls, installer, no_deps=True):
        """Install LINX from PyPI (``linx-bbn``). 0.1.3 is the first release with LINX's extra-radiation fix (f4c96c6).

        ``linx-bbn`` pins ``diffrax==0.7.1`` and ``optimistix==0.0.11``, which would downgrade the newer versions other JAX
        packages in the environment use (cosmodesiconda: diffrax 0.7.2, optimistix 0.1.0, with which LINX's results are
        unchanged). So by default (``no_deps=True``) LINX is installed without its dependencies, and diffrax and optimistix
        are installed unpinned, only where missing. ``no_deps=False`` is a plain install, pins included.
        """
        # pkgname: LINX is imported as 'linx', which is how an existing install is recognised and skipped
        installer.pip('linx-bbn>=0.1.3', pkgname='linx', no_deps=bool(no_deps))
        if no_deps:
            for name in ('diffrax', 'optimistix'):
                installer.pip(name, no_deps=False)

    @staticmethod
    def _reaction_names(nuclear_net):
        from linx.nuclear import NuclearRates
        return [reaction.name for reaction in NuclearRates(nuclear_net=nuclear_net).reactions]

    @classmethod
    def propose_params(cls, nuclear_net='key_PRIMAT_2023', dNeff_convention='final'):
        """Return the default :class:`~desilike.parameter.VariableCollection`: ``tau_n``, ``q_<reaction>``
        and, for ``dNeff_convention='init'``, ``Delta_N_eff_init``."""
        from linx import const
        params = VariableCollection()
        if dNeff_convention == 'init':
            # limits: the Delta_Neff_init prior of arXiv:2609.12065 (Table 1)
            params.set(Parameter('Delta_N_eff_init', value=0., fixed=True, prior=dict(limits=[-7., 7.]),
                                 latex=r'\Delta N_\mathrm{eff}^\mathrm{init}'))
        params.set(Parameter('tau_n', value=const.tau_n, fixed=False,
                             prior=dict(dist='norm', loc=const.tau_n, scale=const.sigma_tau_n),
                             ref=dict(dist='norm', loc=const.tau_n, scale=const.sigma_tau_n),
                             fd=dict(eps=const.sigma_tau_n), latex=r'\tau_n'))
        for name in cls._reaction_names(nuclear_net):
            params.set(Parameter(f'q_{name}', value=0., fixed=False,
                                 prior=dict(dist='norm', loc=0., scale=1.),
                                 ref=dict(dist='norm', loc=0., scale=1.),
                                 fd=dict(eps=0.5), latex=rf'q_{{\mathrm{{{name}}}}}'))
        return params

    def __init__(self, cosmo=None, nuclear_net='key_PRIMAT_2023', params=None, dNeff_convention='final',
                 linx_kwargs=None, throw=True):
        if dNeff_convention not in ('final', 'init'):
            raise ValueError(f"dNeff_convention must be 'final' or 'init', not {dNeff_convention!r}")
        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(fiducial='DESI')
        self.cosmo = cosmo
        proposed = self.propose_params(nuclear_net=nuclear_net, dNeff_convention=dNeff_convention)
        if params is not None:
            params = VariableCollection(params)
            if dNeff_convention == 'final' and 'Delta_N_eff_init' in params:
                raise ValueError("Delta_N_eff_init is only used with dNeff_convention='init'; with 'final', N_eff comes from cosmo")
            proposed = proposed + params
        self.tau_n = proposed['tau_n']
        # in network order: LINX reads `nuclear_rates_q` positionally
        self.q = [proposed[f'q_{name}'] for name in self._reaction_names(nuclear_net)]
        if dNeff_convention == 'init':
            self.delta_neff_init = proposed['Delta_N_eff_init']

    def __post_init__(self, cosmo=None, nuclear_net='key_PRIMAT_2023', params=None, dNeff_convention='final',
                      linx_kwargs=None, throw=True):
        from linx import const
        from linx.background import BackgroundModel
        from linx.abundances import AbundanceModel
        from linx.nuclear import NuclearRates
        self._dNeff_convention = dNeff_convention
        self._background = BackgroundModel(throw=throw)
        self.cosmo.add_requirements({'params.omega_b': None})
        if dNeff_convention == 'final':
            self.cosmo.add_requirements({'params.N_eff': None})
            # final N_eff of the installed LINX at DELTA_NEFF_INIT_NODES; a build re-runs __post_init__
            # on the same instance, so solve only the first time
            if getattr(self, '_neff_nodes', None) is None:
                self._neff_nodes = np.array([float(self._background(jnp.asarray(delta_neff_init))[-1][-1])
                                             for delta_neff_init in DELTA_NEFF_INIT_NODES])
        self._abundance = AbundanceModel(NuclearRates(nuclear_net=nuclear_net), throw=throw)
        self._linx_kwargs = dict(linx_kwargs or {})
        self._omega_b_fid, self._tau_n_fid = const.Omegabh2, const.tau_n

    def __call__(self):
        if self._dNeff_convention == 'final':
            delta_neff_init = delta_neff_init_from_final(self.cosmo['N_eff'], self._neff_nodes)
        else:
            delta_neff_init = self.delta_neff_init.value
        time, scale_factor, rho_photon, rho_neutrino, rho_extra, pressure_extra, neff = self._background(jnp.asarray(delta_neff_init))
        # LINX's own omega_b <-> eta conversion, as in its scripts: eta = eta_fac * eta0
        eta_fac = self.cosmo['omega_b'] / self._omega_b_fid
        tau_n_fac = self.tau_n.value / self._tau_n_fid
        nuclear_rates_q = jnp.array([param.value for param in self.q])
        yields = self._abundance(rho_photon, rho_neutrino, rho_extra, pressure_extra, t_vec=time, a_vec=scale_factor,
                                 eta_fac=jnp.asarray(eta_fac), tau_n_fac=jnp.asarray(tau_n_fac),
                                 nuclear_rates_q=nuclear_rates_q, **self._linx_kwargs)
        # species order: n, p, d, t, He3, a, Li7, Be7
        yield_hydrogen = yields[1]
        self.Y_p = 4. * yields[5]
        self.D_H = yields[2] / yield_hydrogen
        self.He3_H = (yields[3] + yields[4]) / yield_hydrogen
        self.Li7_H = (yields[6] + yields[7]) / yield_hydrogen
        self.N_eff_bbn = neff[-1]
        self.YHe = ypBBN_to_yhe(self.Y_p)
        return self

    def tree_flatten(self):
        return [self.Y_p, self.D_H, self.He3_H, self.Li7_H, self.N_eff_bbn, self.YHe], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.Y_p, obj.D_H, obj.He3_H, obj.Li7_H, obj.N_eff_bbn, obj.YHe = children
        return obj


from desilike.theories.primordial_cosmology import CosmoprimoCosmology


class BBNConsistentCosmology(CosmoprimoCosmology):
    r"""A :class:`CosmoprimoCosmology` whose helium fraction and :math:`N_\mathrm{eff}` are those predicted by LINX.

    At every call the Boltzmann code receives ``YHe = bbn.YHe`` (the LINX helium mass fraction) and
    ``N_eff = bbn.N_eff_bbn`` (the :math:`N_\mathrm{eff}` LINX's background integration ends with) as
    inputs, instead of its own BBN interpolation table and a sampled :math:`N_\mathrm{eff}` -- the
    coupling of ``linx_cobaya`` (github.com/Kushallodha/linx_cobaya) and arXiv:2609.12065.

    ``bbn`` reads :math:`\omega_\mathrm{b}` (and, for ``'final'``, :math:`N_\mathrm{eff}`) from its own cosmology, which
    must therefore not be this one (the graph would be cyclic); build both with
    :func:`linx_bbn_cosmology`, which also makes the two :math:`\omega_\mathrm{b}` one parameter.
    ``params`` must not include ``N_eff`` or ``YHe``.

    Parameters
    ----------
    bbn : LINXAbundances
        The BBN calculation supplying ``YHe`` and ``N_eff``.
    *args, **kwargs
        As :class:`CosmoprimoCosmology`.
    """
    def __init__(self, *args, bbn=None, **kwargs):
        super().__init__(*args, **kwargs)
        if bbn is None:
            raise ValueError('BBNConsistentCosmology needs a LINXAbundances calculator: bbn=...')
        overridden = [name for name in ('N_eff', 'YHe') if name in self.params]
        if overridden:
            raise ValueError(f'{overridden} are set from bbn and must not be parameters of BBNConsistentCosmology')
        self.bbn = bbn

    def _input_overrides(self):
        return {'YHe': self.bbn.YHe, 'N_eff': self.bbn.N_eff_bbn}


def linx_bbn_cosmology(params, engine='camb', fiducial=None, N_eff=None, nuclear_net='key_PRIMAT_2023', **kwargs):
    r"""Build a Boltzmann cosmology whose ``YHe`` and :math:`N_\mathrm{eff}` come from LINX, and its LINX node.

    Graph: Parameters (:math:`\omega_\mathrm{b}`, :math:`N_\mathrm{eff}`) -> a JAX-native carrier
    cosmology -> :class:`LINXAbundances` -> :class:`BBNConsistentCosmology` (``engine``), with
    :math:`\omega_\mathrm{b}` shared between the carrier and the main cosmology.

    Parameters
    ----------
    params : VariableCollection
        Parameters of the main cosmology. Must include ``omega_b``; must not include ``N_eff`` / ``YHe``.
    engine : str, default='camb'
        Cosmoprimo engine of the main cosmology.
    fiducial : cosmoprimo.Cosmology, default=None
        Fiducial of the main cosmology; ``cosmoprimo.fiducial.DESI(engine=engine)`` if ``None``.
    N_eff : Parameter, default=None
        The sampled final :math:`N_\mathrm{eff}` (``dNeff_convention='final'``). ``None``: LINX runs with
        ``dNeff_convention='init'``; ``Delta_N_eff_init`` is fixed at 0 (LINX's standard-model background);
        free it with ``bbn.delta_neff_init.update(fixed=False)``.
    nuclear_net : str, default='key_PRIMAT_2023'
        LINX network.
    **kwargs
        Forwarded to :class:`LINXAbundances` (``linx_kwargs``, ``throw``, ...); not ``params``, which is the main cosmology's.

    Returns
    -------
    cosmo : BBNConsistentCosmology
    bbn : LINXAbundances
    """
    import cosmoprimo
    from desilike.base import share_params
    if fiducial is None:
        fiducial = cosmoprimo.fiducial.DESI(engine=engine)
    params = VariableCollection(params)
    carrier_params = [params['omega_b'].clone()] + ([] if N_eff is None else [N_eff])
    carrier = CosmoprimoCosmology(engine='eisenstein_hu', fiducial=cosmoprimo.fiducial.DESI(engine='eisenstein_hu'),
                                  params=VariableCollection(carrier_params))
    bbn = LINXAbundances(cosmo=carrier, nuclear_net=nuclear_net,
                         dNeff_convention='init' if N_eff is None else 'final', **kwargs)
    cosmo = BBNConsistentCosmology(engine=engine, fiducial=fiducial, params=params, bbn=bbn)
    share_params([cosmo], names=['omega_b'])
    return cosmo, bbn
