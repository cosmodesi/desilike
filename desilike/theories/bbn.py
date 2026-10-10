"""Big Bang Nucleosynthesis (BBN) light-element abundances, computed with LINX.

LINX (Light Isotope Nucleosynthesis with JAX; https://github.com/cgiovanetti/LINX,
arXiv:2408.14538) solves the BBN background and nuclear network in JAX, so it is
jit-, vmap- and grad-compatible and runs as a native (non-external) desilike calculator.
"""

import jax.numpy as jnp

from desilike.base import Calculator
from desilike.parameter import Parameter, VariableCollection


#: Atomic masses [kg] used by CAMB's ``bbn.ypBBN_to_yhe`` (camb 1.6.6).
_M_HE4, _M_H = 6.646479073e-27, 1.673575e-27


def ypBBN_to_yhe(y_p):
    """Helium nucleon fraction :math:`4 n_{^4\\mathrm{He}} / n_\\mathrm{b}` to mass fraction, as CAMB's ``bbn.ypBBN_to_yhe``."""
    return -y_p * _M_HE4 / (-y_p * _M_HE4 + 4. * y_p * _M_H - 4. * _M_H)


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

    :math:`N_\mathrm{eff}` mapping: extra radiation is injected as LINX's initial
    :math:`\Delta N_\mathrm{eff}` (at :math:`T \sim 30` MeV), chosen so that the late-time value
    is ``cosmo['N_eff']`` relative to the standard-model value ``N_eff_sm``.  LINX's response is
    linear to :math:`10^{-4}` over :math:`|\Delta N_\mathrm{eff}| \lesssim 1`, with slope
    ``_neff_slope``; ``N_eff_bbn`` reports what LINX actually obtained.

    Parameters
    ----------
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator providing ``omega_b`` and ``N_eff``.
        If ``None``, defaults to ``CosmoprimoCosmology(fiducial='DESI')``.
    nuclear_net : str, default='key_PRIMAT_2023'
        LINX nuclear network: 'key_PRIMAT_2023', 'key_recommended', 'key_PRIMAT_2018',
        'key_YOF', 'key_PArthENoPE' or 'full_PRIMAT_2023'.
    params : Parameter, VariableCollection, dict, default=None
        Overrides of the default ``tau_n`` / ``q_<reaction>`` parameters.
    N_eff_sm : float, default=3.044
        Standard-model :math:`N_\mathrm{eff}`, the value at which no extra radiation is injected.
    linx_kwargs : dict, default=None
        Accuracy settings forwarded to ``linx.abundances.AbundanceModel.__call__``
        (e.g. ``rtol``, ``sampling_nTOp``, ``max_steps``).
    throw : bool, default=True
        Whether LINX raises on ODE solver failure.
    """
    #: Late-time :math:`\Delta N_\mathrm{eff}` per unit LINX initial :math:`\Delta N_\mathrm{eff}`,
    #: measured from ``BackgroundModel()`` at :math:`\Delta N_\mathrm{eff,init} = 0.5, 1, -0.5`
    #: (0.27184, 0.27194, 0.27166); close to, but not exactly, :math:`(4/11)^{4/3}`.
    _neff_slope = 0.2719

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cgiovanetti/LINX')

    @staticmethod
    def _reaction_names(nuclear_net):
        from linx.nuclear import NuclearRates
        return [reaction.name for reaction in NuclearRates(nuclear_net=nuclear_net).reactions]

    @classmethod
    def propose_params(cls, nuclear_net='key_PRIMAT_2023'):
        """Return the default :class:`~desilike.parameter.VariableCollection`: ``tau_n`` and ``q_<reaction>``."""
        import linx.const as const
        params = VariableCollection()
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

    def __init__(self, cosmo=None, nuclear_net='key_PRIMAT_2023', params=None, N_eff_sm=3.044,
                 linx_kwargs=None, throw=True):
        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(fiducial='DESI')
        self.cosmo = cosmo
        proposed = self.propose_params(nuclear_net=nuclear_net)
        if params is not None:
            proposed = proposed + VariableCollection(params)
        self.tau_n = proposed['tau_n']
        # in network order: LINX reads `nuclear_rates_q` positionally
        self.q = [proposed[f'q_{name}'] for name in self._reaction_names(nuclear_net)]

    def __post_init__(self, cosmo=None, nuclear_net='key_PRIMAT_2023', params=None, N_eff_sm=3.044,
                      linx_kwargs=None, throw=True):
        import linx.const as const
        from linx.background import BackgroundModel
        from linx.abundances import AbundanceModel
        from linx.nuclear import NuclearRates
        self.cosmo.add_requirements({'params.omega_b': None, 'params.N_eff': None})
        self._background = BackgroundModel(throw=throw)
        self._abundance = AbundanceModel(NuclearRates(nuclear_net=nuclear_net), throw=throw)
        self._linx_kwargs = dict(linx_kwargs or {})
        self._N_eff_sm = float(N_eff_sm)
        self._omega_b_fid, self._tau_n_fid = const.Omegabh2, const.tau_n

    def __call__(self):
        delta_neff_init = (self.cosmo['N_eff'] - self._N_eff_sm) / self._neff_slope
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

    ``bbn`` reads :math:`\omega_\mathrm{b}` and :math:`N_\mathrm{eff}` from its own cosmology, which
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
        The sampled :math:`N_\mathrm{eff}` (LINX convention, see :class:`LINXAbundances`). ``None``:
        fixed at 3.044, i.e. no extra radiation (LINX's own standard-model background).
    nuclear_net : str, default='key_PRIMAT_2023'
        LINX network.
    **kwargs
        Forwarded to :class:`LINXAbundances` (``params``, ``linx_kwargs``, ``throw``, ...).

    Returns
    -------
    cosmo : BBNConsistentCosmology
    bbn : LINXAbundances
    """
    import cosmoprimo
    from desilike.base import share_params
    if fiducial is None:
        fiducial = cosmoprimo.fiducial.DESI(engine=engine)
    if N_eff is None:
        N_eff = Parameter('N_eff', value=3.044, fixed=True, latex=r'N_{\mathrm{eff}}')
    params = VariableCollection(params)
    carrier = CosmoprimoCosmology(engine='eisenstein_hu', fiducial=cosmoprimo.fiducial.DESI(engine='eisenstein_hu'),
                                  params=VariableCollection([params['omega_b'].clone(), N_eff]))
    bbn = LINXAbundances(cosmo=carrier, nuclear_net=nuclear_net, **kwargs)
    cosmo = BBNConsistentCosmology(engine=engine, fiducial=fiducial, params=params, bbn=bbn)
    share_params([cosmo], names=['omega_b'])
    return cosmo, bbn
