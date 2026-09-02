"""PocoMC preconditioned Monte Carlo kernel."""

import logging

import numpy as np

try:
    import pocomc as _pocomc
    POCOMC_INSTALLED = True
except ModuleNotFoundError:
    POCOMC_INSTALLED = False

from .base import PopulationKernel, update_kwargs


class _Prior:
    """Prior wrapper for ``pocoMC`` built from prior callables."""

    def __init__(self, prior_logpdf, prior_rvs, prior_bounds, ndim, rng):
        self._logpdf = prior_logpdf
        self._rvs = prior_rvs
        self._rng = rng
        self._bounds = prior_bounds   # (ndim, 2)
        self._ndim = ndim

    def logpdf(self, x):
        x = np.asarray(x)
        log_p = np.asarray([result for result in self._logpdf(x)])
        in_bounds = np.all((x >= self._bounds[:, 0]) & (x <= self._bounds[:, 1]), axis=1)
        log_p[~in_bounds] = -np.inf
        return log_p

    def rvs(self, size=1):
        # Drawing through the sampler rather than through a PPF of our own also lets pocoMC
        # start from proposals that have no inverse CDF, such as an existing chain.
        return np.asarray(self._rvs(size, self._rng))

    @property
    def bounds(self):
        return self._bounds

    @property
    def dim(self):
        return self._ndim


def _default_device():
    """Torch device matching the one JAX is using, or ``None`` to stay on the CPU.

    A JAX likelihood already holds the GPU, and the flow costs 0.12 GiB beside it (see
    :func:`_flow_on_device`), so there is nothing to gain from leaving the preconditioner on
    the CPU when both libraries can see the accelerator. ROCm builds of ``torch`` also expose
    their device as ``'cuda'``, hence the single name for both JAX GPU backends.
    """
    try:
        import jax
        backend = jax.default_backend()
    except Exception:
        return None
    if backend not in ('gpu', 'cuda', 'rocm'):
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return None


def _flow_on_device(device):
    """Return a subclass of ``pocomc.Flow`` whose normalizing flow lives on *device*.

    ``pocoMC`` has no device handling anywhere: :class:`pocomc.Flow` builds the ``zuko`` flow on
    the CPU, ``numpy_to_torch`` makes CPU tensors, and ``torch_to_numpy`` calls ``.numpy()``
    with no ``.cpu()`` -- so the preconditioner never reaches the GPU the likelihood is already
    using, and an already-moved flow would raise on the way back out.

    It is worth moving because the flow's INVERSE is what the MCMC loop calls every step, and
    ``zuko.transforms.AutoregressiveTransform._inverse`` runs ``passes`` *sequential* network
    evaluations -- ``n_dim`` of them for a fully autoregressive flow, so 47 x 6 transforms = 282
    per step at 47 parameters, against one for the forward direction. Measured on an A100, 512
    points, 47 dimensions: one ``flow.inverse`` costs 2717 ms on the CPU and 376 ms on the GPU
    (7.2x), or 164 ms and 32.5 ms with ``passes=2`` coupling transformations (5.0x).

    Every tensor crossing back out is returned on the CPU, so nothing downstream changes.

    Notes
    -----
    Sharing the device with a JAX likelihood needs no memory tuning: measured on an A100-40GB
    with JAX initialised first, the default preallocation takes 29.6 GiB and leaves 9.4, while
    the flow needs 0.12 GiB (15.7 MiB of parameters plus context) and runs at the same speed.
    It still fits at ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.98``, with 0.4 GiB left. Only reach for
    ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` on a small device or a much larger flow.
    A checkpoint written with a GPU-resident flow reloads onto the same device.
    """
    import torch
    from pocomc.flow import Flow as _Flow

    class DeviceFlow(_Flow):

        def __init__(self, n_dim, flow='nsf3'):
            super().__init__(n_dim, flow)
            self.device = torch.device(device)
            self.flow = self.flow.to(self.device)

        def to_device(self, tensor):
            return tensor.to(self.device) if torch.is_tensor(tensor) else tensor

        @staticmethod
        def to_host(result):
            # Every pocoMC consumer casts with `torch_to_numpy`, which is `.detach().numpy()`
            # with no `.cpu()`, so everything must come back on the host.
            if torch.is_tensor(result):
                return result.cpu()
            return tuple(tensor.cpu() for tensor in result)

        def forward(self, x):
            return self.to_host(super().forward(self.to_device(x)))

        def inverse(self, u):
            return self.to_host(super().inverse(self.to_device(u)))

        def log_prob(self, x):
            return self.to_host(super().log_prob(self.to_device(x)))

        def sample(self, size=1):
            return self.to_host(super().sample(size))

        def fit(self, x, weights=None, **kwargs):
            return super().fit(self.to_device(x), weights=self.to_device(weights), **kwargs)

    return DeviceFlow


_CLEAR_BEFORE_SAVE = ('log_likelihood', 'log_prior', 'sample_prior', 'prior', 'pool', 'distribute', 'save_state')


def _patch_save_state(pocomc_sampler):
    """Monkey-patch ``pocoMC``'s ``save_state`` to null unpicklable attributes before dumping."""
    _original_save_state = pocomc_sampler.save_state

    def _save_state_no_likelihood(path):
        saved = {attr: getattr(pocomc_sampler, attr, None) for attr in _CLEAR_BEFORE_SAVE}
        for attr in _CLEAR_BEFORE_SAVE:
            setattr(pocomc_sampler, attr, None)
        try:
            _original_save_state(path)
        finally:
            for attr, val in saved.items():
                setattr(pocomc_sampler, attr, val)

    pocomc_sampler.save_state = _save_state_no_likelihood
    return _save_state_no_likelihood


class PocoMC(PopulationKernel):
    """Preconditioned Monte Carlo sampler via ``pocomc``.

    .. rubric:: References
    - https://github.com/minaskar/pocomc
    - https://doi.org/10.21105/joss.04634
    - https://doi.org/10.1093/mnras/stac2272
    """

    logger = logging.getLogger('PocoMC')

    def __init__(self, device=None, **kwargs):
        """
        Parameters
        ----------
        device : str or None, optional
            Torch device for the normalizing flow, e.g. ``'cuda'``, or ``'cpu'`` to pin it to
            the host. ``None`` (default) follows JAX: the flow goes on the GPU if JAX is using
            one and ``torch`` can see it, and stays on the CPU otherwise -- ``pocoMC`` itself
            only ever builds it on the CPU. The flow's inverse is most of a high-dimensional
            step (82% at 47 parameters), so moving it is worth 8.7x end to end there, and it
            shares the device with a JAX likelihood without any memory tuning; see
            :func:`_flow_on_device` and :func:`_default_device`.
        **kwargs
            Extra keyword arguments forwarded to ``pocomc.Sampler``. ``flow`` accepts a
            ``zuko.flows.Flow`` object as well as a name, which is how to get coupling
            transformations: ``zuko.flows.NSF(..., passes=2)`` makes the inverse exact in 2
            sequential passes instead of ``n_dim``.
        """
        self._device = device
        self._kwargs = kwargs
        self._sampler = None

    def reset_state(self):
        self._sampler = None

    @classmethod
    def install(cls, installer):
        installer.pip('pocomc')

    def init(self, likelihood, prior, rng, **context):
        _, self._likelihood_logpdf_with_derived = likelihood
        self._prior_logpdf, _, self._prior_rvs, self._prior_bounds = prior
        self._rng = rng
        self._pool = context['pool']
        self._ndim = context['ndim']
        self._output_dir = context.get('output_dir')

    def run(self, **kwargs):
        if not POCOMC_INSTALLED:
            raise ImportError("The 'pocomc' package is required but not installed.")

        if self._pool.main:
            if self._sampler is None:
                prior_obj = _Prior(self._prior_logpdf, self._prior_rvs, self._prior_bounds, self._ndim, self._rng)
                init_kwargs = update_kwargs(
                    dict(**self._kwargs), 'pocoMC',
                    prior=prior_obj, likelihood=self._likelihood_logpdf_with_derived,
                    n_dim=self._ndim, pool=self._pool,
                    output_dir=self._output_dir,
                    random_state=self._rng.integers(2**32 - 1))
                device = self._device
                if device is None:
                    device = _default_device()
                    if device is not None:
                        self.logger.info('JAX is on GPU, placing the normalizing flow on {}.'.format(device))
                if device is None:
                    self._sampler = _pocomc.Sampler(**init_kwargs)
                else:
                    # pocoMC instantiates its Flow internally, so substitute the class for the
                    # duration of the construction rather than reaching into the built sampler.
                    original_flow = _pocomc.sampler.Flow
                    _pocomc.sampler.Flow = _flow_on_device(device)
                    try:
                        self._sampler = _pocomc.Sampler(**init_kwargs)
                    finally:
                        _pocomc.sampler.Flow = original_flow

                _patch_save_state(self._sampler)

                # Restore checkpoint if available.
                if self._output_dir is not None:
                    filepath_max = None
                    state_max = -1
                    for filepath in self._output_dir.glob('pmc_*.state'):
                        state = str(filepath.stem).split('_')[1]
                        if state == 'final':
                            filepath_max = filepath
                            break
                        state = int(state)
                        if state > state_max:
                            state_max = state
                            filepath_max = filepath
                    if filepath_max is not None:
                        saved = {attr: getattr(self._sampler, attr, None)
                                 for attr in _CLEAR_BEFORE_SAVE}
                        self._sampler.load_state(filepath_max)
                        for attr, val in saved.items():
                            setattr(self._sampler, attr, val)
                        _patch_save_state(self._sampler)

            run_kwargs = update_kwargs(
                kwargs, 'pocoMC',
                resume_state_path=None,
                save_every=1 if self._output_dir is not None else None)
            self._sampler.run(**run_kwargs)

            samples, weights, logl, logp, blobs = self._sampler.posterior(return_blobs=True)
            blobs = blobs.reshape(len(samples), -1)

            self._pool.stop_wait()
            self.logger.info('Finished sampling.')
            return samples, blobs, dict(aweight=weights, logposterior=logl + logp)
        self._pool.wait()
        return None
