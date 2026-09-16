"""Importance sampling kernel."""

import logging

import numpy as np
from scipy.special import logsumexp

from ..samples import diagnostics
from .base import StaticKernel


class Importance(StaticKernel):
    """Reweight an existing sample under a new posterior via importance sampling.

    Pass an :class:`~desilike.samples.MCSamples` object as the ``samples``
    keyword argument to :meth:`~desilike.samplers.base.StaticSampler.run`:

    .. code-block:: python

        sampler = Sampler(new_posterior, kernel=Importance())
        new_samples = sampler.run(samples=old_samples)

    Importance sampling degrades quickly when the proposal (the distribution the
    input samples were drawn from) does not cover the new posterior: the weight
    variance grows as ``exp(sigma^2)`` with ``sigma`` the scatter of
    ``log p_new - log p_old``. The returned samples therefore carry
    ``attrs['ess']``, ``attrs['khat']`` and ``attrs['logevidence']``, and the
    kernel warns when the diagnostics say the correction cannot be trusted. When
    it does, bridge with :class:`~desilike.samplers.smc.SMC` instead: it is the
    same correction split over a sequence of tempered intermediate distributions.
    """

    logger = logging.getLogger('Importance')

    # Above this Pareto k-hat the weight variance is effectively infinite.
    khat_threshold = 0.7
    # Below this ESS fraction the correction is carried by too few samples.
    ess_threshold = 0.1

    def get_samples(self, varied_params, samples=None, **kwargs):
        """Extract parameter columns from the input samples.

        Parameters
        ----------
        varied_params : VariableCollection
        samples : MCSamples
            Input samples from the old posterior.

        Returns
        -------
        numpy.ndarray, shape ``(n_samples, ndim)``
        """
        if samples is None:
            raise ValueError('Importance requires the input samples, as run(samples=...).')
        missing = [param.name for param in varied_params if param.name not in samples]
        if missing:
            raise ValueError(f'Input samples are missing varied parameters {missing}.')
        columns = [np.asarray(samples[key].value) for key in varied_params]
        return np.column_stack([column.reshape(len(column), -1) for column in columns])

    def post_process(self, results, samples=None, combine=False, resample=False, **kwargs):
        """Reweight ``results`` relative to the original ``samples``.

        Parameters
        ----------
        results : MCSamples or None
            Newly evaluated samples on the main rank; ``None`` on workers.
        samples : MCSamples
            Original samples, carrying the old log-posterior and, if they are
            weighted, the weights under which they represent the old posterior.
        combine : bool, optional
            If ``False`` (default), the new posterior *replaces* the old one:
            weights are ``old_weight * exp(new_log_post - old_log_post)``.
            If ``True``, the new likelihood is *added* to the old posterior:
            weights are ``old_weight * exp(new_log_post - new_log_prior)``.
        resample : bool, optional
            If ``True``, systematically resample to an equal-weight set of the
            same size instead of returning weighted samples. Default is ``False``.

        Returns
        -------
        MCSamples or None
        """
        if results is None:
            return None
        if samples is None:
            raise ValueError('Importance requires the input samples, as run(samples=...).')
        if combine:
            # The old posterior stays in the target, so it must not appear in the ratio:
            # the input weights alone carry it, and the new likelihood multiplies on top.
            log_w = results.logposterior - results.logprior
        else:
            log_w = results.logposterior - samples.logposterior
        # The input samples represent the old distribution *with* their own weights; dropping
        # them silently mis-weights any nested / population / previously-reweighted input.
        log_weight = np.log(np.asarray(samples.weight, dtype='f8'))
        log_ratio, log_w = log_w, log_w + log_weight

        nsamples = log_w.size
        # Two effective sample sizes, and they answer different questions: 'ess' is what the
        # returned weighted samples are worth, while 'ess_correction' isolates the damage done
        # by *this* reweighting (the input may already have been degenerate).  k-hat is only
        # meaningful on the correction ratio: that is the factor whose tail can be heavy when
        # the proposal fails to cover the new posterior.
        ess = diagnostics.kish_ess(log_w)
        ess_correction = diagnostics.kish_ess(log_ratio)
        khat = diagnostics.pareto_khat(log_ratio)
        # log(Z_new / Z_old); the input weight normalization cancels.
        log_evidence = float(logsumexp(log_w) - logsumexp(log_weight))

        self.logger.info('Importance weights: ESS = %.1f / %d (%.3f), correction ESS = %.1f (%.3f), '
                         'k-hat = %.2f, log(Z_new / Z_old) = %.3f', ess, nsamples, ess / nsamples,
                         ess_correction, ess_correction / nsamples, khat, log_evidence)
        if not (khat <= self.khat_threshold):
            self.logger.warning('Pareto k-hat = %.2f > %.2f: the importance-weight variance is '
                                'effectively infinite and the ESS above is optimistic. The old '
                                'distribution does not cover the new posterior; bridge with SMC instead.',
                                khat, self.khat_threshold)
        if ess_correction / nsamples < self.ess_threshold:
            self.logger.warning('Correction ESS fraction %.3f < %.2f: the reweighting rests on ~%d '
                                'samples. Bridge with SMC instead.',
                                ess_correction / nsamples, self.ess_threshold, int(ess_correction))

        results.aweight = np.exp(log_w - logsumexp(log_w))
        results.attrs.update(ess=ess, ess_correction=ess_correction, khat=khat,
                             logevidence=log_evidence)
        if resample:
            index = diagnostics.systematic_resample(results.aweight, nsamples, self.rng)
            results = results[index]
            results.aweight = np.ones(nsamples)
        return results
