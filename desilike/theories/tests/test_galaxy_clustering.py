import numpy as np

from desilike import setup_logging


def test_integ():
    from desilike.theories.galaxy_clustering.base import BaseTheoryPowerSpectrumMultipolesFromWedges
    from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles

    topoles = BaseTheoryPowerSpectrumMultipolesFromWedges(mu=8)
    mu, wmu = topoles.mu, topoles.wmu
    assert np.isclose(np.sum(wmu), 1.)
    template = StandardPowerSpectrumTemplate()
    pk_trapz = KaiserTracerPowerSpectrumMultipoles(template=template, mu=100, method='trapz')()
    pk_leggauss = KaiserTracerPowerSpectrumMultipoles(template=template, mu=20, method='leggauss')()
    print(pk_trapz)
    print(pk_leggauss)


def test_bao():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmapar=3.)
    theory = ResummedBAOWigglesTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmas=3.)
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmapar=3.)
    theory = ResummedBAOWigglesTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmas=3.)

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    theory(qpar=1.1, sigmapar=3.)

    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, StandardPowerSpectrumTemplate
    template = BAOPowerSpectrumTemplate(z=0.1, fiducial='DESI', apmode='qiso')
    theory.init.update(template=template)
    theory(df=0.8)

    template = StandardPowerSpectrumTemplate(z=0.1, fiducial='DESI', apmode='qiso', with_now='peakaverage')
    theory.init.update(template=template)
    theory()
    template.pk_dd


def test_full_shape():

    def test_emulator_likelihood(theory, test_likelihood=True):
        print('Emulating', theory)
        if test_likelihood:
            from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix
            from desilike.likelihoods import ObservablesGaussianLikelihood
            if 'Power' in theory.__class__.__name__:
                observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01], 4: [0.05, 0.2, 0.01]},
                                                                     data={}, theory=theory)
            else:
                observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [20, 150, 4], 2: [20, 150, 4], 4: [20, 150, 4]},
                                                                           data={}, theory=theory)
            observable()
            cov = np.eye(observable.flatdata.shape[0])
            likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
            for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*']):
                param.update(derived='.best')
            likelihood()
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        #theory()
        bak = theory()
        calculator = theory.pt
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=0))
        emulator.set_samples()
        emulator.fit()
        pt = emulator.to_calculator()
        theory.init.update(pt=pt)
        assert np.allclose(theory(), bak)
        if test_likelihood:
            likelihood()

    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    theory = KaiserTracerPowerSpectrumMultipoles()
    theory(logA=3.04, b1=1.).shape
    theory = KaiserTracerCorrelationFunctionMultipoles()
    theory(logA=3.04, b1=1.).shape

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    theory(dm=0.01, b1=1.).shape
    assert not np.allclose(theory(dm=-0.01), theory(dm=0.01))
    assert not np.allclose(theory(qpar=0.99), theory(qper=1.01))
    test_emulator_likelihood(theory)
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory = EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory = LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles

    theory = PyBirdTracerPowerSpectrumMultipoles(eft_basis='westcoast')
    test_emulator_likelihood(theory)
    theory(logA=3.04, b1=1.).shape
    theory = PyBirdTracerCorrelationFunctionMultipoles(eft_basis='westcoast')
    test_emulator_likelihood(theory)  # no P(k) computed
    theory(logA=3.04, b1=1.).shape


from scipy import special
from desilike import utils
from desilike.jax import numpy as jnp
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, APEffect
from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
from desilike.theories.galaxy_clustering.full_shape import BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles

class CorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    config_fn = None

    def initialize(self, *args, pt=None, template=None, mu=20, method='leggauss', **kwargs):
        #power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')]()
        power = KaiserTracerPowerSpectrumMultipoles()
        if pt is not None: power.init.update(pt=pt)
        if template is None:
            template = DirectPowerSpectrumTemplate()
        power.init.update(template=template)
        super(CorrelationFunctionMultipoles, self).initialize(*args, power=power, **kwargs)
        self.apeffect = power.template.apeffect
        power.template.runtime_info.requires = [require for require in power.template.runtime_info.requires if require is not self.apeffect]
        power.template.apeffect = APEffect(mode='qparqper')
        power.template.apeffect()
        self.mu, wmu = utils.weights_mu(mu, method=method)
        self.legendre = [special.legendre(ell) for ell in self.ells]
        self.wmu = np.array([wmu * (2 * ell + 1) * leg(self.mu) for ell, leg in zip(self.ells, self.legendre)])

    def calculate(self):
        power = []
        for pk in self.power.power:
            slope_high = (pk[-1] - pk[-2]) / np.log10(self.kin[-1] / self.kin[-2])
            power.append(jnp.concatenate([jnp.interp(np.log10(self.k_mid), np.log10(self.kin), pk), (pk[-1] + slope_high * self.k_high) * self.pad_high], axis=-1))
        s, corr = self.fftlog(np.vstack(power))
        sap, muap = self.apeffect.ap_s_mu(self.s, self.mu)[1:]
        self.corr = jnp.array([jnp.sum(jnp.interp(sap, ss, cc) * leg(muap) * wmu, axis=-1) for ss, cc, leg, wmu in zip(s, corr, self.legendre, self.wmu)])

    @property
    def pt(self):
        return self.power.pt

    @property
    def template(self):
        return self.power.template

    def get(self):
        return self.corr


def test_pk_to_xi():
    from matplotlib import pyplot as plt
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.1))
    from cosmoprimo import PowerToCorrelation

    k = np.logspace(-4.5, 3., 2048)
    theory.init.update(k=np.geomspace(k[0], 1., 300))
    fftlog = PowerToCorrelation(k, ell=theory.ells, q=0, lowring=False)

    def interp(k, kin, pk):
        mask = k > kin[-1]
        k_high = np.log10(k[mask] / kin[-1])
        pad_high = np.exp(-(k[mask] / kin[-1] - 1.)**2 / (2. * (10.)**2))
        slope_high = (pk[-1] - pk[-2]) / np.log10(kin[-1] / kin[-2])
        k_mid = k[~mask]
        return np.concatenate([np.interp(np.log10(k_mid), np.log10(kin), pk), (pk[-1] + slope_high * k_high) * pad_high], axis=-1)

    fig, lax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 6), squeeze=True)
    fig.subplots_adjust(hspace=0)
    ax = plt.gca()
    for qpar in [0.95, 1., 1.05]:
        pk = theory(qpar=qpar, dm=0.)
        pk = [interp(k, theory.k, pk) for pk in pk]
        s, xi = fftlog(pk)
        for ill, ell in enumerate(theory.ells):
            lax[0].plot(k, pk[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
            si = s[ill]
            mask = (si > 1.) & (si < 200.)
            lax[1].plot(si[mask], si[mask]**2 * xi[ill][mask], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
    lax[0].set_xscale('log')
    lax[1].set_xscale('linear')
    plt.show()

    theory = CorrelationFunctionMultipoles(s=np.linspace(1., 200., 1000), template=ShapeFitPowerSpectrumTemplate(z=1.1))
    ax = plt.gca()
    xi_ref = theory()
    for qpar in [0.999, 1., 1.001]:
        xi = theory(qpar=qpar)
        for ill, ell in enumerate(theory.ells):
            ax.plot(theory.s, theory.s**2 * (xi[ill] - xi_ref[ill]), color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
    plt.show()

    theory = CorrelationFunctionMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.1))
    #print(theory.all_params)
    #theory.all_params['b1'].update(fixed=True)
    theory.all_params['df'].update(fixed=True)
    calculator = theory
    emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=3))
    emulator.set_samples(method='finite', accuracy=4)
    emulator.fit()
    calculator = emulator.to_calculator()
    ax = plt.gca()
    for qpar, dm, df in [(0.95, -0.05, 0.95), (1., 0., 1.), (1.05, 0.05, 1.05)]:
        params = dict(qpar=qpar, dm=0., df=1.)
        #params = dict(qpar=1., dm=dm, df=1.)
        #params = dict(qpar=1., dm=0., df=df)
        xi = theory(**params)
        for ill, ell in enumerate(theory.ells):
            ax.plot(theory.s, theory.s**2 * xi[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        xi = calculator(**params)
        for ill, ell in enumerate(calculator.ells):
            ax.plot(calculator.s, calculator.s**2 * xi[ill], color='C{:d}'.format(ill), linestyle='--')
    plt.show()


def test_ap_diff():

    from matplotlib import pyplot as plt
    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate
    from desilike.theories.galaxy_clustering import (KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles,
                                                     LPTVelocileptorsTracerCorrelationFunctionMultipoles, PyBirdTracerCorrelationFunctionMultipoles,
                                                     EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles)
    from desilike.emulators import Emulator, TaylorEmulatorEngine


    for Theory in [KaiserTracerPowerSpectrumMultipoles][:0]:
        theory = Theory(template=ShapeFitPowerSpectrumTemplate(z=1.1))
        ax = plt.gca()
        pk_ref = theory()
        for qpar in [0.998, 1., 1.002]:
            pk = theory(qpar=qpar)
            for ill, ell in enumerate(theory.ells):
                ax.plot(theory.k, theory.k * (pk[ill] - pk_ref[ill]), color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        plt.show()

    for Theory in [KaiserTracerCorrelationFunctionMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles,
                   PyBirdTracerCorrelationFunctionMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                   LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles][2:]:

        theory = Theory(s=np.linspace(10., 200., 1000), template=ShapeFitPowerSpectrumTemplate(z=1.1))
        ax = plt.gca()
        xi_ref = theory()
        for qpar in [0.998, 1., 1.002]:
            xi = theory(qpar=qpar)
            for ill, ell in enumerate(theory.ells):
                ax.plot(theory.s, theory.s**2 * (xi[ill] - xi_ref[ill]), color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        plt.show()
    exit()

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.1))
    theory.all_params['b1'].update(fixed=True)
    theory.all_params['df'].update(fixed=True)
    calculator = theory
    emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=4))
    emulator.set_samples(method='finite', accuracy=4)
    emulator.fit()
    calculator = emulator.to_calculator()
    ax = plt.gca()
    for qpar, dm, df in [(0.95, -0.05, 0.95), (1., 0., 1.), (1.05, 0.05, 1.05)]:
        params = dict(qpar=qpar, dm=0., df=1.)
        #params = dict(qpar=1., dm=dm, df=1.)
        #params = dict(qpar=1., dm=0., df=df)
        pk = theory(**params)
        for ill, ell in enumerate(theory.ells):
            ax.plot(theory.k, theory.k * pk[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        pk = calculator(**params)
        for ill, ell in enumerate(calculator.ells):
            ax.plot(calculator.k, calculator.k * pk[ill], color='C{:d}'.format(ill), linestyle='--')
    plt.show()

    theory = KaiserTracerCorrelationFunctionMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.1))
    #theory.all_params['b1'].update(fixed=True)
    #theory.all_params['df'].update(fixed=True)
    calculator = theory
    emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=3))
    emulator.set_samples(method='finite')
    emulator.fit()
    calculator = emulator.to_calculator()
    ax = plt.gca()
    for qpar, dm, df in [(0.95, -0.05, 0.95), (1., 0., 1.), (1.05, 0.05, 1.05)]:
        params = dict(qpar=qpar, dm=0., df=1.)
        #params = dict(qpar=1., dm=dm, df=1.)
        #params = dict(qpar=1., dm=0., df=df)
        xi = theory(**params)
        for ill, ell in enumerate(theory.ells):
            ax.plot(theory.s, theory.s**2 * xi[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        xi = calculator(**params)
        for ill, ell in enumerate(calculator.ells):
            ax.plot(calculator.s, calculator.s**2 * xi[ill], color='C{:d}'.format(ill), linestyle='--')
    plt.show()


def test_png():

    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles

    theory = PNGTracerPowerSpectrumMultipoles(method='prim')
    params = dict(fnl_loc=100., b1=2.)
    theory2 = PNGTracerPowerSpectrumMultipoles(method='matter')
    assert np.allclose(theory2(**params), theory(**params), rtol=2e-3)
    assert not np.allclose(theory2(fnl_loc=0.), theory(), rtol=2e-3)


if __name__ == '__main__':

    setup_logging()
    #test_integ()
    #test_bao()
    #test_full_shape()
    #test_pk_to_xi()
    test_ap_diff()
    #test_png()