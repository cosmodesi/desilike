
import numpy as np
import scipy as sp
import lsstypes as types


def test_compression():

    from desilike.observables.galaxy_clustering import (BAOCompressionObservable, BAOPhaseShiftCompressionObservable, StandardCompressionObservable,
                                                        ShapeFitCompressionObservable, WiggleSplitCompressionObservable, BandVelocityCompressionObservable,
                                                        TurnOverCompressionObservable)
    from desilike.observables.lya import P1DCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.emulators import Emulator, TaylorEmulatorEngine

    def test_emulator(likelihood):
        likelihood_bak = likelihood()
        emulator = Emulator(likelihood.observables, engine=TaylorEmulatorEngine(order=1))
        emulator.set_samples()
        emulator.fit()
        likelihood.init.update(observables=emulator.to_calculator())
        assert np.allclose(likelihood(), likelihood_bak)

    Observables = {BAOCompressionObservable: [{'parameters': ('qpar', 'qper'), 'z': 2.},
                                              {'parameters': ('qiso',), 'z': 2., 'rs_drag_varied': True},
                                              {'parameters': ('DM_over_rd', 'DH_over_rd'), 'z': 2.},
                                              {'parameters': ('DV_over_rd',), 'z': 2.}],
                   BAOPhaseShiftCompressionObservable: [{'parameters': ('qiso', 'baoshift'), 'z': 2.}],
                   StandardCompressionObservable: [{'parameters': ('qpar', 'qper', 'df'), 'z': 2.}],
                   ShapeFitCompressionObservable: [{'parameters': ('qpar', 'qper', 'm', 'f_sqrt_Ap'), 'z': 2.}],
                   WiggleSplitCompressionObservable: [{'parameters': ('qap', 'qbao', 'df', 'dm'), 'z': 2.}],
                   BandVelocityCompressionObservable: [{'parameters': ('dptt0', 'dptt1', 'qap'), 'kp': [0.02, 0.1], 'z': 2.}],
                   TurnOverCompressionObservable: [{'parameters': ('qto',), 'z': 2.}],
                   P1DCompressionObservable: [{'parameters': ('delta2star', 'nstar'), 'z': 3.}]}

    for Observable, list_kwargs in Observables.items():
        for kwargs in list_kwargs:
            parameters = kwargs.pop('parameters')
            data_value = [1.] * len(parameters)
            covariance_value = np.diag([0.01] * len(parameters))
            leaves = [types.ObservableLeaf(value=np.atleast_1d(value)) for value in data_value]
            data = types.ObservableTree(leaves, parameters=parameters)
            covariance = types.CovarianceMatrix(value=covariance_value, observable=data)
            observable = Observable(data=data_value, covariance=covariance_value, parameters=parameters, **kwargs)
            observable()
            likelihood = ObservablesGaussianLikelihood(observables=[observable])
            test_emulator(likelihood)
            likelihood = ObservablesGaussianLikelihood(observables=[observable])
            observable2 = Observable(data=data, covariance=covariance, name='bao2', **kwargs)
            likelihood2 = ObservablesGaussianLikelihood(observables=[observable2])
            likelihood(), likelihood2()
            assert np.allclose(likelihood2.loglikelihood, likelihood.loglikelihood)
            likelihood2 = ObservablesGaussianLikelihood(observables=[observable, observable2], covariance=np.diag([0.01] * (2 * len(parameters))))
            likelihood(), likelihood2()
            assert np.allclose(likelihood2.loglikelihood, 2 * likelihood.loglikelihood)


def test_clustering():

    from desilike.observables.galaxy_clustering import TracerSpectrum2PolesObservable, TracerSpectrum3PolesObservable, TracerCorrelation2PolesObservable
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    from desilike.likelihoods import ObservablesGaussianLikelihood

    def get_spectrum2_data(size=10):
        edges = np.linspace(0., 0.2, size + 1)
        edges = np.column_stack([edges[:-1], edges[1:]])
        k = np.mean(edges, axis=-1)
        value = np.zeros_like(k)
        ells = [0, 2, 4]
        data = [types.Mesh2SpectrumPole(k=k, num_raw=value, k_edges=edges, ell=ell) for ell in ells]
        return types.Mesh2SpectrumPoles(data)

    def get_spectrum2_window(observable, size=20):
        edges = np.linspace(0., 0.2, size + 1)
        edges = np.column_stack([edges[:-1], edges[1:]])
        k = np.mean(edges, axis=-1)
        ells = [0, 2, 4]
        theory = [types.Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), k_edges=edges, ell=ell) for ell in ells]
        theory = types.ObservableTree(theory, ells=ells, wa_orders=[0] * len(ells))
        window = np.zeros((observable.size, theory.size))
        return types.WindowMatrix(observable=observable, theory=theory, value=window)

    def get_spectrum3_data(size=10):
        edges = np.linspace(0., 0.2, size + 1)
        edges = np.column_stack([edges[:-1], edges[1:]])
        edges = np.concatenate([edges[:, None, :]] * 2, axis=1)
        k = np.mean(edges, axis=-1)
        value = np.zeros_like(k[..., 0])
        ells = [(0, 0, 0), (2, 0, 2)]
        data = [types.Mesh3SpectrumPole(k=k, num_raw=value, k_edges=edges, basis='sugiyama-diagonal', ell=ell) for ell in ells]
        return types.Mesh3SpectrumPoles(data)

    def get_spectrum3_window(observable, size=20):
        edges = np.linspace(0., 0.2, size + 1)
        edges = np.column_stack([edges[:-1], edges[1:]])
        k = np.mean(edges, axis=-1)

        def get_grid(*arrays):
            arrays = np.meshgrid(*arrays, indexing='ij')
            return np.column_stack([array.ravel() for array in arrays])

        edges = np.column_stack([get_grid(edges[..., axis], edges[..., axis])[:, None, :] for axis in range(2)])
        k = get_grid(k, k)
        ells = [(0, 0, 0), (2, 0, 2), (1, 1, 0)]
        theory = [types.Mesh3SpectrumPole(k=k, num_raw=np.zeros_like(k[..., 0]), k_edges=edges, basis='sugiyama', ell=ell) for ell in ells]
        theory = types.Mesh3SpectrumPoles(theory)
        window = np.zeros((observable.size, theory.size))
        return types.WindowMatrix(observable=observable, theory=theory, value=window)

    def get_correlation2_data_window():

        def get_count(seed=42):
            rng = np.random.RandomState(seed=seed)
            coords = ['s', 'mu']
            edges = [np.linspace(0., 200., 201), np.linspace(-1., 1., 101)]
            edges = [np.column_stack([edge[:-1], edge[1:]]) for edge in edges]
            coords_values = [np.mean(edge, axis=-1) for edge in edges]
            counts = 1. + rng.uniform(size=tuple(v.size for v in coords_values))
            return types.Count2(counts=counts, norm=np.ones_like(counts), **{coord: value for coord, value in zip(coords, coords_values)},
                        **{f'{coord}_edges': value for coord, value in zip(coords, edges)}, coords=coords, attrs=dict(los='x'))

        counts = {label: get_count(seed=i) for i, label in enumerate(['DD', 'DR', 'RD', 'RR'])}
        correlation = types.Count2Correlation(**counts)
        return correlation.project(ells=[0, 2, 4], kw_window=dict(RR=correlation.get('RR')))

    def get_covariance(observable):
        covariance = np.eye(observable.size)
        return types.CovarianceMatrix(observable=observable, value=covariance)

    def test_observable(observable):
        observable()
        observable.plot()
        #if hasattr(Observable, 'plot_bao'):
        #    observable.plot_bao()

    # Test Fourier-space observables first
    Observables = {TracerSpectrum2PolesObservable: [{'data': get_spectrum2_data(),
                                                     'window': get_spectrum2_window(get_spectrum2_data()),
                                                     'theory': KaiserTracerPowerSpectrumMultipoles()}],
                  TracerSpectrum3PolesObservable: [{'data': get_spectrum3_data(),
                                                    'window': get_spectrum3_window(get_spectrum3_data()),
                                                    'theory': FOLPSv2TracerBispectrumMultipoles()}]}
    Observables = {}
    for Observable, list_kwargs in Observables.items():
        for kwargs in list_kwargs:
            data, window, theory = kwargs['data'], kwargs['window'], kwargs['theory']
            covariance = get_covariance(data)
            observable = Observable(**kwargs, covariance=covariance, name='observable1')
            test_observable(observable)
            observable2 = Observable(data=data.value(), window=window.value(), k=[pole.coords('k') for pole in data], ells=data.ells,
                                     ellsin=window.theory.ells, kin=next(iter(window.theory)).coords('k'), covariance=covariance.value(), theory=theory)
            test_observable(observable2)
            assert np.allclose(observable2.flatdata, observable.flatdata)
            shotnoise = 2e3
            observable2 = Observable(data=data.value(), k=[pole.coords('k') for pole in data], ells=data.ells, shotnoise=shotnoise,
                                     theory=theory, name='observable2')
            observable2()
            if hasattr(observable2.theory, 'snd'):
                assert np.allclose(observable2.theory.snd, shotnoise / 1e4)
            else:
                assert np.allclose(observable2.theory.nd, 1. / shotnoise)

            observable = Observable(**kwargs, covariance=covariance, name='observable1')
            observable2 = Observable(**kwargs, covariance=covariance, name='observable2')
            likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance.value())
            likelihood()
            covariance3 = types.CovarianceMatrix(observable=types.ObservableTree([data] * 3, observables=['observable1', 'observable2', 'observable3']),
                                                value=sp.linalg.block_diag(*[covariance.value()] * 3))
            likelihood2 = ObservablesGaussianLikelihood(observables=[observable, observable2], covariance=covariance3)
            likelihood2()
            assert np.allclose(likelihood2.loglikelihood, 2 * likelihood.loglikelihood)
            likelihood3 = ObservablesGaussianLikelihood(observables=[observable, observable2],
                                                        covariance=covariance3.at.observable.get(observables=['observable1', 'observable2']))
            likelihood3()
            assert np.allclose(likelihood3.loglikelihood, 2 * likelihood.loglikelihood)

    # Then configuration-space observables
    data, window = get_correlation2_data_window()
    Observables = {TracerCorrelation2PolesObservable: [{'data': data, 'window': window,
                                                     'theory': KaiserTracerCorrelationFunctionMultipoles()}]}

    for Observable, list_kwargs in Observables.items():
        for kwargs in list_kwargs:
            data, window, theory = kwargs['data'], kwargs['window'], kwargs['theory']
            covariance = get_covariance(data)
            observable = Observable(**kwargs, covariance=covariance, name='observable1')
            test_observable(observable)
            covariance = get_covariance(data)
            observable = Observable(**kwargs, covariance=covariance, name='observable1')
            test_observable(observable)
            observable2 = Observable(data=data.value(), window=window.value(), s=[pole.coords('s') for pole in data], ells=data.ells,
                                     ellsin=window.theory.ells, sin=next(iter(window.theory)).coords('s'), covariance=covariance.value(), theory=theory)
            test_observable(observable2)
            assert np.allclose(observable2.flatdata, observable.flatdata)
            observable = Observable(**kwargs, covariance=covariance, name='observable1')
            observable2 = Observable(**kwargs, covariance=covariance, name='observable2')
            likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance.value())
            likelihood()
            covariance3 = types.CovarianceMatrix(observable=types.ObservableTree([data] * 3, observables=['observable1', 'observable2', 'observable3']),
                                                value=sp.linalg.block_diag(*[covariance.value()] * 3))
            likelihood2 = ObservablesGaussianLikelihood(observables=[observable, observable2], covariance=covariance3)
            likelihood2()
            assert np.allclose(likelihood2.loglikelihood, 2 * likelihood.loglikelihood)

            likelihood3 = ObservablesGaussianLikelihood(observables=[observable, observable2],
                                                        covariance=covariance3.at.observable.get(observables=['observable1', 'observable2']))
            likelihood3()
            assert np.allclose(likelihood3.loglikelihood, 2 * likelihood.loglikelihood)