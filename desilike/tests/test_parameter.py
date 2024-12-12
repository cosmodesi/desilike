import numpy as np

from desilike.parameter import Parameter, ParameterPrior, Deriv, ParameterArray, ParameterCollection, Samples, ParameterPrecision, ParameterCovariance


def test_prior():
    prior = ParameterPrior(limits=(0., 1.))
    print(prior.std())


def test_deriv():
    deriv = Deriv({'a': 0, 'b': 2})
    assert deriv['b'] == 2
    assert 'a' not in deriv
    deriv['a'] = 0
    assert 'a' not in deriv
    deriv.setdefault('a', 0)
    assert 'a' not in deriv
    deriv.update({'a': 0})
    assert 'a' not in deriv
    assert Deriv({'a': 0, 'b': 2}) == Deriv({'b': 2})
    d = Deriv({'a': 0, 'b': 2})
    print(d + d)


def test_param_array():
    param = Parameter('a', shape=4)
    array = ParameterArray(np.ones((1, 4)))
    array * 2
    array = ParameterArray(np.ones(4), param=param)
    print((array + array).param)
    array = ParameterArray(np.ones((2, 1, 4)), param=param, derivs=[(param,)])
    print((array + array)[param])
    array[param] += 1.
    print((array + array)[param])
    assert isinstance(array.ravel(), ParameterArray)
    #assert isinstance(array.reshape(-1), ParameterArray)
    assert array.reshape(-1).derivs is None
    print(array)
    samples = Samples([array])
    print(samples['a'].shape)
    array2 = array.clone(param=Parameter('b', shape=4))
    samples.set(array2)
    assert samples.shape == (2,)
    print(samples['b'].shape)
    print(samples[:10]['a'].derivs)
    samples['b'] = samples['a'].clone(param=param.clone(basename='b')) * 2
    samples['c'] = samples['a'] * 1.2
    assert samples['c'].param.name == 'c'


def test_collection():
    params = ParameterCollection(['a', 'b'])
    params2 = ParameterCollection(['b', 'c'])
    assert (params & params2).names() == ['b']
    params = ParameterCollection(['a1_2', 'b1_3', 'b2_3', 'a2_3'])
    assert params.names(name=['*1_*']) == ['a1_2', 'b1_3']
    assert params.names(name=['*1_[2:3]']) == ['a1_2']
    assert params.names(name=['*[1:3]_3']) == ['b1_3', 'b2_3', 'a2_3']


def test_matrix():
    params = {name: {'prior': {'dist': 'norm', 'loc': 1., 'scale': 10.}} for name in ['a', 'b', 'c']}
    #params = {name: {'prior': {'limits': [-1., 1.]}} for name in ['a', 'b']}
    precision = ParameterPrecision(np.eye(len(params)), params=params)
    precision2 = precision + precision
    assert np.allclose(precision2._value, 2. * precision._value)
    covariance = precision2.to_covariance()
    print(covariance.to_stats())
    gd = covariance.to_getdist(center=[0.] * len(params))

    precision = ParameterPrecision(np.eye(len(params)), params=params)
    covariance = precision.to_covariance()
    #covariance.to_getdist()

    from desilike.samples import plotting
    plotting.plot_triangle(covariance, show=True)

    from desilike.samples import plotting
    plotting.plot_triangle([covariance, covariance.select(name=['a', 'b'])], show=True)

    assert np.ndim(covariance.std('a')) == 0
    std = covariance.std()
    covariance *= 1.2
    assert np.allclose(covariance.std(), 1.2**0.5 * std)
    covariance /= 1.2
    assert np.allclose(covariance.std(), std)
    covariance = covariance.view(params=['a'], return_type=None)
    assert np.ndim(covariance.fom()) == 0


if __name__ == '__main__':

    test_param_array()
    test_prior()
    test_deriv()
    test_param_array()
    test_collection()
    test_matrix()
