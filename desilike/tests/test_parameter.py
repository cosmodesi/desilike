import numpy as np

from desilike.parameter import Parameter, Deriv, ParameterArray, Samples, ParameterPrecision, ParameterCovariance


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
    array *= 2
    array = ParameterArray(np.ones(4), param=param)
    print((array + array).param)
    array = ParameterArray(np.ones((1, 4)), param=param, derivs=[(param,)])
    print((array + array)[param])
    array[param] += 1.
    print((array + array)[param])

    samples = Samples([array])
    print(samples[:10]['a'].derivs)


def test_matrix():
    params = {name: {'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}} for name in ['a', 'b', 'c']}
    #params = {name: {'prior': {'limits': [-1., 1.]}} for name in ['a', 'b']}
    precision = ParameterPrecision(np.eye(len(params)), params=params)
    covariance = (precision + precision).to_covariance()
    print(covariance.to_stats())
    gd = covariance.to_getdist(center=[0.] * len(params))

    precision = ParameterPrecision(np.eye(len(params)), params=params, center=[0.] * len(params))
    covariance = precision.to_covariance()
    #covariance.to_getdist()

    from desilike.samples import plotting
    plotting.plot_triangle(covariance, show=True)

    covariance = covariance.view(params=['a'])


if __name__ == '__main__':

    #test_deriv()
    #test_param_array()
    test_matrix()
