import numpy as np

from desilike import utils


def test_misc():

    shape = (10,) * 2
    rng = np.random.RandomState(seed=42)
    covariance = rng.uniform(-0.1, 0.1, size=shape) + np.eye(*shape)
    indices = [0, 4, 6, 10]
    slices = [slice(start, stop) for start, stop in zip(indices[:-1], indices[1:])]
    blocks = [[covariance[sl1, sl2] for sl2 in slices] for sl1 in slices]
    assert np.allclose(utils.blockinv(blocks), utils.inv(covariance))


def test_logger():

    import logging
    from desilike.utils import setup_logging, LoggingContext

    setup_logging('info')

    logger = logging.getLogger('WarningContext')

    with LoggingContext('warning'):
        logger.info('This should not be printed')

    with LoggingContext():
        logger = logging.getLogger('InfoContext')
        logger.info('This should be printed')

    with LoggingContext('warning'):
        logger = logging.getLogger('WarningContext')
        logger.info('This should not be printed')

    logger = logging.getLogger('Info')
    logger.info('This should be printed')


def test_dict():

    from desilike.io import BaseConfig

    class Test(BaseConfig):

        def __delitem__(self, name):
            print('delitem', name)
            super(Test, self).__delitem__(name)

        def __setitem__(self, name, value):
            print('setitem', name, value)
            super(Test, self).__setitem__(name, value)

    test = Test()
    test.update(b=3)
    test.pop('b')


if __name__ == '__main__':

    #test_misc()
    #test_logger()
    test_dict()
