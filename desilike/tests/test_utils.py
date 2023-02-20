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

    with LoggingContext():
        logger = logging.getLogger('InfoContext')
        logger.info('This should be printed')

    with LoggingContext('warning'):
        logger = logging.getLogger('WarningContext')
        logger.info('This should not be printed')

    logger = logging.getLogger('Info')
    logger.info('This should be printed')


if __name__ == '__main__':

    #test_misc()
    test_logger()
