import numpy as np

from desilike import utils
from desilike.jax import numpy as jnp
from .base import BaseEmulatorEngine
from desilike import mpi


def _make_tuple(obj, length=None):
    """
    Return tuple from ``obj``.

    Parameters
    ----------
    obj : object, tuple, list, array
        If tuple, list or array, cast to list.
        Else return tuple of ``obj`` with length ``length``.

    length : int, default=1
        Length of tuple to return, if ``obj`` not already tuple, list or array.

    Returns
    -------
    toret : tuple
    """
    if np.ndim(obj) == 0:
        obj = (obj,)
        if length is not None:
            obj *= length
    return tuple(obj)


class MLPEmulatorEngine(BaseEmulatorEngine):

    name = 'mlp'

    def initialize(self, varied_params, nhidden=(100, 100, 100), ytransform='', npcs=None, engine='rqrs', niterations=int(1e5)):
        self.nhidden = tuple(nhidden)
        self.npcs = npcs
        self.ytransform = str(ytransform)
        self.sampler_options = dict(engine=engine, niterations=niterations)

    def get_default_samples(self, calculator, **kwargs):
        from desilike.samplers import QMCSampler
        options = {**self.sampler_options, **kwargs}
        sampler = QMCSampler(calculator, engine=options['engine'], mpicomm=self.mpicomm)
        sampler.run(niterations=options['niterations'])
        return sampler.samples

    def fit(self, X, Y, validation_frac=0.2, optimizer='adam', batch_sizes=(320, 640, 1280, 2560, 5120), epochs=1000, learning_rates=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6), seed=None):

        optimizer = str(optimizer)
        validation_frac = float(validation_frac)
        batch_sizes = _make_tuple(batch_sizes, length=1)
        epochs = _make_tuple(epochs, length=len(batch_sizes))
        learning_rates = _make_tuple(learning_rates, length=len(batch_sizes))
        rng = np.random.RandomState(seed=seed)

        self.operations = None

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        class TFModel(tf.keras.Model):

            def __init__(self, architecture, eigenvectors=None, mean=None, sigma=None):
                super(TFModel, self).__init__()
                self.architecture = architecture
                self.nlayers = len(self.architecture) - 1
                self.mean = mean
                self.sigma = sigma
                self.eigenvectors = eigenvectors

                self.W, self.b, self.alpha, self.beta = [], [], [], []
                for i in range(self.nlayers):
                    self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i + 1]], 0., np.sqrt(2. / self.architecture[0])), name='W_{:d}'.format(i), trainable=True))
                    self.b.append(tf.Variable(tf.zeros([self.architecture[i + 1]]), name='b_{:d}'.format(i), trainable=True))
                for i in range(self.nlayers - 1):
                    self.alpha.append(tf.Variable(tf.random.normal([self.architecture[i + 1]]), name='alpha_{:d}'.format(i), trainable=True))
                    self.beta.append(tf.Variable(tf.random.normal([self.architecture[i + 1]]), name='beta_{:d}'.format(i), trainable=True))

            @tf.function
            def call(self, x):
                for i in range(self.nlayers):
                    # linear network operation
                    x = tf.add(tf.matmul(x, self.W[i]), self.b[i])
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        x = tf.multiply(tf.add(self.beta[i], tf.multiply(tf.sigmoid(tf.multiply(self.alpha[i], x)), tf.subtract(1., self.beta[i]))), x)
                # linear output layer
                if self.eigenvectors is not None:
                    x = tf.matmul(tf.add(tf.multiply(x, self.sigma), self.mean), self.eigenvectors)
                return x

            def operations(self):
                operations = []
                for i in range(self.nlayers):
                    # linear network operation
                    operations.append({'eval': 'x @ W + b', 'locals': {'W': self.W[i].numpy(), 'b': self.b[i].numpy()}})
                    # non-linear activation function
                    if i < self.nlayers - 1:
                        operations.append({'eval': '(beta + (1 - beta) / (1 + np.exp(-alpha * x))) * x', 'locals': {'alpha': self.alpha[i].numpy(), 'beta': self.beta[i].numpy()}})
                # linear output layer
                if self.eigenvectors is not None:
                    operations.append({'eval': '(x * sigma + mean) @ eigenvectors', 'locals': {'eigenvectors': eigenvectors, 'mean': mean, 'sigma': sigma}})
                return operations

            def __getstate__(self):
                state = {}
                for name in ['W', 'b', 'alpha', 'beta']:
                    state[name] = [value.numpy() for value in getattr(self, name)]
                return state

            def __setstate__(self, state):
                for name in ['W', 'b', 'alpha', 'beta']:
                    for tfvalue, npvalue in zip(getattr(self, name), state[name]):
                        tfvalue.assign(npvalue)

        nsamples = self.mpicomm.bcast(len(X) if self.mpicomm.rank == 0 else None)
        nvalidation = int(nsamples * validation_frac + 0.5)
        if nvalidation >= nsamples:
            raise ValueError('Cannot use {:d} validation samples (>= {:d} total samples)'.format(nvalidation, nsamples))

        if self.mpicomm.rank == 0:
            samples = {'X': X, 'Y': Y}
            self.operations = {}
            for name, value in samples.items():
                mean, sigma = np.mean(value, axis=0), np.std(value, ddof=1, axis=0)
                self.operations[name] = [{'eval': 'x * sigma + mean' if name == 'Y' else '(x - mean) / sigma',
                                          'locals': {'mean': mean, 'sigma': sigma}}]
                samples[name] = (value - mean) / sigma
            if 'arcsinh' in self.ytransform:
                Y = np.arcsinh(samples['Y'])
                mean, sigma = np.mean(Y, axis=0), np.std(Y, ddof=1, axis=0)
                samples['Y'] = (Y - mean) / sigma
                self.operations['Y'].insert(0, {'eval': 'np.sinh(x) * sigma + mean', 'locals': {'mean': mean, 'sigma': sigma}})
            mask = np.zeros(nsamples, dtype='?')
            mask[rng.choice(nsamples, size=nvalidation, replace=False)] = True
            for name, value in list(samples.items()):
                samples['{}_validation'.format(name)] = value[mask]
                samples['{}_training'.format(name)] = value[~mask]
            eigenvectors, mean, sigma = None, None, None
            architecture = [X.shape[-1]] + list(self.nhidden)
            if self.npcs is not None:
                ndim = samples['Y_training'].shape[-1]
                if self.npcs > ndim:
                    self.log_warning('Number of requested components is {0:d}, but dimension is already {1:d} < {0:d}.'.format(self.npcs, ndim))
                    self.npcs = ndim
                eigenvectors = utils.subspace(samples['Y_training'], npcs=self.npcs)
                tmp = samples['Y_training'].dot(eigenvectors)
                eigenvectors = eigenvectors.T
                mean, sigma = np.mean(tmp, axis=0), np.std(tmp, ddof=1, axis=0)
                architecture += [len(mean)]
            else:
                architecture += [samples['Y'].shape[1]]

            tfmodel = TFModel(architecture, eigenvectors=eigenvectors, mean=mean, sigma=sigma)
            state = getattr(self, 'tfmodel', None)
            if state is not None:
                if not isinstance(state, dict):
                    state = state.__getstate__()
                tfmodel.__setstate__(state)
            self.tfmodel = tfmodel
            self.tfmodel.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

            for batch_size, epoch, lr in zip(batch_sizes, epochs, learning_rates):
                if lr is None:
                    lr = self.tfmodel.optimizer.lr.numpy()
                else:
                    self.tfmodel.optimizer.lr.assign(lr)
                self.log_info('Using (batch size, epochs, learning rate) = ({:d}, {:d}, {:.2e})'.format(batch_size, epoch, lr))
                self.tfmodel.fit(samples['X_training'], samples['Y_training'], batch_size=batch_size, epochs=epoch,
                                 validation_data=(samples['X_validation'], samples['Y_validation']), callbacks=[es], verbose=2)
                self.operations['M'] = self.tfmodel.operations()
            self.operations = self.operations['X'] + self.operations['M'] + self.operations['Y']

        mpi.barrier_idle(self.mpicomm)  # we rely on keras parallelisation; here we make MPI processes idle

        self.operations = self.mpicomm.bcast(self.operations, root=0)

    def predict(self, X):
        x = X
        for operation in self.operations:
            x = eval(operation['eval'], {'np': jnp}, {'x': x, **operation['locals']})
        return x

    def __getstate__(self):
        state = {}
        for name in ['operations', 'tfmodel']:
            if hasattr(self, name):
                tmp = getattr(self, name)
                if hasattr(tmp, '__getstate__'): tmp = tmp.__getstate__()
                state[name] = tmp
        return state

    @classmethod
    def install(cls, config):
        config.pip('tensorflow')
