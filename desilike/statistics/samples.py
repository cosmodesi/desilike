"""Module implementing the samples."""

from pathlib import Path

try:
    import h5py
    H5PY_INSTALLED = True
except ModuleNotFoundError:
    H5PY_INSTALLED = False
import numpy as np
from scipy.special import logsumexp

from desilike.utils import BaseClass


class Samples(BaseClass):
    """Class for storing samples of parameters."""

    def __init__(self, latex=dict(), profiled=None, **kwargs):
        """Initialize a sample of parameters.

        Parameters
        ----------
        latex : dict or None, optional
            LaTeX expression for parameters. Default is ``None``.
        profiled : array-like or None, optional
            List of parameters being profiled, i.e., the ones for which the
            maximum likelihood or posterior is evaluated while other parameters
            are optimized. Default is ``dict()``.
        **kwargs : dict, optional
            Samples of parameters. Each sample must have the same length.

        Raises
        ------
        ValueError
            If not all samples have the same length.

        """
        self.data = {}
        self.n_samples = None
        for key, value in kwargs.items():
            if self.n_samples is None:
                self.n_samples = len(value)
            elif len(value) != self.n_samples:
                raise ValueError("All inputs must have the same length.")
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            self.data[key] = value
        self.latex = latex
        self.profiled = list(profiled) if profiled is not None else []

    @property
    def keys(self):
        """Return the keys of the sample as a list of strings."""
        return list(self.data.keys())

    def __setitem__(self, key, value):
        """Set or add a new sample.

        Parameters
        ----------
        key : str
            Key to use.
        value : array-like
            Value for that key.

        Raises
        ------
        ValueError
            If new sample does not have the same length as current samples.

        """
        if self.n_samples is None:
            self.n_samples = len(value)
        if len(value) != self.n_samples:
            raise ValueError(
                f"Input array must have length {self.n_samples}. Received "
                f"array of length {len(value)}.")
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        self.data[key] = value

    def __getitem__(self, key):
        """Get a sample by column or row(s).

        Parameters
        ----------
        key : str, slice, or int
            Key, slice, or number to use.

        Returns
        -------
        result : numpy.ndarray or desilike.statistics.Samples
            If ``key`` is a ``str``, the value, i.e., column, for that key.
            If ``key`` is a slice, a new ``Samples`` object corresponding to
            those rows. If ``key`` is an integer, a dictionary corresponding
            to that row.

        Raises
        ------
        TypeError
            If ``key`` is not a string, slice, or integer.

        """
        if isinstance(key, str):
            if isinstance(self.data[key], list):
                self.data[key] = np.concatenate(self.data[key])
            return self.data[key]
        elif isinstance(key, slice):
            return self.__class__(latex=self.latex, profiled=self.profiled,
                                  **{k: self[k][key] for k in self.keys})
        elif isinstance(key, int):
            return {k: v[key] for k, v in self.data.items()}
        else:
            raise TypeError(
                "Data can only be accessed via strings, slices, or integers.")

    def __len__(self):
        """Return the number of samples."""
        return 0 if self.n_samples is None else self.n_samples

    def append(self, samples):
        """Append a sample, i.e., add additional rows.

        Parameters
        ----------
        samples : desilike.statistics.Samples
            Samples to add. Must have the same keys as the current samples.

        Raises
        ------
        ValueError
            If keys do not match.

        """
        if set(self.keys) != set(samples.keys):
            raise ValueError("Keys do not match.")

        for key in self.keys:
            if isinstance(self.data[key], list):
                self.data[key].append(samples[key])
            else:
                self.data[key] = [self.data[key], samples[key]]

        self.n_samples += len(samples)

    def __repr__(self):
        """Get a summary of the samples."""
        return f"<Samples: n={len(self)}, keys=[{', '.join(self.keys)}]>"

    def save(self, filepath, keys=None):
        """Save samples to a file.

        This function supports ``csv``, ``npz``, and ``hdf5`` file
        endings. ``csv`` is typically used for sharing results outside of
        ``desilike``.

        Parameters
        ----------
        filepath: str or Path
            Where to save samples.
        keys : list or None, optional
            Keys to write. If ``None``, all keys are used. Default is ``None``.

        Raises
        ------
        ValueError
            If file ending is not supported, file ending is ``hdf5`` but
            ``h5py`` is not installed, or parameters to be saved are
            multidimensional and the output is ``csv``.

        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        keys = list(self.keys) if keys is None else keys

        data = {key: self[key] for key in keys}

        if suffix == '.csv':
            for key, value in data.items():
                if not value.ndim == 1:
                    raise ValueError(
                        f"Data for key '{key}' is multidimensional.")

        if suffix == '.csv':
            np.savetxt(
                filepath, np.column_stack(list(data.values())),
                header=','.join(data.keys()), delimiter=',')
        elif suffix in ['.npz', '.hdf5', '.h5']:
            profiled = np.asarray(self.profiled, dtype='U')
            latex_keys = np.asarray(list(self.latex.keys()), dtype='U')
            latex_values = np.asarray(list(self.latex.values()), dtype='U')

            if suffix == '.npz':
                np.savez(
                    filepath, latex_keys=latex_keys, latex_values=latex_values,
                    profiled=profiled, **data)
            elif suffix in ['.hdf5', '.h5']:
                if not H5PY_INSTALLED:
                    raise ValueError(
                        "`h5py` is required to save samples to HDF5 files.")
                with h5py.File(filepath, 'w') as fstream:
                    fstream['latex_keys'] = latex_keys
                    fstream['latex_values'] = latex_values
                    fstream['profiled'] = profiled
                    for key, value in data.items():
                        fstream[key] = value
        else:
            raise ValueError(f"File ending '{suffix}' not supported.")

    @classmethod
    def load(cls, filepath):
        """Read samples from a file.

        This function supports ``npz``, and ``hdf5`` file endings.

        Parameters
        ----------
        filepath: str or Path
            Where to read samples from.

        Raises
        ------
        ValueError
            If file ending is not supported or file ending is ``hdf5`` but
            ``h5py`` is not installed.

        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == '.npz':
            data = np.load(filepath)
            data = {key: data[key] for key in data}
        elif suffix in ['.hdf5', '.h5']:
            if not H5PY_INSTALLED:
                raise ValueError(
                    "You need `h5py` to read samples to HDF5 files.")
            data = dict()
            with h5py.File(filepath, 'r') as fstream:
                for key in fstream:
                    data[key] = fstream[key][()]
        else:
            raise ValueError(f"File ending '{suffix}' not supported.")

        latex_keys = data.pop('latex_keys')
        latex_values = data.pop('latex_values')
        latex = {key: value for key, value in zip(latex_keys, latex_values)}
        profiled = list(data.pop('profiled'))

        return cls(latex=latex, profiled=profiled, **data)

    @property
    def weight(self):
        """Return the (normalized) weight of each sample."""
        if 'log_weight' in self.keys:
            return np.exp(self['log_weight'] - logsumexp(self['log_weight']))
        else:
            return np.ones(self.n_samples) / self.n_samples

    def mean(self, keys=None, return_as_dict=False):
        """Compute the mean of the sample.

        Parameters
        ----------
        keys : list or None, optional
            Keys to compute the mean for. If ``None``, all keys are used.
            Default is ``None``.
        return_as_dict : bool, optional
            If ``True``, return a dictionary. Otherwise, return a numpy
            array. Default is ``False``.

        Returns
        -------
        means : list or dict
            Means of the samples.

        """
        keys = list(self.keys) if keys is None else keys

        means = [np.average(self[key], weights=self.weight, axis=0) for key in
                 keys]

        if return_as_dict:
            return dict(zip(keys, means))
        else:
            return means

    def covariance(self, keys=None):
        """Compute the covariance of the sample.

        Parameters
        ----------
        keys : list or None, optional
            Keys to compute the covariance for. If ``None``, all keys are used.
            Default is ``None``.

        Returns
        -------
        cov : numpy.ndarray
            Covariance of the samples. The ordering is the same as ``keys``
            or ``self.keys`` if ``keys`` is ``None``.

        """
        keys = list(self.keys) if keys is None else keys

        m = np.column_stack([
            self[key].reshape(self.n_samples, -1) for key in keys])

        return np.cov(m, aweights=self.weight, rowvar=False)

    def copy(self):
        """Return a copy of the samples object."""
        kwargs = {key: self.data[key].copy() for key in self.keys}
        return self.__class__(
            latex=self.latex.copy(), profiled=self.profiled.copy(), **kwargs)

    @classmethod
    def concatenate(cls, samples):
        """Concatenate samples.

        Parameters
        ----------
        samples : list of desilike.Samples
            Samples to concatenate.

        Returns
        -------
        combined : desilike.Samples
            Concatenated samples.

        """
        if not samples:
            return cls()
        combined = samples[0].copy()
        for sample in samples[1:]:
            combined.append(sample)
        return combined
