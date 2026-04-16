"""Module implementing the samples."""

from pathlib import Path

try:
    import h5py
    H5PY_INSTALLED = True
except ModuleNotFoundError:
    H5PY_INSTALLED = False
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import logsumexp

from desilike.utils import BaseClass


SPECIAL_KEYS = ['fixed', 'log_weight', 'log_prior', 'log_likelihood',
                'log_posterior']


class Samples(BaseClass):
    """Class for storing samples of parameters."""

    def __init__(self, latex=dict(), fixed=None, **kwargs):
        """Initialize a sample of parameters.

        Parameters
        ----------
        latex : dict or None, optional
            LaTeX expression for parameters. Default is ``None``.
        fixed : str, array-like or None, optional
            List of parameter combinations that are fixed. Each element can be
            a string listing keys separated by a "/" or a list of strings,
            each indicating a key. Alternatively, use a single string
            if the same paramters are fixed for all samples. Default is
            ``None``.
        **kwargs
            Samples of parameters. Each sample must have the same length.

        Raises
        ------
        ValueError
            If not all samples have the same length.

        """
        if fixed is not None:
            if isinstance(fixed, str):
                fixed = '/'.join(sorted(fixed.split('/')))
            else:
                for i, element in enumerate(fixed):
                    if isinstance(element, (tuple, list, set)):
                        fixed[i] = '/'.join(list(element))
                    else:
                        fixed[i] = '/'.join(sorted(element.split('/')))
                fixed = np.asarray(fixed, dtype='U')
            kwargs['fixed'] = fixed

        self.data = {}
        self.n_samples = None
        for key, value in kwargs.items():
            self[key] = value
        self.latex = latex

    @property
    def keys(self):
        """Return the keys of the sample as a list of strings."""
        return list(self.data.keys())

    @property
    def params(self):
        """Return the parameters of the sample as a list of strings."""
        return [key for key in self.keys if key not in SPECIAL_KEYS]

    def __setitem__(self, key, value):
        """Manipulate the samples.

        Parameters
        ----------
        key : str or int
            Key (column) to modify or add. Alternatively, index (row) to
            modify.
        value : object, array-like, or dict
            Value for that key or index.

        Raises
        ------
        ValueError
            If new sample does not have the same length as current samples or
            if ``key`` contains a comma or a backslash.

        """
        if isinstance(key, str):
            if isinstance(value, str) or not hasattr(value, '__len__'):
                value = np.repeat(value, 1 if self.n_samples is None else
                                  self.n_samples)
            if self.n_samples is None:
                self.n_samples = len(value)
            if len(value) != self.n_samples:
                raise ValueError(
                    f"Input array must have length {self.n_samples}. Received "
                    f"array of length {len(value)}.")
            if ',' in key or '/' in key:
                raise ValueError("Keys cannot contain commas or backslashes.")
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            self.data[key] = value
        else:
            index = key
            for key in self.keys:
                self.data[key][index] = value[key]

    def __getitem__(self, key):
        """Get a sample by column or row(s).

        Parameters
        ----------
        key : str, slice, numpy.ndarray, or int
            Key, slice, or number to use.

        Returns
        -------
        result : numpy.ndarray or desilike.statistics.Samples
            If ``key`` is a ``str``, the value, i.e., column, for that key.
            If ``key`` is a slice or boolean array, a new ``Samples`` object
            corresponding to those rows. If ``key`` is an integer, a dictionary
            corresponding to that row.

        Raises
        ------
        TypeError
            If ``key`` is not a string, slice, or integer.

        """
        if isinstance(key, str):
            if isinstance(self.data[key], list):
                self.data[key] = np.concatenate(self.data[key])
            return self.data[key]
        elif isinstance(key, (slice, np.ndarray)):
            return self.__class__(
                latex=self.latex, **{k: self[k][key] for k in self.keys})
        elif np.issubdtype(type(key), np.integer):
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
        if len(self) == 0:
            self.data = samples.data
            self.n_samples = len(samples)
            self.latex.update(samples.latex)
        else:
            if set(self.keys) != set(samples.keys):
                raise ValueError("Keys do not match.")

            for key in samples.keys:
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
            data = {key: value for key, value in data.items() if
                    key != 'fixed'}
            for key, value in data.items():
                if not value.ndim == 1:
                    raise ValueError(
                        f"Data for key '{key}' is multidimensional.")

        if suffix == '.csv':
            np.savetxt(
                filepath, np.column_stack(list(data.values())),
                header=','.join(data.keys()), delimiter=',')
        elif suffix in ['.npz', '.hdf5', '.h5']:
            latex_keys = np.asarray(list(self.latex.keys()), dtype='U')
            latex_values = np.asarray(list(self.latex.values()), dtype='U')

            if suffix == '.npz':
                np.savez(filepath, allow_pickle=False, latex_keys=latex_keys,
                         latex_values=latex_values, **data)
            elif suffix in ['.hdf5', '.h5']:
                if not H5PY_INSTALLED:
                    raise ValueError(
                        "`h5py` is required to save samples to HDF5 files.")
                with h5py.File(filepath, 'w') as fstream:
                    dtype = h5py.string_dtype(encoding='utf-8')
                    fstream['latex_keys'] = latex_keys.astype(dtype)
                    fstream['latex_values'] = latex_values.astype(dtype)
                    for key, value in data.items():
                        if key == 'fixed':
                            fstream[key] = value.astype(dtype)
                        else:
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

        latex_keys = data.pop('latex_keys').astype('U')
        latex_values = data.pop('latex_values').astype('U')
        latex = {key: value for key, value in zip(latex_keys, latex_values)}
        if 'fixed' in data:
            fixed = data.pop('fixed').astype('U')
        else:
            fixed = None

        return cls(latex=latex, fixed=fixed, **data)

    @property
    def weight(self):
        """Return the (normalized) weight of each sample."""
        if 'log_weight' in self.keys:
            return np.exp(self['log_weight'] - logsumexp(self['log_weight']))
        else:
            return np.ones(self.n_samples) / self.n_samples

    def mean(self, params=None, return_as_dict=False):
        """Compute the mean of the sample.

        Parameters
        ----------
        params : list or None, optional
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
        if params is None:
            params = self.params

        means = [np.average(self[key], weights=self.weight, axis=0) for key in
                 params]

        if return_as_dict:
            return dict(zip(params, means))
        else:
            return means

    def covariance(self, params=None):
        """Compute the covariance of the sample.

        Parameters
        ----------
        params : list or None, optional
            Keys to compute the covariance for. If ``None``, all keys are used.
            Default is ``None``.

        Returns
        -------
        cov : numpy.ndarray
            Covariance of the samples. The ordering is the same as ``keys``
            or ``self.keys`` if ``keys`` is ``None``.

        """
        if params is None:
            params = self.params

        m = np.column_stack([
            self[key].reshape(self.n_samples, -1) for key in params])

        return np.cov(m, aweights=self.weight, rowvar=False)

    def copy(self):
        """Return a copy of the samples object."""
        kwargs = {key: self.data[key].copy() for key in self.keys}
        samples = self.__class__(latex=self.latex.copy(), **kwargs)
        return samples

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

    def _get_fixed(self):
        """Return a list of dictionaries of parameters."""
        fixed_params = []
        for i in range(len(self)):
            if len(self['fixed'][i]) > 0:
                fixed_params.append(
                    {key: self[key][i] for key in self['fixed'][i].split('/')})
            else:
                fixed_params.append({})
        return fixed_params

    def tabulate(self, keys=None, use_latex=False, **kwargs):
        """Use the `tabulate` package to print the table.

        Parameters
        ----------
        keys : array-like or None, optional
            List of keys to print. If ``None``, all columns are printed.
            Default is ``None``.
        use_latex : bool, optional
            Whether to use the LaTeX names in the columns headers. Default is
            ``False``.
        **kwargs
            Additional keyword arguments passed to :meth:`tabulate.tabulate`.

        Raises
        ------
        ImportError
            If `tabulate` is not installed.

        Returns
        -------
        str
            Table as plain text.

        """
        try:
            import tabulate
        except ImportError:
            raise ImportError(
                "The 'tabulate' package is required for 'Samples.tabulate'.")

        if keys is None:
            keys = self.keys

        if use_latex:
            latex = self.latex
        else:
            latex = {}

        kwargs = dict(tablefmt='simple_grid') | kwargs
        data = {latex.get(key, key): self.data[key] for key in keys}
        return tabulate.tabulate(data, headers='keys', **kwargs)

    def getdist(self, params=None):
        """Convert the sample into a ``getdist.MCSamples`` instance.

        Parameters
        ----------
        params : array-like or None, optional
            List of parameters to convert. If ``None``, all parameters are
            included. Default is ``None``.

        Raises
        ------
        ImportError
            If `getdist` is not installed.

        Returns
        -------
        getdist.MCSamples
            Samples converted to `getdist` format.

        """
        try:
            from getdist import MCSamples
        except ImportError:
            raise ImportError(
                "The 'tabulate' package is required for 'Samples.getdist'.")

        if params is None:
            params = self.params

        return MCSamples(
            samples=np.column_stack([self[key] for key in params]),
            weights=self.weight, names=params, labels=[
                self.latex.get(key, key).replace('$', '') for key in params])

    def interval(self, param, threshold, posterior=None):
        """Get interval where likelihood/posterior is above a threshold.

        Parameters
        ----------
        param : str
            Parmater for which to get interval.
        threshold : float
            Threshold such that the likelihood/posterior is at least
            its maximum plus the threshold. Must be positive.
        posterior: bool or None, optional
            Whether to use the posterior or likelihood. If ``None``, determine
            based on what is computed. Default is ``None``.

        Raises
        ------
        ValueError
            If ``posterior`` is ``None`` but both posterior and likelihood
            have been computed, if there are not enough points to compute the
            interval, or if ``threshold`` is not positive.

        Returns
        -------
        x_min : float
            Lowest value at threshold.
        x_opt : float
            Value where likelihood/posterior is maximal.
        x_max : float
            Highest value at threshold.

        """
        if posterior is None:
            if 'log_posterior' in self.keys:
                if 'log_likelihood' in self.keys:
                    raise ValueError(
                        "Samples have both posterior and likelihood.")
                key = 'log_posterior'
            else:
                key = 'log_likelihood'

        if not threshold > 0:
            raise ValueError("'threshold' must positive.")

        use = np.isin(self['fixed'], [param, ''])

        if np.sum(use) < 4:
            raise ValueError("Not enough points to compute interval.")

        x = self[param][use]
        y = self[key][use]
        y = y[np.argsort(x)]
        x = np.sort(x)

        interp = interp1d(x, y, kind='cubic')
        bounds = (np.amin(x), np.amax(x))

        def f(x):
            return -interp(x)
        res = minimize_scalar(f, bounds=(np.amin(x), np.amax(x)))

        # TODO: Add robustness.
        x_opt = res.x
        y_max = -res.fun

        def f(x):
            return interp(x) - (y_max - threshold)

        x = np.linspace(*bounds, 1000)
        y = interp(x)

        res = root_scalar(
            f, bracket=(bounds[0], x_opt),
            x0=np.amin(x[y > y_max - threshold]))
        x_min = res.root

        res = root_scalar(
            f, bracket=(x_opt, bounds[1]),
            x0=np.amax(x[y > y_max - threshold]))
        x_max = res.root

        return x_min, x_opt, x_max
