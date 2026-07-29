"""Module implementing the samples."""

import re
from pathlib import Path

try:
    import h5py
    H5PY_INSTALLED = True
except ModuleNotFoundError:
    H5PY_INSTALLED = False
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.special import logsumexp

from desilike.utils import BaseClass

SPECIAL_KEYS = ['log_weight', 'log_prior', 'log_likelihood', 'log_posterior',
                'flag_.*']
FLAGS = ['optimize']


def _sort_into_grid(x, y):
    """Sort points into a regular grid.

    Parameters
    ----------
    x : numpy.ndarray of shape (n, m)
        Coordinates of grid points.
    y : numpy.ndarray of shape (n, )
        Associated values.

    Returns
    -------
    x_grid : tuple of numpy.ndarray with shapes (k1, ), ..., (kn, )
        Points defining the grid in each dimension.
    y : numpy.ndarray of of shape (k1, ..., kn)
        Associated values.

    """
    # Sort values.
    for i in reversed(range(x.shape[1])):
        idx = np.argsort(x[:, i], stable=True)
        x = x[idx]
        y = y[idx]

    # Create the unique values and the grid.
    x_grid = []
    for i in range(x.shape[1]):
        x_grid.append(np.unique(x[:, i]))
    y = y.reshape(tuple([len(x) for x in x_grid]))

    return x_grid, y


class Samples(BaseClass):
    """Class for storing samples of parameters."""

    def __init__(self, latex=None, **kwargs):
        """Initialize a sample of parameters.

        Parameters
        ----------
        latex : dict or None, optional
            LaTeX expression for parameters. Default is ``None``.
        **kwargs
            Samples of parameters. Each sample must have the same length.

        Raises
        ------
        ValueError
            If not all samples have the same length.

        """
        self.data = {}
        self.n_samples = None
        for key, value in kwargs.items():
            self[key] = value
        if latex is None:
            latex = {}
        self.latex = latex

    @property
    def keys(self):
        """Return the keys of the sample as a list of strings."""
        return list(self.data.keys())

    @property
    def params(self):
        """Return the parameters of the sample as a list of strings."""
        params = []
        for key in self.keys:
            match = False
            for special_key in SPECIAL_KEYS:
                if re.fullmatch(special_key, key):
                    match = True
                    break
            if not match:
                params.append(key)
        return params

    def __setitem__(self, key, value):
        """Manipulate the samples.

        Parameters
        ----------
        key : str or int
            Key (column) to modify or add. Alternatively, index (row) to
            modify.
        value : object, array-like, or dict
            Value for that key or index. Can also be a single value for a all
            rows in a specific column.

        Raises
        ------
        ValueError
            - If new sample does not have the same length as current samples.
            - If setting a column to a single value but object has no length.
            - If `key` is a string but not in a valid format.

        """
        if isinstance(key, str):
            for forbidden in [':', '/', ',']:
                if forbidden in key:
                    msg = "Keys cannot contain '{forbidden}'."
                    raise ValueError(msg)
            if isinstance(value, str) or not hasattr(value, '__len__'):
                if self.n_samples is None:
                    msg = "Samples have no specified length."
                    raise ValueError(msg)
                value = np.repeat(value, self.n_samples)
            if self.n_samples is None:
                self.n_samples = len(value)
            if len(value) != self.n_samples:
                raise ValueError(
                    f"Input array must have length {self.n_samples}. Received "
                    f"array of length {len(value)}.")
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            self.data[key] = value
        else:
            index = key
            for key in self.keys:  # noqa: PLR1704
                self.data[key][index] = value[key]

    def __getitem__(self, key):
        """Get a sample by column or row(s).

        Parameters
        ----------
        key : str, slice, numpy.ndarray, or int
            Key, slice, filter array, or number to use.

        Returns
        -------
        result : numpy.ndarray or desilike.statistics.Samples
            If ``key`` is a ``str``, the value, i.e., column, for that key.
            If ``key`` is a slice or filter array, a new ``Samples`` object
            corresponding to those rows. If ``key`` is an integer, a dictionary
            corresponding to that row.

        Raises
        ------
        TypeError
            If ``key`` is not a string, slice, filter array, or integer.

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
                    fstream['latex_keys'] = latex_keys.astype(
                        h5py.string_dtype(encoding='utf-8'))
                    fstream['latex_values'] = latex_values.astype(
                        h5py.string_dtype(encoding='utf-8'))
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
            data = {}
            with h5py.File(filepath, 'r') as fstream:
                for key in fstream:
                    data[key] = fstream[key][()]
        else:
            raise ValueError(f"File ending '{suffix}' not supported.")

        latex_keys = data.pop('latex_keys').astype('U')
        latex_values = data.pop('latex_values').astype('U')
        latex = {key: value for key, value in zip(latex_keys, latex_values)}

        return cls(latex=latex, **data)

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

    def _check_valid_flag(self, flag, param):
        """Check if the flag and/or parameter is valid."""
        if flag not in FLAGS:
            msg = f"Unknown flag '{flag}'. Known flags are {FLAGS}."
            raise ValueError(msg)
        if param not in self.params:
            msg = (f"Unknown parameter '{param}'. Known parameters are "
                   f"{self.params}.")
            raise ValueError(msg)

    def get_flag(self, flag, param):
        """Get the value of the status flag for all samples.

        Parameters
        ----------
        flag : str
            Status flag.
        param : str or None, optional
            The parameter to which the flag applies.

        Returns
        -------
        value : numpy.ndarray
            Boolean array contain the status flag for each sample.

        Raises
        ------
        ValueError
            If the status is not known, the parameter does not exist for this
            sample, or the flag has not been set for this specific combination
            of status and parameter.

        """
        self._check_valid_flag(flag, param)
        if f'flag_{flag}_{param}' in self.keys:
            return self[f'flag_{flag}_{param}']
        else:
            msg = f"Flag '{flag}' not set for parameter '{param}'."
            raise ValueError(msg)

    def set_flag(self, flag, param, value):
        """Get the value of the status flag for all samples.

        Parameters
        ----------
        flag : str
            Status flag.
        param : str or None, optional
            The parameter to which the flag applies.
        value : numpy.ndarray
            Boolean array contain the status flag for each sample.

        Raises
        ------
        ValueError
            If the status is not known or the parameter does not exist for
            this sample.

        """
        self._check_valid_flag(flag, param)
        self[f'flag_{flag}_{param}'] = value

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

        Returns
        -------
        table : str
            Table as plain text.

        Raises
        ------
        ImportError
            If `tabulate` is not installed.

        """
        try:
            import tabulate
        except ImportError:
            msg = "The `tabulate` package is required for `Samples.tabulate`."
            raise ImportError(msg)

        if keys is None:
            keys = self.keys

        if use_latex:
            latex = self.latex
        else:
            latex = {}

        if 'tablefmt' not in kwargs:
            kwargs['tablefmt'] = 'simple_grid'

        data = {latex.get(key, key): self.data[key] for key in keys}
        return tabulate.tabulate(data, headers='keys', **kwargs)

    def getdist(self, params=None):
        """Convert the sample into a ``getdist.MCSamples`` instance.

        Parameters
        ----------
        params : array-like or None, optional
            List of parameters to convert. If ``None``, all parameters are
            included. Default is ``None``.

        Returns
        -------
        getdist.MCSamples
            Samples converted to `getdist` format.

        Raises
        ------
        ImportError
            If `getdist` is not installed.

        """
        try:
            from getdist import MCSamples
        except ImportError:
            msg = "The `getdist` package is required for `Samples.getdist`."
            raise ImportError(msg)

        if params is None:
            params = self.params

        return MCSamples(
            samples=np.column_stack([self[key] for key in params]),
            weights=self.weight, names=params, labels=[
                self.latex.get(key, key).replace('$', '') for key in params])

    def profile_interpolator(self, params, posterior=True):
        """Get a cubic profile interpolator.

        Parameters
        ----------
        params : str or list
            Parameter(s) for which to compute the interpolator.
        posterior : bool, optional
            If ``True``, get a profile for the (log) posterior. If ``False``, a
            profile for the (log) likelihood is returned. Default is ``True``.

        Returns
        -------
        interp : scipy.interpolate.CubicSpline or\
                 scipy.interpolate.RegularGridInterpolator
            Profile interpolator.

        Raises
        ------
        ValueError
            If there are not enough points to compute an interpolation.

        """
        use = np.ones(len(self), dtype=bool)
        params = np.atleast_1d(params)
        for param in self.params:
            # In case only one parameter is requested, use even the case
            # where the parameter itself is optimized. In all other cases, the
            # grid will not be regular, so don't.
            if param in params and len(params) > 1:
                use = use & ~self.get_flag('optimize', param)
            elif param not in params:
                try:
                    use = use & self.get_flag('optimize', param)
                except ValueError:
                    # Flag may not be set because the user added the parameter
                    # later. Ignore.
                    pass

        x = np.column_stack([self[param][use] for param in params])
        y = self['log_posterior' if posterior else 'log_likelihood'][use]

        # Remove duplicates by only choosing the one with the highest
        # likelihood/posterior. First, sort by decreasing likelihood/posterior.
        idx = np.argsort(-y)
        x = x[idx]
        y = y[idx]
        # np.unique will return the first occurrence, i.e., the one with the
        # higher likelihood/posterior.
        idx = np.unique(x, return_index=True, axis=0)[1]
        x = x[idx]
        y = y[idx]

        x_grid, y = _sort_into_grid(x, y)

        if len(x_grid) == 1:
            return CubicSpline(x_grid[0], y, extrapolate=False)

        return RegularGridInterpolator(x_grid, y, method='cubic')

    def interval(self, param, threshold, posterior=True):
        """Get interval where likelihood/posterior is above a threshold.

        Parameters
        ----------
        param : str
            Paramater for which to get the interval.
        threshold : float
            Threshold such that the likelihood/posterior is at least
            its maximum plus the threshold. Must be negative.
        posterior : bool, optional
            If ``True``, compute the intervals for the (log) posterior. If
            ``False``, the intervals for the (log) likelihood are returned.
            Default is ``True``.

        Returns
        -------
        bounds : list
            List of pairs of lower and upper bound. For unimodal likelihood,
            this should typically be a single pair. If a lower and/or upper
            bound cannot be determined inside the range sampled, the value
            will be ``np.nan``.

        Raises
        ------
        ValueError
            If ``threshold`` is not negative.
        RuntimeError
            If the likelihood/posterior is identical to the maximum plus the
            threshold over some range instead of specific points.

        """
        if not threshold < 0:
            msg = "`threshold` must negative."
            raise ValueError(msg)

        interp = self.profile_interpolator(param, posterior=posterior)

        # Find the maximum by setting the derivative to 0.
        x = interp.derivative().roots()
        y_max = np.amax(interp(x))

        roots = np.sort(interp.solve(y_max + threshold))
        if np.any(np.isnan(roots)):
            msg = ("The likelihood/posterior is identical to the maximum plus "
                   "the threshold over some range.")
            raise RuntimeError(msg)

        y_der = interp.derivative()(roots)
        if y_der[0] < 0:
            roots = np.insert(roots, 0, np.nan)
        if y_der[-1] > 0:
            roots = np.append(roots, np.nan)

        return [tuple(r) for r in np.split(roots, len(roots) // 2)]
