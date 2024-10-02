import os
import sys
import logging

from .io import BaseConfig
from .utils import BaseClass
from . import utils


logger = logging.getLogger('Install')


class InstallError(Exception):

    pass


def download(url, target, size=None):
    """
    Download file from input ``url``.

    Parameters
    ----------
    url : str, Path
        url to download file from.

    target : str, Path
        Path where to save the file, on disk.

    size : int, default=None
        Expected file size, in bytes, used to show progression bar.
        If not provided, taken from header (if the file is larger than a couple of GBs,
        it may be wrong due to integer overflow).
        If a sensible file size is obtained, a progression bar is printed.
    """
    # Adapted from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    logger.info('Downloading {} to {}.'.format(url, target))
    import requests
    utils.mkdir(os.path.dirname(target))
    # See https://stackoverflow.com/questions/61991164/python-requests-missing-content-length-response
    if size is None:
        size = requests.head(url, headers={'Accept-Encoding': None}).headers.get('content-length')
    r = requests.get(url, allow_redirects=True, stream=True)

    with open(target, 'wb') as file:
        if size is None or int(size) < 0:  # no content length header
            file.write(r.content)
        else:
            import shutil
            width = shutil.get_terminal_size((80, 20))[0] - 9  # pass fallback
            dl, size, current = 0, int(size), 0
            for data in r.iter_content(chunk_size=2048):
                dl += len(data)
                file.write(data)
                if size:
                    frac = min(dl / size, 1.)
                    done = int(width * frac)
                    if done > current:  # it seems, when content-length is not set iter_content does not care about chunk_size
                        print('\r[{}{}] [{:3.0%}]'.format('#' * done, ' ' * (width - done), frac), end='', flush=True)
                        current = done
            print('')


def extract(in_fn, out_fn, remove=True):
    """
    Extract ``in_fn`` to ``out_fn``.

    Parameters
    ----------
    in_fn : str, Path
        Path to input, compressed, filename.

    out_fn : str, Path
        Path to output file / directory.

    remove : bool, default=True
        If ``True``, remove input file ``in_fn``.
    """
    in_fn, out_fn = (os.path.normpath(fn) for fn in [in_fn, out_fn])
    if any(in_fn.endswith(ext) and not in_fn.endswith('tar' + ext) for ext in ['.gz']):
        import gzip
        with open(out_fn, 'wb') as out:
            with gzip.open(in_fn, 'r') as gz:
                out.write(gz.read())
    elif any(in_fn.endswith(ext) and not in_fn.endswith('tar' + ext) for ext in ['.zip']):
        from zipfile import ZipFile
        with ZipFile(in_fn, 'r') as zip:
            zip.extractall(out_fn)
    else:
        import tarfile
        ext = os.path.splitext(in_fn)[-1][1:]
        if ext == 'tgz': ext = 'gz'
        with tarfile.open(in_fn, 'r:' + ext) as tar:
            tar.extractall(out_fn)
    if remove and out_fn != in_fn:
        os.remove(in_fn)


def exists_package(pkgname):
    """Check wether package with name ``pkgname`` can be imported."""
    try:
        pkg = __import__(pkgname)
    except ImportError:
        return False
    logger.info('Requirement already satisfied: {} in {}'.format(pkgname, os.path.dirname(os.path.dirname(pkg.__file__))))
    del pkg
    return True


def exists_path(path):
    """Check whether this ``path`` exists on disk."""
    return os.path.exists(path)


def pip(pkgindex, pkgname=None, install_dir=None, no_deps=False, force_reinstall=False, ignore_installed=False):
    """
    Install with PIP.

    Parameter
    ---------
    pkgindex : str
        Where to find the package.
        A package name (if registered on pypi), or a url, if on github;
        e.g. git+https://github.com/cosmodesi/desilike.

    pkgname : str, default=None
        Package name, to check whether the package is already installed.
        If ``None``, defaults to ``pkgindex``, or the end of ``pkgindex``,
        if 'https://' is found in it.

    install_dir : str, Path, default=None
        Installation directory. Defaults to PIP's default.

    no_deps : bool, default=False
        Does not install package's dependencies.

    force_reinstall : bool, default=False
        Force package's installation.

    ignore_installed : bool, default=False
        Ignore all (including e.g. package dependencies) previously installed packages.
    """
    if not force_reinstall:
        # Check if package already installed (to cope with git-provided package)
        if pkgname is None:
            if 'https://' in pkgindex:
                for pkgname in pkgindex.split('#')[0].split('/')[::-1]:
                    if pkgname: break
            else:
                pkgname = pkgindex
        if exists_package(pkgname): return
    command = [sys.executable, '-m', 'pip', 'install', pkgindex, '--disable-pip-version-check']
    if install_dir is not None:
        command = ['PYTHONUSERBASE={}'.format(install_dir)] + command + ['--user']
    if no_deps:
        command.append('--no-deps')
    if force_reinstall:
        command.append('--force-reinstall')
    if ignore_installed:
        command.append('--ignore-installed')
    command = ' '.join(command)
    logger.info(command)
    from subprocess import Popen, PIPE
    proc = Popen(command, universal_newlines=True, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    logger.info(out)
    if len(err):
        # Pass STDERR messages to the user, but do not
        # raise an error unless the return code was non-zero.
        if proc.returncode == 0:
            message = ('pip emitted messages on STDERR; these can probably be ignored:\n' + err)
            logger.warning(message)
        else:
            raise InstallError('potentially serious error detected during pip installation:\n' + err)


def _insert_first(li, el):
    # Remove element el from list li if exists,
    # then add it at the start of li
    while True:
        try:
            li.remove(el)
        except ValueError:
            break
    li.insert(0, el)
    return li


def source(fn):
    """Source input file ``fn`` and set associated environment variables."""
    import subprocess
    result = subprocess.run(['bash', '-c', 'source {} && env'.format(fn)], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        try:
            key, value = line.split('=')
            if key == 'PYTHONPATH':
                for path in value.split(':')[::-1]: _insert_first(sys.path, path)
            else:
                os.environ[key] = value
        except ValueError:
            pass


class Installer(BaseClass):
    """
    Installer. desilike's configuration ('config.yaml' and 'profile.sh') is saved
    under 'DESILIKE_CONFIG_DIR' environment variable if defined, else '~/.desilike'.

    Given some calculator one would like to install, the installer is typically used as:

    >>> installer = Installer(user=True)
    >>> installer(calculator)

    To install a profiler (e.g. :class:`MinuitProfiler`), a sampler (e.g. :class:`EmceeSampler`),
    or an emulator (e.g. :class:`MLPEmulatorEngine`):

    >>> installer(MinuitProfiler)
    >>> installer(EmceeSampler)
    >>> installer(MLPEmulatorEngine)
    """
    home_dir = os.path.expanduser('~')

    def __init__(self, install_dir=None, user=False, no_deps=False, force_reinstall=False, ignore_installed=False, **kwargs):
        """
        Initialize installer.

        Parameters
        ----------
        install_dir : str, Path, default=None
            Installation directory. Defaults to directory in :attr:`config_fn` if provided,
            else 'DESILIKE_INSTALL_DIR' environment variable if defined, else PIP's default.

        user : bool, default=False
            If ``True``, installation directory is home directory.

        no_deps : bool, default=False
            Does not install package's dependencies.

        force_reinstall : bool, default=False
            Force package's installation.

        ignore_installed : bool, default=False
            Ignore all (including e.g. package dependencies) previously installed packages.
        """
        import site
        if user:
            if install_dir is not None:
                raise ValueError('Cannot provide both user and install_dir')
            install_dir = os.getenv('PYTHONUSERBASE', site.getuserbase())
        default_install_dir = os.getenv('DESILIKE_INSTALL_DIR', '')
        if not default_install_dir:
            default_install_dir = os.path.dirname(os.path.dirname(os.path.dirname(site.getsitepackages()[0])))
        lib_rel_install_dir = os.path.relpath(site.getsitepackages()[0], default_install_dir)
        if install_dir is not None:
            install_dir = str(install_dir)

        self.config_dir = os.getenv('DESILIKE_CONFIG_DIR', '')
        default_config_dir = os.path.join(self.home_dir, '.desilike')
        if not self.config_dir:
            self.config_dir = default_config_dir

        config_fn = {}
        if os.path.isfile(self.config_fn):
            config_fn = self.config_fn
            try:
                with open(self.config_fn, 'a'): pass
            except PermissionError:  # from now on, write to home
                self.config_dir = default_config_dir
        config = BaseConfig(config_fn)

        if 'install_dir' not in config:
            config['install_dir'] = default_install_dir
            if install_dir is not None:
                config['install_dir'] = install_dir
            self.write({'install_dir': config['install_dir']})
        if install_dir is None:
            install_dir = config['install_dir']
        self.config = config
        self.install_dir = install_dir
        self.no_deps = bool(no_deps)
        self.force_reinstall = bool(force_reinstall)
        self.ignore_installed = bool(ignore_installed)
        default = {'pylib_dir': os.path.normpath(os.path.join(self.install_dir, lib_rel_install_dir)),
                   'bin_dir': os.path.join(self.install_dir, 'bin'),
                   'include_dir': os.path.join(self.install_dir, 'include'),
                   'dylib_dir': os.path.join(self.install_dir, 'lib')}
        for name, value in default.items():
            setattr(self, name, kwargs.pop(name, value))
        if kwargs:
            raise ValueError('Did not understand {}'.format(kwargs))

    @property
    def config_fn(self):
        """Path to .yaml configuration file."""
        return os.path.join(self.config_dir, 'config.yaml')

    @property
    def profile_fn(self):
        """Path to .sh profile to be sourced to set all paths."""
        return os.path.join(self.config_dir, 'profile.sh')

    def get(self, *args, **kwargs):
        """Get config option, e.g. ``install_dir``."""
        return self.config.get(*args, **kwargs)

    def __contains__(self, name):
        return name in self.config

    def __getitem__(self, name):
        """Get config option, e.g. ``install_dir``."""
        try:
            return self.config[name]
        except KeyError as exc:
            raise KeyError('Config option {} does not exist in config {}; maybe the corresponding calculator should be installed?'.format(name, self.config_fn)) from exc

    def __call__(self, obj):
        """
        Install input object ``obj``, which can be:

        - a calculator instance
        - a Sampler, Profiler, Emulator class

        More generally, whatever has an :meth:`install` method.
        """
        self.log_info('Installation directory is {}.'.format(self.install_dir))

        def install(obj):
            try:
                func = obj.install
            except AttributeError:
                return
            func(self)

        from .base import BaseCalculator
        if isinstance(obj, BaseCalculator):
            from .base import RuntimeInfo
            installer_bak = RuntimeInfo.installer
            RuntimeInfo.installer = self
            obj.runtime_info.pipeline
            RuntimeInfo.installer = installer_bak
        else:
            install(obj)

    @property
    def reinstall(self):
        return self.force_reinstall or self.ignore_installed

    def pip(self, pkgindex, **kwargs):
        """
        Install Python package with PIP.

        Parameters
        ----------
        pkgindex : str
            Where to find the package.
            A package name (if registered on pypi), or a url, if on github;
            e.g. git+https://github.com/cosmodesi/desilike.

        **kwargs : dict
            Optionally, one can provide ``no_deps``, ``force_reinstall``, ``ignore_installed``
            to override :class:`Installer` attributes.
        """
        kwargs = {**dict(no_deps=self.no_deps, force_reinstall=self.force_reinstall, ignore_installed=self.ignore_installed), **kwargs}
        pip(pkgindex, install_dir=self.install_dir, **kwargs)
        self.write({name: getattr(self, name) for name in ['pylib_dir', 'bin_dir']})

    def data_dir(self, section=None):
        """
        Return path to data directory, where one will typically save / install
        specific calculator data or code.

        Parameters
        ----------
        section : str, default=None
            Section; typically this will be calculator's name.

        Returns
        -------
        data_dir : str
            Path to data directory.
        """
        base_dir = os.path.join(self.install_dir, 'data')
        if section is None:
            return base_dir
        return os.path.join(base_dir, section)

    def write(self, config, update=True):
        """
        Write configuration to :attr:`config_fn`.

        Parameters
        ----------
        config : dict
            Configuration.

        update : bool, default=True
            If ``True``, insert new 'pylib_dir', 'bin_dir', 'dylib_dir', 'source' entries
            on top of previous ones.
            If ``False``, such entries are overriden.
        """
        def _make_list(li):
            if not utils.is_sequence(li): li = [li]
            return list(li)

        config = BaseConfig(config).copy()
        dirs = ['pylib_dir', 'bin_dir', 'dylib_dir']
        for key in dirs + ['source']:
            if key in config: config[key] = _make_list(config[key])
        if update and os.path.isfile(self.config_fn):
            base_config = BaseConfig(self.config_fn)
            config = base_config.clone(config)
            for keynew in dirs + ['source']:
                paths = _make_list(config.get(key, []))
                config[key] = paths + [path for path in _make_list(base_config.get(key, [])) if path not in paths]
        config.write(self.config_fn)
        utils.mkdir(os.path.dirname(self.profile_fn))
        with open(self.profile_fn, 'w') as file:
            file.write('#!/bin/bash\n')
            for key, keybash in zip(dirs, ['PYTHONPATH', 'PATH', 'LD_LIBRARY_PATH']):
                if key in config: file.write('export {}={}\n'.format(keybash, ':'.join(config[key] + [f'${keybash}'])))
            for src in config.get('source', []):
                file.write('source {}'.format(src))

    def setenv(self):
        """Set environment (i.e. set paths). Called in desilike's __init__.py."""
        if os.path.isfile(self.profile_fn):
            source(self.profile_fn)
