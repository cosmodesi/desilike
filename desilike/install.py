import os
import sys
import logging

from pathlib import Path
import yaml


logger = logging.getLogger('Install')


class InstallError(Exception):
    """Error raised at installation."""


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
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    in_fn, out_fn = Path(in_fn), Path(out_fn)
    is_tar = in_fn.suffixes[-2:-1] == ['.tar']
    if in_fn.suffix == '.gz' and not is_tar:
        import gzip
        with open(out_fn, 'wb') as out, gzip.open(in_fn, 'r') as gz:
            out.write(gz.read())
    elif in_fn.suffix == '.zip' and not is_tar:
        from zipfile import ZipFile
        with ZipFile(in_fn, 'r') as zip:
            zip.extractall(out_fn)
    else:
        import tarfile
        ext = in_fn.suffix[1:]
        if ext == 'tgz': ext = 'gz'
        with tarfile.open(in_fn, 'r:' + ext) as tar:
            tar.extractall(out_fn)
    if remove and out_fn != in_fn:
        in_fn.unlink()


def exists_package(pkgname):
    """Check wether package with name ``pkgname`` can be imported."""
    try:
        pkg = __import__(pkgname)
    except ImportError:
        return False
    logger.info('Requirement already satisfied: {} in {}'.format(pkgname, Path(pkg.__file__).parent.parent))
    del pkg
    return True


def exists_path(path):
    """Check whether this ``path`` exists on disk."""
    return Path(path).exists()


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


class Installer(object):
    """
    Installer. desilike's configuration ('config.yaml' and 'profile.sh') is saved
    under 'DESILIKE_CONFIG_DIR' environment variable if defined, else '~/.desilike'.

    Given some calculator one would like to install, the installer is typically used as:

    >>> installer = Installer(user=True)
    >>> installer(calculator)

    To install a profiler (e.g. :class:`MinuitProfiler`):

    >>> installer(MinuitProfiler)
    """
    home_dir = str(Path.home())

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
            default_install_dir = str(Path(site.getsitepackages()[0]).parents[2])
        # os.path.relpath: Path has no lexical equivalent (relative_to does not walk up before 3.12)
        lib_rel_install_dir = os.path.relpath(site.getsitepackages()[0], default_install_dir)
        if install_dir is not None:
            install_dir = str(install_dir)

        self.config_dir = os.getenv('DESILIKE_CONFIG_DIR', '')
        default_config_dir = str(Path(self.home_dir) / '.desilike')
        if not self.config_dir:
            self.config_dir = default_config_dir

        config_source = {}
        if Path(self.config_fn).is_file():
            config_source = self.config_fn
            try:
                with open(self.config_fn, 'a'): pass
            except PermissionError:  # from now on, write to home
                self.config_dir = default_config_dir
        config = self._load_config(config_source)

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
        install_dir_path = Path(self.install_dir)
        # os.path.normpath: collapses any '..' from lib_rel_install_dir lexically (Path keeps them)
        default = {'pylib_dir': os.path.normpath(install_dir_path / lib_rel_install_dir),
                   'bin_dir': str(install_dir_path / 'bin'),
                   'include_dir': str(install_dir_path / 'include'),
                   'dylib_dir': str(install_dir_path / 'lib')}
        for name, value in default.items():
            setattr(self, name, kwargs.pop(name, value))
        if kwargs:
            raise ValueError('Did not understand {}'.format(kwargs))

    @staticmethod
    def _load_config(source):
        """Load configuration from a dict or a .yaml file path; return a plain dict."""
        if isinstance(source, dict):
            return dict(source)
        if source and Path(source).is_file():
            with open(source, 'r') as file:
                return yaml.safe_load(file) or {}
        return {}

    @property
    def config_fn(self):
        """Path to .yaml configuration file."""
        return str(Path(self.config_dir) / 'config.yaml')

    @property
    def profile_fn(self):
        """Path to .sh profile to be sourced to set all paths."""
        return str(Path(self.config_dir) / 'profile.sh')

    def log_info(self, msg, *args, **kwargs):
        logger.info(msg, *args, **kwargs)

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

        - a calculator instance (all calculators in its dependency tree are installed)
        - a Profiler / Sampler class

        More generally, whatever exposes an :meth:`install` classmethod.
        """
        self.log_info('Installation directory is {}.'.format(self.install_dir))

        def install(cls):
            func = getattr(cls, 'install', None)
            if func is None:
                return
            func(self)
            self.setenv()

        from .base import Calculator, _iter_calculators
        if isinstance(obj, Calculator):
            for calculator in _iter_calculators(obj):
                install(type(calculator))
        else:
            install(obj if isinstance(obj, type) else type(obj))

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

    def data_dir(self, section=None, ro=False):
        """
        Return path to data directory, where one will typically save / install
        specific calculator data or code.

        Parameters
        ----------
        section : str, default=None
            Section; typically this will be calculator's name.
        ro : bool, default=None
            Read-only?

        Returns
        -------
        data_dir : str
            Path to data directory.
        """
        base_dir = Path(self.install_dir) / 'data'
        if section is None:
            toret = str(base_dir)
        else:
            try:
                toret = self[section]['data_dir']
            except KeyError:
                toret = str(base_dir / section)
        if ro:
            ro = self.get('ro', None)
            if ro is not None:
                toret = toret.replace(*ro)
        return toret

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
            if not isinstance(li, (list, tuple, set, frozenset)): li = [li]
            return list(li)

        config = dict(config)
        dirs = ['pylib_dir', 'bin_dir', 'dylib_dir']
        for key in dirs + ['source']:
            if key in config: config[key] = _make_list(config[key])
        if update and Path(self.config_fn).is_file():
            base_config = self._load_config(self.config_fn)
            config = {**base_config, **config}
            for key in dirs + ['source']:
                paths = _make_list(config.get(key, []))
                config[key] = paths + [path for path in _make_list(base_config.get(key, [])) if path not in paths]
        Path(self.config_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_fn, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        Path(self.profile_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_fn, 'w') as file:
            file.write('#!/bin/bash\n')
            for key, keybash in zip(dirs, ['PYTHONPATH', 'PATH', 'LD_LIBRARY_PATH']):
                if key in config: file.write('export {}={}\n'.format(keybash, ':'.join(config[key] + [f'${keybash}'])))
            for src in config.get('source', []):
                file.write('source {}'.format(src))

    def setenv(self):
        """Set environment (i.e. set paths). Called in desilike's __init__.py."""
        if Path(self.profile_fn).is_file():
            source(self.profile_fn)
