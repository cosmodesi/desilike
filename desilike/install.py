import os
import sys
import logging

from .io import BaseConfig
from .utils import BaseClass
from . import utils


logger = logging.getLogger('Install')


class InstallError(Exception):

    pass


def download(url, target):
    logger.info('Downloading {} to {}.'.format(url, target))
    import requests
    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    utils.mkdir(os.path.dirname(target))
    with open(target, 'wb') as file:
        file.write(r.content)


def extract(in_fn, out_fn, remove=True):
    in_fn, out_fn = (os.path.normpath(fn) for fn in [in_fn, out_fn])
    ext = os.path.splitext(in_fn)[-1][1:]
    if ext == 'zip':
        from zipfile import ZipFile
        with ZipFile(in_fn, 'r') as zip:
            zip.extractall(out_fn)
    else:
        import tarfile
        if ext == 'tgz': ext = 'gz'
        with tarfile.open(in_fn, 'r:' + ext) as tar:
            tar.extractall(out_fn)
    if remove and out_fn != in_fn:
        os.remove(in_fn)


def exists_package(pkgname):
    try:
        pkg = __import__(pkgname)
    except ImportError:
        return False
    logger.info('Requirement already satisfied: {} in {}'.format(pkgname, os.path.dirname(os.path.dirname(pkg.__file__))))
    del pkg
    return True


def exists_path(path):
    return os.path.exists(path)


def pip(pkgindex, pkgname=None, install_dir=None, no_deps=False, force_reinstall=False, ignore_installed=False):
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
        command.append('--force_reinstall')
    if ignore_installed:
        command.append('--ignore_installed')
    command = ' '.join(command)
    logger.info(command)
    from subprocess import Popen, PIPE
    result = Popen(command, universal_newlines=True, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    logger.info(out)
    if len(err):
        #
        # Pass STDERR messages to the user, but do not
        # raise an error unless the return code was non-zero.
        #
        if proc.returncode == 0:
            message = ('pip emitted messages on STDERR; these can probably be ignored:\n' + err)
            logger.warning(message)
        else:
            raise InstallError('potentially serious error detected during pip installation:\n' + err)


def _insert_first(li, first):
    try:
        li.remove(first)
    except ValueError:
        pass
    li.insert(0, first)
    return li


def source(fn):
    import subprocess
    result = subprocess.run(['bash', '-c', 'source {} && env'.format(fn)], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        try:
            key, value = line.split('=')
            if key == 'PYTHONPATH':
                for path in value.split(':'): _insert_first(sys.path, path)
            os.environ[key] = value
        except ValueError:
            pass


class Installer(BaseClass):

    home_dir = os.path.expanduser('~')
    config_fn = os.path.join(home_dir, '.desilike', 'config.yaml')
    profile_fn = os.path.join(home_dir, '.desilike', 'profile.sh')

    def __init__(self, install_dir=None, user=False, no_deps=False, force_reinstall=False, ignore_installed=False, **kwargs):
        import site
        if user:
            if install_dir is not None:
                raise ValueError('Cannot provide both user and install_dir')
            install_dir = os.getenv('PYTHONUSERBASE', site.getuserbase())
        default_install_dir = os.path.dirname(os.path.dirname(os.path.dirname(site.getsitepackages()[0])))
        lib_rel_install_dir = os.path.relpath(site.getsitepackages()[0], default_install_dir)
        if install_dir is not None:
            install_dir = str(install_dir)
        config = BaseConfig(self.config_fn if os.path.isfile(self.config_fn) else {})
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
        default = {'pylib_dir': os.path.join(self.install_dir, lib_rel_install_dir),
                   'bin_dir': os.path.join(self.install_dir, 'bin'),
                   'include_dir': os.path.join(self.install_dir, 'include'),
                   'dylib_dir': os.path.join(self.install_dir, 'lib')}
        for name, value in default.items():
            setattr(self, name, kwargs.get(name, value))

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

    def __contains__(self, name):
        return name in self.config

    def __getitem__(self, name):
        return self.get(name)

    @staticmethod
    def parser(parser=None):
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--install-dir', type=str, default=None, help='Installation directory')
        parser.add_argument('--user', action='store_true', default=False, help='Install to the Python user install directory for your platform. Typically ~/.local/')
        parser.add_argument('--no-deps', action='store_true', default=False, help='Do not install package dependencies.')
        parser.add_argument('--force-reinstall', action='store_true', default=False, help='Reinstall all packages even if they are already up-to-date.')
        parser.add_argument('--ignore-installed', action='store_true', default=False, help='Ignore the installed packages, overwriting them. '
                            'This can break your system if the existing package is of a different version or was installed with a different package manager!')
        return parser

    @classmethod
    def from_args(cls, args, **kwargs):
        data = dict(user=args.user, no_deps=args.no_deps, force_reinstall=args.force_reinstall, ignore_installed=args.ignore_installed)
        if args.install_dir is not None:
            data['install_dir'] = args.install_dir
        data.update(kwargs)
        return cls(**data)

    def __call__(self, obj):

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

    def pip(self, pkgindex, **kwargs):
        kwargs = {**dict(no_deps=self.no_deps, force_reinstall=self.force_reinstall, ignore_installed=self.ignore_installed), **kwargs}
        pip(pkgindex, install_dir=self.install_dir, **kwargs)
        self.write({name: getattr(self, name) for name in ['pylib_dir', 'bin_dir']})

    def data_dir(self, section=None):
        base_dir = os.path.join(self.install_dir, 'data')
        if section is None:
            return base_dir
        return os.path.join(base_dir, section)

    def write(self, config, update=True):

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
            for key in dirs + ['source']:
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

    @classmethod
    def setenv(cls):
        if os.path.isfile(cls.profile_fn):
            source(cls.profile_fn)
