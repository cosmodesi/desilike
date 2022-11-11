import os
import sys
import requests
import logging
import argparse

from .io import BaseConfig, ConfigError
from . import utils

logger = logging.getLogger('Install')


class InstallationError(Exception):

    pass


def download(url, target):
    logger.info('Downloading {} to {}.'.format(url, target))
    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    utils.mkdir(os.path.dirname(target))
    with open(target, 'wb') as file:
        file.write(r.content)


def pip(package, install_dir=None, no_deps=False, force_reinstall=False, ignore_installed=False):
    command = [sys.executable, '-m', 'pip', 'install', package, '--disable-pip-version-check']
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
    proc = Popen(command, universal_newlines=True, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    logger.info(out)
    if len(err):
        #
        # Pass STDERR messages to the user, but do not
        # raise an error unless the return code was non-zero.
        #
        if proc.returncode == 0:
            message = ('Pip emitted messages on STDERR; these can probably be ignored:\n' + err)
            logger.warning(message)
        else:
            raise InstallationError('Potentially serious error detected during pip installation:\n' + err)


class InstallerConfig(BaseConfig):

    home_dir = os.path.expanduser('~')
    config_fn = os.path.join(home_dir, '.desilike', 'config.yaml')
    profile_fn = os.path.join(home_dir, '.desilike', 'profile.sh')

    def __init__(self, *args, **kwargs):
        super(InstallerConfig, self).__init__(*args, **kwargs)
        import site
        if self.pop('user', False):
            if 'install_dir' in self:
                raise ConfigError('Cannot provide both user and install_dir')
            self['install_dir'] = os.getenv('PYTHONUSERBASE', site.getuserbase())
        default_install_dir = os.path.dirname(os.path.dirname(os.path.dirname(site.getsitepackages()[0])))
        lib_rel_install_dir = os.path.relpath(site.getsitepackages()[0], default_install_dir)
        config = BaseConfig(self.__class__.config_fn if os.path.isfile(self.__class__.config_fn) else {})
        if 'install_dir' not in config:
            config['install_dir'] = default_install_dir
            if 'install_dir' in self:
                config['install_dir'] = self['install_dir']
            self.write({'install_dir': config['install_dir']})
        if 'install_dir' not in self:
            self['install_dir'] = config['install_dir']
        default = {'no_deps': False, 'force_reinstall': False, 'ignore_installed': False,
                   'lib_dir': os.path.join(self['install_dir'], lib_rel_install_dir),
                   'bin_dir': os.path.join(self['install_dir'], 'bin'),
                   'data_dir': os.path.join(self['install_dir'], 'data')}
        for name, value in default.items():
            self.setdefault(name, value)
        for name, value in config.items():
            self.setdefault(name, value)

    def pip(self, package):
        pip(package, install_dir=self['install_dir'], no_deps=self['no_deps'], force_reinstall=self['force_reinstall'], ignore_installed=self['ignore_installed'])
        self.write(self.select(['lib_dir', 'bin_dir']))

    def data_dir(self, section=None):
        base_dir = os.path.join(self['install_dir'], 'data')
        if section is None:
            return base_dir
        return os.path.join(base_dir, section)

    @staticmethod
    def parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--install-dir', type=str, default=None, help='Installation directory')
        parser.add_argument('--user', action='store_true', default=False, help='Install to the Python user install directory for your platform. Typically ~/.local/')
        parser.add_argument('--no-deps', action='store_true', default=False, help='Don’t install package dependencies.')
        parser.add_argument('--force-reinstall', action='store_true', default=False, help='Reinstall all packages even if they are already up-to-date.')
        parser.add_argument('--ignore-installed', action='store_true', default=False, help='Ignore the installed packages, overwriting them. '
                            'This can break your system if the existing package is of a different version or was installed with a different package manager!')
        return parser

    @classmethod
    def from_args(cls, args):
        data = dict(user=args.user, no_deps=args.no_deps, force_reinstall=args.force_reinstall, ignore_installed=args.ignore_installed)
        if args.install_dir is not None:
            data['install_dir'] = args.install_dir
        return cls(data)

    def write(self, config, update=True):
        config = BaseConfig(config).copy()
        if update and os.path.isfile(self.config_fn):
            base_config = BaseConfig(self.config_fn)
            config = base_config.clone(config)
            for key in ['lib_dir', 'bin_dir']:
                paths = config.get(key, [])
                if not utils.is_sequence(paths): paths = [paths]
                config[key] = paths + [path for path in base_config.get(key, []) if path not in paths]
        config.write(self.config_fn)
        utils.mkdir(os.path.dirname(self.profile_fn))
        with open(self.profile_fn, 'w') as file:
            file.write('#!/bin/bash\n')
            for key, keybash in zip(['lib_dir', 'bin_dir'], ['PYTHONPATH', 'PATH']):
                if key in config:
                    file.write('export {}={}\n'.format(keybash, ':'.join(config[key] + [f'${keybash}'])))

    @classmethod
    def setenv(cls):
        if os.path.isfile(cls.config_fn):
            config = BaseConfig(cls.config_fn)
            paths = config.get('lib_dir', [])
            for path in paths:
                sys.path.insert(0, path)
