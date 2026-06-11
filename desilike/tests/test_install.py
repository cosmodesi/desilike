"""Tests for desilike/install.py"""

import os
import sys
import gzip
import tarfile
import zipfile

import yaml
import pytest

from desilike import install
from desilike.install import (InstallError, download, extract, exists_package, exists_path,
                              pip, _insert_first, source, Installer)


# ── helpers / fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point desilike's config and install directories to a temporary location."""
    config_dir = tmp_path / 'config'
    install_dir = tmp_path / 'install'
    monkeypatch.setenv('DESILIKE_CONFIG_DIR', str(config_dir))
    monkeypatch.setenv('DESILIKE_INSTALL_DIR', str(install_dir))
    monkeypatch.delenv('PYTHONUSERBASE', raising=False)
    return config_dir, install_dir


# ── module-level helpers ──────────────────────────────────────────────────────

class TestInsertFirst:

    def test_insert_new(self):
        assert _insert_first(['a', 'b'], 'c') == ['c', 'a', 'b']

    def test_move_existing_to_front(self):
        assert _insert_first(['a', 'b', 'c'], 'c') == ['c', 'a', 'b']

    def test_removes_all_duplicates(self):
        assert _insert_first(['c', 'a', 'c', 'b'], 'c') == ['c', 'a', 'b']

    def test_mutates_in_place(self):
        li = ['a', 'b']
        out = _insert_first(li, 'a')
        assert out is li
        assert li == ['a', 'b']


class TestExistsPackage:

    def test_existing_package(self):
        assert exists_package('os') is True

    def test_missing_package(self):
        assert exists_package('this_package_does_not_exist_xyz') is False


class TestExistsPath:

    def test_existing_path(self, tmp_path):
        assert exists_path(str(tmp_path)) is True

    def test_missing_path(self, tmp_path):
        assert exists_path(str(tmp_path / 'nope')) is False


# ── extract ────────────────────────────────────────────────────────────────

class TestExtract:

    def test_gzip(self, tmp_path):
        content = b'hello gzip'
        in_fn = tmp_path / 'data.gz'
        with gzip.open(in_fn, 'wb') as file:
            file.write(content)
        out_fn = tmp_path / 'data.txt'
        extract(str(in_fn), str(out_fn))
        assert out_fn.read_bytes() == content
        assert not in_fn.exists()  # removed by default

    def test_gzip_keep_input(self, tmp_path):
        in_fn = tmp_path / 'data.gz'
        with gzip.open(in_fn, 'wb') as file:
            file.write(b'keep me')
        out_fn = tmp_path / 'data.txt'
        extract(str(in_fn), str(out_fn), remove=False)
        assert in_fn.exists()

    def test_zip(self, tmp_path):
        in_fn = tmp_path / 'archive.zip'
        with zipfile.ZipFile(in_fn, 'w') as zip:
            zip.writestr('inner.txt', 'zipped')
        out_dir = tmp_path / 'out'
        extract(str(in_fn), str(out_dir))
        assert (out_dir / 'inner.txt').read_text() == 'zipped'
        assert not in_fn.exists()

    def test_tar_gz(self, tmp_path):
        member = tmp_path / 'member.txt'
        member.write_text('tarred')
        in_fn = tmp_path / 'archive.tar.gz'
        with tarfile.open(in_fn, 'w:gz') as tar:
            tar.add(str(member), arcname='member.txt')
        out_dir = tmp_path / 'out'
        extract(str(in_fn), str(out_dir))
        assert (out_dir / 'member.txt').read_text() == 'tarred'
        assert not in_fn.exists()

    def test_tgz(self, tmp_path):
        member = tmp_path / 'member.txt'
        member.write_text('tgz content')
        in_fn = tmp_path / 'archive.tgz'
        with tarfile.open(in_fn, 'w:gz') as tar:
            tar.add(str(member), arcname='member.txt')
        out_dir = tmp_path / 'out'
        extract(str(in_fn), str(out_dir))
        assert (out_dir / 'member.txt').read_text() == 'tgz content'


# ── download (requests mocked) ───────────────────────────────────────────────

class _FakeResponse:

    def __init__(self, content=b'', headers=None):
        self.content = content
        self.headers = headers or {}

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]


class TestDownload:

    def test_download_with_explicit_size(self, tmp_path, monkeypatch):
        content = b'x' * 4096
        import requests
        monkeypatch.setattr(requests, 'get', lambda *a, **kw: _FakeResponse(content))
        # head must not be called when size is provided
        monkeypatch.setattr(requests, 'head', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('head called')))
        target = tmp_path / 'sub' / 'file.bin'
        download('http://example.com/file', str(target), size=len(content))
        assert target.read_bytes() == content  # parent dir auto-created

    def test_download_size_from_header(self, tmp_path, monkeypatch):
        content = b'abcdef'
        import requests
        monkeypatch.setattr(requests, 'head', lambda *a, **kw: _FakeResponse(headers={'content-length': str(len(content))}))
        monkeypatch.setattr(requests, 'get', lambda *a, **kw: _FakeResponse(content))
        target = tmp_path / 'file.bin'
        download('http://example.com/file', str(target))
        assert target.read_bytes() == content

    def test_download_no_content_length(self, tmp_path, monkeypatch):
        content = b'no length here'
        import requests
        monkeypatch.setattr(requests, 'head', lambda *a, **kw: _FakeResponse(headers={}))
        monkeypatch.setattr(requests, 'get', lambda *a, **kw: _FakeResponse(content))
        target = tmp_path / 'file.bin'
        download('http://example.com/file', str(target))
        assert target.read_bytes() == content


# ── pip (subprocess mocked) ──────────────────────────────────────────────────

class _FakeProc:

    def __init__(self, out='', err='', returncode=0):
        self._out, self._err, self.returncode = out, err, returncode

    def communicate(self):
        return self._out, self._err


def _patch_popen(monkeypatch, proc, captured):
    def fake_popen(command, **kwargs):
        captured['command'] = command
        captured['kwargs'] = kwargs
        return proc
    monkeypatch.setattr(install, 'Popen', fake_popen, raising=False)
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', fake_popen)


class TestPip:

    def test_skips_if_already_installed(self, monkeypatch):
        called = {'popen': False}
        monkeypatch.setattr(install, 'exists_package', lambda name: True)
        import subprocess
        monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: called.__setitem__('popen', True))
        pip('os')
        assert called['popen'] is False

    def test_builds_basic_command(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(install, 'exists_package', lambda name: False)
        _patch_popen(monkeypatch, _FakeProc(out='done'), captured)
        pip('somepkg')
        assert 'pip install somepkg' in captured['command']
        assert '--disable-pip-version-check' in captured['command']

    def test_command_flags(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(install, 'exists_package', lambda name: False)
        _patch_popen(monkeypatch, _FakeProc(), captured)
        pip('pkg', install_dir='/opt/x', no_deps=True, force_reinstall=True, ignore_installed=True)
        command = captured['command']
        assert 'PYTHONUSERBASE=/opt/x' in command
        assert '--user' in command
        assert '--no-deps' in command
        assert '--force-reinstall' in command
        assert '--ignore-installed' in command

    def test_force_reinstall_skips_exists_check(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(install, 'exists_package', lambda name: (_ for _ in ()).throw(AssertionError('should not check')))
        _patch_popen(monkeypatch, _FakeProc(), captured)
        pip('pkg', force_reinstall=True)
        assert 'command' in captured

    def test_github_url_package_name(self, monkeypatch):
        checked = {}
        monkeypatch.setattr(install, 'exists_package', lambda name: checked.setdefault('name', name) or True)
        pip('git+https://github.com/cosmodesi/desilike')
        assert checked['name'] == 'desilike'

    def test_stderr_with_zero_returncode_warns(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(install, 'exists_package', lambda name: False)
        _patch_popen(monkeypatch, _FakeProc(err='some warning', returncode=0), captured)
        pip('pkg')  # should not raise

    def test_stderr_with_nonzero_returncode_raises(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(install, 'exists_package', lambda name: False)
        _patch_popen(monkeypatch, _FakeProc(err='boom', returncode=1), captured)
        with pytest.raises(InstallError):
            pip('pkg')


# ── source ───────────────────────────────────────────────────────────────────

class TestSource:

    def test_sets_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv('DESILIKE_TEST_VAR', raising=False)
        fn = tmp_path / 'profile.sh'
        fn.write_text('export DESILIKE_TEST_VAR=hello\n')
        source(str(fn))
        assert os.environ['DESILIKE_TEST_VAR'] == 'hello'

    def test_prepends_pythonpath_to_sys_path(self, tmp_path, monkeypatch):
        extra = str(tmp_path / 'mylib')
        monkeypatch.setattr(sys, 'path', list(sys.path))
        fn = tmp_path / 'profile.sh'
        fn.write_text('export PYTHONPATH={}\n'.format(extra))
        source(str(fn))
        assert sys.path[0] == extra


# ── Installer ─────────────────────────────────────────────────────────────────

class TestInstallerInit:

    def test_install_dir_from_env(self, isolated_config):
        config_dir, install_dir = isolated_config
        installer = Installer()
        assert installer.install_dir == str(install_dir)
        assert installer.config_dir == str(config_dir)

    def test_explicit_install_dir_overrides(self, isolated_config):
        installer = Installer(install_dir='/custom/path')
        assert installer.install_dir == '/custom/path'

    def test_user_and_install_dir_conflict(self, isolated_config):
        with pytest.raises(ValueError):
            Installer(user=True, install_dir='/custom/path')

    def test_unknown_kwarg_raises(self, isolated_config):
        with pytest.raises(ValueError):
            Installer(unknown_option=1)

    def test_flags_stored_as_bool(self, isolated_config):
        installer = Installer(no_deps=1, force_reinstall=0, ignore_installed=1)
        assert installer.no_deps is True
        assert installer.force_reinstall is False
        assert installer.ignore_installed is True

    def test_derived_dirs(self, isolated_config):
        config_dir, install_dir = isolated_config
        installer = Installer()
        assert installer.bin_dir == os.path.join(str(install_dir), 'bin')
        assert installer.include_dir == os.path.join(str(install_dir), 'include')
        assert installer.dylib_dir == os.path.join(str(install_dir), 'lib')

    def test_dir_override_via_kwargs(self, isolated_config):
        installer = Installer(bin_dir='/my/bin')
        assert installer.bin_dir == '/my/bin'

    def test_config_fn_written_on_init(self, isolated_config):
        config_dir, install_dir = isolated_config
        installer = Installer()
        assert os.path.isfile(installer.config_fn)
        with open(installer.config_fn) as file:
            config = yaml.safe_load(file)
        assert config['install_dir'] == str(install_dir)


class TestInstallerConfigAccess:

    def test_paths(self, isolated_config):
        config_dir, _ = isolated_config
        installer = Installer()
        assert installer.config_fn == os.path.join(str(config_dir), 'config.yaml')
        assert installer.profile_fn == os.path.join(str(config_dir), 'profile.sh')

    def test_getitem_and_contains(self, isolated_config):
        installer = Installer()
        assert 'install_dir' in installer
        assert installer['install_dir'] == installer.install_dir
        assert 'missing_key' not in installer

    def test_getitem_missing_raises_keyerror(self, isolated_config):
        installer = Installer()
        with pytest.raises(KeyError):
            installer['missing_key']

    def test_get_default(self, isolated_config):
        installer = Installer()
        assert installer.get('missing_key', 'fallback') == 'fallback'

    def test_reinstall_property(self, isolated_config):
        assert Installer().reinstall is False
        assert Installer(force_reinstall=True).reinstall is True
        assert Installer(ignore_installed=True).reinstall is True


class TestInstallerLoadConfig:

    def test_from_dict(self):
        config = Installer._load_config({'a': 1})
        assert config == {'a': 1}

    def test_from_dict_is_copy(self):
        source_dict = {'a': 1}
        config = Installer._load_config(source_dict)
        assert config is not source_dict

    def test_from_missing_returns_empty(self):
        assert Installer._load_config(None) == {}
        assert Installer._load_config('/no/such/file.yaml') == {}

    def test_from_file(self, tmp_path):
        fn = tmp_path / 'config.yaml'
        fn.write_text(yaml.safe_dump({'install_dir': '/x', 'foo': 'bar'}))
        config = Installer._load_config(str(fn))
        assert config == {'install_dir': '/x', 'foo': 'bar'}

    def test_from_empty_file(self, tmp_path):
        fn = tmp_path / 'empty.yaml'
        fn.write_text('')
        assert Installer._load_config(str(fn)) == {}


class TestInstallerDataDir:

    def test_no_section_returns_base(self, isolated_config):
        _, install_dir = isolated_config
        installer = Installer()
        assert installer.data_dir() == os.path.join(str(install_dir), 'data')

    def test_section(self, isolated_config):
        _, install_dir = isolated_config
        installer = Installer()
        assert installer.data_dir('mysection') == os.path.join(str(install_dir), 'data', 'mysection')

    def test_section_from_config(self, isolated_config):
        installer = Installer()
        installer.config['mysection'] = {'data_dir': '/explicit/data'}
        assert installer.data_dir('mysection') == '/explicit/data'

    def test_ro_replacement(self, isolated_config):
        installer = Installer()
        installer.config['mysection'] = {'data_dir': '/install/path/data'}
        installer.config['ro'] = ['/install', '/readonly']
        assert installer.data_dir('mysection', ro=True) == '/readonly/path/data'

    def test_ro_false_keeps_path(self, isolated_config):
        installer = Installer()
        installer.config['mysection'] = {'data_dir': '/install/path/data'}
        installer.config['ro'] = ['/install', '/readonly']
        assert installer.data_dir('mysection', ro=False) == '/install/path/data'


class TestInstallerWrite:

    def test_write_creates_config_and_profile(self, isolated_config):
        installer = Installer()
        installer.write({'pylib_dir': '/lib/python'}, update=False)
        with open(installer.config_fn) as file:
            config = yaml.safe_load(file)
        assert config['pylib_dir'] == ['/lib/python']  # scalar coerced to list
        assert os.path.isfile(installer.profile_fn)

    def test_profile_exports_pythonpath(self, isolated_config):
        installer = Installer()
        installer.write({'pylib_dir': '/lib/python', 'bin_dir': '/bin/x'}, update=False)
        profile = open(installer.profile_fn).read()
        assert profile.startswith('#!/bin/bash\n')
        assert 'export PYTHONPATH=/lib/python:$PYTHONPATH' in profile
        assert 'export PATH=/bin/x:$PATH' in profile

    def test_write_update_prepends_new_paths(self, isolated_config):
        installer = Installer()
        installer.write({'pylib_dir': '/lib/a'}, update=False)
        installer.write({'pylib_dir': '/lib/b'}, update=True)
        with open(installer.config_fn) as file:
            config = yaml.safe_load(file)
        assert config['pylib_dir'] == ['/lib/b', '/lib/a']

    def test_write_update_dedups_paths(self, isolated_config):
        installer = Installer()
        installer.write({'pylib_dir': '/lib/a'}, update=False)
        installer.write({'pylib_dir': '/lib/a'}, update=True)
        with open(installer.config_fn) as file:
            config = yaml.safe_load(file)
        assert config['pylib_dir'] == ['/lib/a']


class TestInstallerSetenv:

    def test_setenv_sources_profile(self, isolated_config, monkeypatch):
        monkeypatch.delenv('DESILIKE_SETENV_VAR', raising=False)
        installer = Installer()
        with open(installer.profile_fn, 'w') as file:
            file.write('#!/bin/bash\nexport DESILIKE_SETENV_VAR=fromprofile\n')
        installer.setenv()
        assert os.environ['DESILIKE_SETENV_VAR'] == 'fromprofile'

    def test_setenv_no_profile_is_noop(self, isolated_config):
        installer = Installer()
        os.remove(installer.profile_fn)
        installer.setenv()  # should not raise


class TestInstallerCall:

    def test_install_classmethod_invoked(self, isolated_config):
        installer = Installer()
        calls = []

        class WithInstall:
            @classmethod
            def install(cls, inst):
                calls.append(inst)

        installer(WithInstall)
        assert calls == [installer]

    def test_install_instance_uses_type(self, isolated_config):
        installer = Installer()
        calls = []

        class WithInstall:
            @classmethod
            def install(cls, inst):
                calls.append(cls)

        installer(WithInstall())
        assert calls == [WithInstall]

    def test_no_install_method_is_noop(self, isolated_config):
        installer = Installer()

        class NoInstall:
            pass

        installer(NoInstall)  # should not raise


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
