"""Tests for desilike/samples/profiles.py."""

import copy
import pytest
import numpy as np

from desilike.parameter import Parameter, VariableCollection
from desilike.samples import Profiles


RNG = np.random.default_rng(42)

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_profiles(n_runs=3, n_params=4, n_scan=101, n_contour=21):
    """Build a Profiles with all slots filled, n_runs minimiser runs."""
    pnames = [f'p{i}' for i in range(n_params)]
    params = VariableCollection({
        n: Parameter(n, value=0.0, prior={'dist': 'norm', 'loc': 0., 'scale': 1.})
        for n in pnames
    })

    lp = -0.5 * RNG.chisquare(n_params, n_runs)

    start = {n: RNG.normal(0., 0.3, n_runs) for n in pnames}
    start['logpdf'] = lp - 1.

    best = {n: RNG.normal(0., 0.1, n_runs) for n in pnames}
    best['logpdf'] = lp

    error = {n: np.abs(RNG.normal(0.5, 0.05, n_runs)) for n in pnames}

    interval = {
        n: (RNG.normal(-0.5, 0.05, n_runs), RNG.normal(0.5, 0.05, n_runs))
        for n in pnames
    }

    x = np.linspace(-1., 1., n_scan)
    profile = {n: (x, -0.5 * x ** 2) for n in pnames}

    grid_x = np.linspace(-1., 1., 5)
    grid = {n: grid_x for n in pnames}
    grid['logpdf'] = -0.5 * grid_x ** 2

    t = np.linspace(0., 2. * np.pi, n_contour)
    params2 = [(pnames[i], pnames[j])
               for i in range(n_params) for j in range(i)]
    contour = {
        1: {pair: (np.cos(t), np.sin(t)) for pair in params2},
        2: {pair: (2. * np.cos(t), 2. * np.sin(t)) for pair in params2},
    }

    return Profiles(
        params=params,
        start=start,
        best=best,
        error=error,
        interval=interval,
        profile=profile,
        grid=grid,
        contour=contour,
        attrs={'ndof': 10},
    )


# ── construction ──────────────────────────────────────────────────────────────

def test_init_empty():
    p = Profiles()
    assert p.params is None
    assert p.best is None
    for name in ('start', 'error', 'interval', 'profile', 'grid', 'contour'):
        assert name not in p


def test_init_with_kwargs():
    p = _make_profiles(n_runs=2)
    assert p.params is not None
    assert 'best' in p
    assert p.nruns == 2


def test_set_unknown_slot_raises():
    p = Profiles()
    with pytest.raises(ValueError, match='Unknown Profiles slot'):
        p.set(unknown=42)


def test_contains():
    p = Profiles()
    assert 'best' not in p
    p.best = {'logpdf': np.array([-1.])}
    assert 'best' in p


def test_get_default():
    p = Profiles()
    # slots are set to None in __init__, so getattr always finds them
    assert p.get('best') is None
    # unknown attr respects default
    assert p.get('_nonexistent_', 'default') == 'default'


# ── ParameterDict indexing ────────────────────────────────────────────────────

def test_parameter_dict_str_key():
    p = _make_profiles(n_runs=2)
    arr = p.best['p0']
    assert arr.shape == (2,)


def test_parameter_dict_parameter_key():
    p = _make_profiles(n_runs=2)
    param = Parameter('p0', value=0.)
    arr = p.best[param]
    np.testing.assert_array_equal(arr, p.best['p0'])


def test_parameter_dict_contains():
    p = _make_profiles(n_runs=2)
    param = Parameter('p0', value=0.)
    assert param in p.best
    assert 'p0' in p.best
    assert 'missing' not in p.best


# ── per-run helpers ───────────────────────────────────────────────────────────

def test_nruns():
    p = _make_profiles(n_runs=5)
    assert p.nruns == 5


def test_argmax():
    p = _make_profiles(n_runs=4)
    expected = int(np.argmax(p.best['logpdf']))
    assert p.argmax == expected


def test_chi2min():
    p = _make_profiles(n_runs=4)
    expected = float(-2. * np.max(p.best['logpdf']))
    assert abs(p.chi2min - expected) < 1e-12


# ── choice ────────────────────────────────────────────────────────────────────

def test_choice_argmax_preserves_axis():
    p = _make_profiles(n_runs=5)
    c = p.choice()
    assert len(c.best['logpdf']) == 1
    assert len(c.start['p0']) == 1
    assert len(c.error['p0']) == 1
    lo, hi = c.interval['p0']
    assert len(lo) == 1 and len(hi) == 1


def test_choice_int_index():
    p = _make_profiles(n_runs=5)
    c = p.choice(index=2)
    assert len(c.best['logpdf']) == 1
    np.testing.assert_array_equal(c.best['logpdf'],
                                  p.best['logpdf'][[2]])


def test_choice_list_index():
    p = _make_profiles(n_runs=5)
    c = p.choice(index=[0, 1])
    assert len(c.best['logpdf']) == 2


def test_choice_copies_profile_contour_unchanged():
    p = _make_profiles(n_runs=3)
    c = p.choice()
    # profile and contour should be the same dict objects (shallow copy)
    assert c.profile is not p.profile
    assert c.profile['p0'] is p.profile['p0']   # tuple shared
    assert c.contour is not p.contour
    assert c.contour[1] is not p.contour[1]


def test_choice_best_is_best():
    p = _make_profiles(n_runs=5)
    c = p.choice()
    assert c.best['logpdf'][0] == np.max(p.best['logpdf'])


# ── concatenate / update ──────────────────────────────────────────────────────

def test_concatenate_n_runs():
    p1 = _make_profiles(n_runs=3)
    p2 = _make_profiles(n_runs=2)
    pc = Profiles.concatenate(p1, p2)
    assert pc.nruns == 5


def test_concatenate_list_syntax():
    runs = [_make_profiles(n_runs=2) for _ in range(3)]
    pc = Profiles.concatenate(runs)
    assert pc.nruns == 6


def test_concatenate_best_values():
    p1 = _make_profiles(n_runs=2)
    p2 = _make_profiles(n_runs=3)
    pc = Profiles.concatenate(p1, p2)
    expected = np.concatenate([p1.best['p0'], p2.best['p0']])
    np.testing.assert_array_equal(pc.best['p0'], expected)


def test_concatenate_interval():
    p1 = _make_profiles(n_runs=2)
    p2 = _make_profiles(n_runs=3)
    pc = Profiles.concatenate(p1, p2)
    lo_exp = np.concatenate([p1.interval['p0'][0], p2.interval['p0'][0]])
    np.testing.assert_array_equal(pc.interval['p0'][0], lo_exp)


def test_concatenate_profile_merge():
    p1 = _make_profiles(n_runs=1)
    p2 = _make_profiles(n_runs=1)
    pc = Profiles.concatenate(p1, p2)
    # profile keeps first entry (self wins); values must match
    np.testing.assert_array_equal(pc.profile['p0'][0], p1.profile['p0'][0])
    np.testing.assert_array_equal(pc.profile['p0'][1], p1.profile['p0'][1])


def test_concatenate_contour_merge():
    p1 = _make_profiles(n_runs=1)
    p2 = _make_profiles(n_runs=1)
    pc = Profiles.concatenate(p1, p2)
    assert 1 in pc.contour
    assert 2 in pc.contour


def test_extend():
    p1 = _make_profiles(n_runs=2)
    p2 = _make_profiles(n_runs=3)
    p1_copy = copy.deepcopy(p1)
    p1_copy.extend(p2)
    assert p1_copy.nruns == 5


def test_original_unchanged_after_concatenate():
    p1 = _make_profiles(n_runs=2)
    orig_lp = p1.best['logpdf'].copy()
    p2 = _make_profiles(n_runs=3)
    _ = Profiles.concatenate(p1, p2)
    np.testing.assert_array_equal(p1.best['logpdf'], orig_lp)


# ── copy ──────────────────────────────────────────────────────────────────────

def test_shallow_copy_independence():
    p = _make_profiles(n_runs=3)
    p2 = copy.copy(p)
    # Modifying p2's dicts shouldn't affect p
    p2.best['p0'] = np.zeros(3)
    assert not np.all(p.best['p0'] == 0)


def test_deepcopy():
    p = _make_profiles(n_runs=3)
    p2 = p.deepcopy()
    p2.best['p0'][:] = 999.
    assert not np.all(p.best['p0'] == 999.)


# ── contour access ────────────────────────────────────────────────────────────

def test_contour_levels():
    p = _make_profiles(n_runs=1)
    assert set(p.contour.keys()) == {1, 2}


def test_contour_pair_access():
    p = _make_profiles(n_runs=1)
    x, y = p.contour[1][('p1', 'p0')]
    assert x.shape == y.shape


def test_contour_pair_access_parameter_tuple():
    p = _make_profiles(n_runs=1)
    param1 = Parameter('p1', value=0.)
    param0 = Parameter('p0', value=0.)
    x, y = p.contour[1][(param1, param0)]
    sx, sy = p.contour[1][('p1', 'p0')]
    np.testing.assert_array_equal(x, sx)
    np.testing.assert_array_equal(y, sy)


# ── select ────────────────────────────────────────────────────────────────────

def test_select_filters_slots():
    p = _make_profiles(n_runs=2)
    sub = p.select(name=['p0', 'p1'])
    assert set(sub.params.names()) == {'p0', 'p1'}
    assert set(k for k in sub.best if k != 'logpdf') == {'p0', 'p1'}
    assert 'logpdf' in sub.best  # always kept
    assert set(sub.error) == {'p0', 'p1'}
    # contour pairs restricted to selected names
    for cl, pairs in sub.contour.items():
        for p1, p2 in pairs:
            assert p1 in {'p0', 'p1'} and p2 in {'p0', 'p1'}
    # original is untouched
    assert set(k for k in p.best if k != 'logpdf') == {'p0', 'p1', 'p2', 'p3'}


def test_select_requires_params():
    p = Profiles(best={'p0': np.zeros(2), 'logpdf': np.zeros(2)})
    with pytest.raises(ValueError):
        p.select(name='p0')


# ── items ─────────────────────────────────────────────────────────────────────

def test_items():
    p = _make_profiles(n_runs=2)
    names = [name for name, _ in p.items()]
    assert 'params' in names
    assert 'best' in names
    assert 'contour' in names


# ── repr / eq ─────────────────────────────────────────────────────────────────

def test_repr():
    p = _make_profiles(n_runs=1)
    r = repr(p)
    assert 'Profiles' in r
    assert 'best' in r


def test_eq_self():
    p = _make_profiles(n_runs=2)
    assert p == p


def test_eq_copy():
    p = _make_profiles(n_runs=2)
    assert p == copy.deepcopy(p)


def test_neq_different_best():
    p1 = _make_profiles(n_runs=2)
    p2 = copy.deepcopy(p1)
    p2.best['p0'] += 1.
    assert p1 != p2


# ── serialisation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_roundtrip(tmp_path, ext):
    p = _make_profiles(n_runs=3)
    fn = str(tmp_path / f'profiles{ext}')
    p.write(fn)
    loaded = Profiles.read(fn)
    assert loaded == p


@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_attrs(tmp_path, ext):
    p = _make_profiles(n_runs=1)
    p.attrs['sampler'] = 'minuit'
    fn = str(tmp_path / f'profiles{ext}')
    p.write(fn)
    loaded = Profiles.read(fn)
    assert loaded.attrs.get('sampler') == 'minuit'
    assert loaded.attrs.get('ndof') == '10'   # attrs stored as strings


@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_params(tmp_path, ext):
    p = _make_profiles(n_runs=2)
    fn = str(tmp_path / f'profiles{ext}')
    p.write(fn)
    loaded = Profiles.read(fn)
    assert isinstance(loaded.params, VariableCollection)
    assert set(loaded.params.names()) == set(p.params.names())


@pytest.mark.parametrize('ext', ['.h5', '.txt'])
def test_write_read_partial(tmp_path, ext):
    """Profiles with only best (no other slots)."""
    p = Profiles()
    p.best = {'logpdf': np.array([-1., -2., -0.5]),
              'omega_m': np.array([0.3, 0.31, 0.29])}
    fn = str(tmp_path / f'partial{ext}')
    p.write(fn)
    loaded = Profiles.read(fn)
    np.testing.assert_array_equal(loaded.best['omega_m'], p.best['omega_m'])
    assert loaded.start is None
    assert loaded.contour is None


# ── to_stats ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('tablefmt', ['simple', 'list'])
def test_to_stats_runs(tablefmt):
    pytest.importorskip('tabulate')
    p = _make_profiles(n_runs=3)
    result = p.to_stats(tablefmt=tablefmt)
    if tablefmt == 'list':
        rows, headers = result
        assert isinstance(rows, list)
        assert isinstance(headers, list)
    else:
        assert isinstance(result, str)
        assert 'p0' in result


def test_to_stats_no_best_raises():
    pytest.importorskip('tabulate')
    p = Profiles()
    with pytest.raises(ValueError, match='No best'):
        p.to_stats()


def test_to_stats_subset_params():
    pytest.importorskip('tabulate')
    p = _make_profiles(n_runs=3)
    result = p.to_stats(params=['p0', 'p1'], tablefmt='simple')
    assert 'p0' in result
    assert 'p2' not in result


def test_to_stats_quantities():
    pytest.importorskip('tabulate')
    p = _make_profiles(n_runs=3)
    rows, headers = p.to_stats(quantities=['best'], tablefmt='list')
    assert 'error' not in headers
    assert 'interval' not in headers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
