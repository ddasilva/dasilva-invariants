"""Tests the dasilva_invariants._fortran package"""

from numpy.testing import assert_allclose
from dasilva_invariants._fortran import geopack2008, ts05, t96

def test_functions_exist():
    assert 'recalc' in dir(geopack2008)
    assert 'dipnumpy' in dir(geopack2008)
    assert 'ts05numpy' in dir(ts05)
    assert 't96numpy' in dir(t96)


def test_basic_calls():
    res = geopack2008.recalc((2010, 1, 1, 1, 1), (-400, 0, 0))
    assert res is None

    res = geopack2008.dipnumpy([-5], [-5], [5])
    assert_allclose(res[0], [46.11125])
    assert_allclose(res[1], [46.11125])
    assert_allclose(res[2], [1.9073486e-06])

    res = ts05.ts05numpy([0]*10, 0, [-5], [-5], [5])
    assert_allclose(res[0], [-46.365074])
    assert_allclose(res[1], [-46.365074])
    assert_allclose(res[2], [0.])

    res = t96.t96numpy([0]*10, 0, [-5], [-5], [5])
    assert_allclose(res[0], [-47.07174])
    assert_allclose(res[1], [-47.07174])
    assert_allclose(res[2], [0.])
