from datetime import datetime

import numpy as np
from numpy.testing import assert_allclose

from dasilva_invariants import models


def test_single_time():
    """Test calling get_tsyganenko_params() with a single time"""
    time = datetime(2013, 10, 2, 13, 5)
    params = models.get_tsyganenko_params(time)
    expected = {'Pdyn': 5.460000038146973, 'SymH': -56.0, 'By': -1.0700000524520874, 'Bz': 5.579999923706055}

    for key, value in params.items():
        assert isinstance(value, float)
        assert abs(value - expected[key]) < .1

        
def test_multiple_times():
    """Test calling get_tsyganenko_params() with multiple times"""    
    times = [
        datetime(2013, 10, 2, 13, 5),
        datetime(2013, 10, 2, 13, 10),
        datetime(2013, 10, 2, 13, 15),
    ]
    params = models.get_tsyganenko_params(times)
    expected = {
        'Pdyn': np.array([5.46000004, 4.88999987, 4.80000019]),
        'SymH': np.array([-56., -56., -56.]),
        'By': np.array([-1.07000005,  2.43000007, -1.26999998]),
        'Bz': np.array([5.57999992, 3.72000003, 4.15999985])
    }

    for key, value in params.items():
        assert_allclose(value, expected[key], rtol=.01, atol=0.1)

