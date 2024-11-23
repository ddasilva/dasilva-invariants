from dasilva_invariants import models, invariants
from datetime import datetime
import numpy as np


def test_tsyganenko():
    # Get TS05 model input parameters
    time = datetime(2015, 10, 2)
    params = {'Pdyn': 2.56, 'SymH': -25.0, 'By': 10.81, 'Bz': 0.64, 'W1': 0.524, 'W2': 0.352, 'W3': 0.376, 'W4': 0.342, 'W5': 0.159, 'W6': 0.479}
    
    # Evaluate TS05 model on regular grid
    axis = np.arange(-10, 10, 0.5)
    x, y, z = np.meshgrid(axis, axis, axis)
    model = models.get_tsyganenko(
        "TS05", params, time,
        x_re_sm_grid=x,
        y_re_sm_grid=y,
        z_re_sm_grid=z,
        inner_boundary=1.5
    )

    # Calculate L*
    result = invariants.calculate_LStar(
        model,
        starting_point=(-6.6, 0, 0),
        starting_pitch_angle=60
    )
    
    assert abs(result.LStar - 5.4) < .1

