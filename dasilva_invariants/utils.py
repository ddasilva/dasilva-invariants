"""Utility functions for the entire package.

"""

import numpy as np


def sm_to_gsm(x_sm, y_sm, z_sm, dipole_tilt):
    """Convert Vector in SM coordinate frame to GSM coordinate frame.

    Args
      x_sm: X-coordinate in SM coordinate system
      y_sm: Y-coordinate in SM coordinate system
      z_sm: Z-coordinate in SM coordinate system
      dipole_tilt: Dipole tilt in radians
    Returns
      x_gsm: X-coordinate in GSM coordinate system
      y_gsm: Y-coordinate in GSM coordinate system
      z_gsm: Z-coordinate in GSM coordinate system
    """
    cps = np.cos(dipole_tilt)
    sps = np.sin(dipole_tilt)
    
    x_gsm = x_sm * cps + z_sm * sps
    y_gsm = y_sm
    z_gsm = z_sm * cps - x_sm * sps
    
    return x_gsm, y_gsm, z_gsm


def gsm_to_sm(x_gsm, y_gsm, z_gsm, dipole_tilt):
    """Convert Vector in SM coordinate frame to GSM coordinate frame.

    Args
      x_gsm: X-coordinate in GSM coordinate system
      y_gsm: Y-coordinate in GSM coordinate system
      z_gsm: Z-coordinate in GSM coordinate system
      dipole_tilt: Dipole tilt in radians
    Returns
      x_sm: X-coordinate in SM coordinate system
      y_sm: Y-coordinate in SM coordinate system
      z_sm: Z-coordinate in SM coordinate system
    """
    cps = np.cos(dipole_tilt)
    sps = np.sin(dipole_tilt)

    x_sm = x_gsm * cps - z_gsm * sps
    y_sm = y_gsm    
    z_sm = x_gsm * sps + z_gsm * cps

    return x_sm, y_sm, z_sm


