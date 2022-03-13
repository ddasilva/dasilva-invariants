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


def lfm_get_eq_slice(data):
    """Gets an equitorial slice from data on an LFM grid.       

    args
      data: numpy array, 3D on LFM grid
    Returns
      eq_c: numpy array, 2D equitorial slice only
    """
    # Adapted from PyLTR
    nk = data.shape[2]-1
    dusk = data[:, :, 0]
    dawn = data[:, :, nk//2]
    dawn = dawn[:, ::-1]
    eq = np.hstack((dusk, dawn[:, 1:]))
    eq_c = 0.25*(eq[:-1, :-1] + eq[:-1, 1:] + eq[1:, :-1] + eq[1:, 1:])
    eq_c = np.append(eq_c.transpose(), [eq_c[:, 0]], axis=0).transpose()

    return eq_c


def lfm_get_mer_slice(data):
    """Gets an meridional slice from data on an LFM grid.       

    args
      data: numpy array, 3D on LFM grid
    Returns
      mer_c: numpy array, 2D meridional slice only
    """
    # Adapted from pyLTR
    nk = data.shape[2] - 1
    north = data[:, :, nk//4]
    south = data[:, :, 3*nk//4]
    south = south[:, ::-1] # reverse the j-index
    mer = np.hstack((north, south[:,1:]))
    mer_c = 0.25*(mer[:-1,:-1] + mer[:-1,1:] + mer[1:,:-1] + mer[1:,1:])
    mer_c = np.append(mer_c.transpose(),[mer_c[:,0]],axis=0).transpose()

    return mer_c
