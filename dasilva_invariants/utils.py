"""Utility functions for the entire package.

"""
import numpy as np


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
