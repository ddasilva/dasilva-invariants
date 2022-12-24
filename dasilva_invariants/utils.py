"""Utility functions for the entire package.

"""
from typing import Tuple

from ai import cs
from astropy import units
import numpy as np
import numpy.typing as npt
import pyvista
import vtk


def nanoTesla2Gauss(nT_values: npt.ArrayLike) -> npt.NDArray:
    """Convert array in nano tesla to Gauss.

    Args
      nT_values: nano tesla values
    Returns
      gauss_values: array of gauss values, same shape
    """
    # ignore types because astropy units broken with typing
    with_units = np.array(nT_values) * units.nT  # type: ignore
    as_gauss = with_units.to(units.G).value

    return as_gauss


def sp2cart_point(r: float, phi: float, theta: float) -> Tuple[float, float, float]:
    """Spherical coordinate to cartesian coordinate conversion for a single point.

    Args
      r: radius
      phi: longitude
      theta: latitude
    Returns
      x, y, z: Cartesian coordinates
    """
    point = cs.sp2cart(r=r, phi=phi, theta=theta)  # returns tuple of 0d arrays
    point = tuple(np.array(point).tolist())

    return point


def cart2sp_point(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Spherical coordinate to cartesian coordinate conversion for a single point.

    Args
      x, y, z: Cartesian coordinates
    Returns
      r: radius
      phi: longitude
      theta: latitude
    """
    point = cs.cart2sp(x=x, y=y, z=z)  # returns tuple of 0d arrays
    point = tuple(np.array(point).tolist())

    return point


def lfm_get_eq_slice(data):
    """Gets an equitorial slice from data on an LFM grid.

    args
      data: numpy array, 3D on LFM grid
    Returns
      eq_c: numpy array, 2D equitorial slice only
    """
    # Adapted from PyLTR
    nk = data.shape[2] - 1
    dusk = data[:, :, 0]
    dawn = data[:, :, nk // 2]
    dawn = dawn[:, ::-1]
    eq = np.hstack((dusk, dawn[:, 1:]))
    eq_c = 0.25 * (eq[:-1, :-1] + eq[:-1, 1:] + eq[1:, :-1] + eq[1:, 1:])
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
    north = data[:, :, nk // 4]
    south = data[:, :, 3 * nk // 4]
    south = south[:, ::-1]  # reverse the j-index
    mer = np.hstack((north, south[:, 1:]))
    mer_c = 0.25 * (mer[:-1, :-1] + mer[:-1, 1:] + mer[1:, :-1] + mer[1:, 1:])
    mer_c = np.append(mer_c.transpose(), [mer_c[:, 0]], axis=0).transpose()

    return mer_c


def interpolate_mesh(
    mesh: pyvista.StructuredGrid, point: Tuple[float, float, float], key: str = "B"
):
    """Linearly interpolate mesh to find value (such as magnetic field) at
    given point.

    Args
      mesh: Grid and magnetic field, loaded using meshes module
      point: Tuple of (x, y, z)
      key: Name of the variable in the mesh to interpolate
    Returns
      Interpolated value of the mesh at given point.
    """
    points_search = pyvista.PolyData(np.array([point]))

    interp = vtk.vtkPointInterpolator()  # linear interpolation
    interp.SetInputData(points_search)
    interp.SetSourceData(mesh)
    interp.Update()

    interp_result = pyvista.PolyData(interp.GetOutput())

    return np.array(interp_result[key])[0]
