"""Tools for obtaining meshes for use in calculating invariants.

Meshes are grids + magnetic field vectors at those grid points. They
are instances of pyvista.StructuredGrid. PyVista is used throughout
this project.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from ai import cs
from astropy import constants, units
import numpy as np
from pyhdf.SD import SD, SDC
import pyvista


def _fix_lfm_hdf4_array_order(data):
    """Apply fix to LFM data to account for strange format in HDF4 file.

    This must be called on all data read from an LFM HDF4 file.

    Adapted from pyLTR, which has the following comment:
       Data is stored in C-order, but shape is transposed in
       Fortran order!  Why?  Because LFM stores its arrays using
       the rather odd A++/P++ library.  Let's reverse this evil:
    Args
      data: array with 3 dimensions
    Returns
      data: array with (i, j, k) fixed
    """
    s = data.shape
    data = np.reshape( data.ravel(), (s[2], s[1], s[0]), order='F' )
    return data


def _convert_to_pyvista_order(data):
    """This function converts an array used in a PyVista mesh from its 3D
    form to its a 1D form required to be stored in the mesh.

    Args
      data: array with three dimensions
    Returns
      data: 1D flattened array suitable to be added to a PyVista mesh
    """
    s = data.shape
    data = np.reshape(data.ravel(), (s[2], s[1], s[0]), order='F').flatten()
    return data


def get_dipole_mesh_on_lfm_grid(lfm_hdf4_path, xstretch=1, ystretch=1, zstretch=1):
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      lfm_hdf4_path: Path to LFM hdf4 file
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units o
        Gauss.
    """
    # Read LFM grid from HDF file and attach units of cm
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)
    X_grid = _fix_lfm_hdf4_array_order(hdf.select('X_grid').get())
    Y_grid = _fix_lfm_hdf4_array_order(hdf.select('Y_grid').get())
    Z_grid = _fix_lfm_hdf4_array_order(hdf.select('Z_grid').get())

    X_grid *= units.cm
    Y_grid *= units.cm
    Z_grid *= units.cm

    X_grid = X_grid.to(units.km)   # prevent overflow
    Y_grid = Y_grid.to(units.km)
    Z_grid = Z_grid.to(units.km)
    
    # Calculate dipole model
    # ------------------------------------------------------------------------
    # Dipole model, per Kivelson and Russel equations
    # 6.3(a)-(c), page 165.
    r = np.sqrt(X_grid**2 + Y_grid**2 + Z_grid**2)
    r = r.to(constants.R_earth).value
    
    x = X_grid.to(constants.R_earth).value * xstretch
    y = Y_grid.to(constants.R_earth).value * ystretch
    z = Z_grid.to(constants.R_earth).value * zstretch
    
    B0 = 30e3 
    Bx = 3 * x * z * B0 / r**5
    By = 3 * y * z * B0 / r**5
    Bz = (3 * z**2 - r**2) * B0 / r**5

    Bx *= units.nT
    By *= units.nT
    Bz *= units.nT

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    X_grid_in_re = (X_grid / constants.R_earth).to(1).value
    Y_grid_in_re = (Y_grid / constants.R_earth).to(1).value
    Z_grid_in_re = (Z_grid / constants.R_earth).to(1).value

    mesh = pyvista.StructuredGrid(X_grid_in_re, Y_grid_in_re, Z_grid_in_re)
    
    B = np.empty((mesh.n_points, 3))
    B[:, 0] = _convert_to_pyvista_order(Bx.to(units.G).value)
    B[:, 1] = _convert_to_pyvista_order(By.to(units.G).value)
    B[:, 2] = _convert_to_pyvista_order(Bz.to(units.G).value)
    mesh['B'] = B

    # Return output
    return mesh


def get_lfm_hdf4_data(lfm_hdf4_path):
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      lfm_hdf4_path: Path to LFM hdf4 file
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units o
        Gauss.
    """
    # Read LFM grid from HDF file and attach units of cm
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)
    X_grid = _fix_lfm_hdf4_array_order(hdf.select('X_grid').get())
    Y_grid = _fix_lfm_hdf4_array_order(hdf.select('Y_grid').get())
    Z_grid = _fix_lfm_hdf4_array_order(hdf.select('Z_grid').get())

    X_grid *= units.cm
    Y_grid *= units.cm
    Z_grid *= units.cm

    X_grid = X_grid.to(units.km)   # prevent overflow
    Y_grid = Y_grid.to(units.km)
    Z_grid = Z_grid.to(units.km)
    
    # Read LFM B values from HDF file and attach units of Gauss
    # ------------------------------------------------------------------------
    Bx = _fix_lfm_hdf4_array_order(hdf.select('bx_').get())
    By = _fix_lfm_hdf4_array_order(hdf.select('by_').get())
    Bz = _fix_lfm_hdf4_array_order(hdf.select('bz_').get())

    Bx = Bx[:-1, :-1, :-1]
    By = By[:-1, :-1, :-1]
    Bz = Bz[:-1, :-1, :-1]
    
    Bx *= units.G
    By *= units.G
    Bz *= units.G

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    X_grid_in_re = (X_grid / constants.R_earth).to(1).value
    Y_grid_in_re = (Y_grid / constants.R_earth).to(1).value
    Z_grid_in_re = (Z_grid / constants.R_earth).to(1).value
    
    mesh = pyvista.StructuredGrid(X_grid_in_re, Y_grid_in_re, Z_grid_in_re)
    
    B = np.empty((mesh.n_cells, 3))
    B[:, 0] = _convert_to_pyvista_order(Bx.to(units.G).value)
    B[:, 1] = _convert_to_pyvista_order(By.to(units.G).value)
    B[:, 2] = _convert_to_pyvista_order(Bz.to(units.G).value)
    mesh['B'] = B    
    
    mesh = mesh.cell_data_to_point_data()
    
    # Return output
    return mesh
