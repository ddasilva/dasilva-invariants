"""Tools for obtaining meshes for use in calculating invariants.

Meshes are grids + magnetic field vectors at those grid points. They
are instances of pyvista.StructuredGrid. PyVista is used throughout
this project.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from astropy import constants, units
from datetime import datetime
from geopack import geopack, t96
from joblib import Parallel, delayed
import numpy as np
from pyhdf.SD import SD, SDC
import pyvista

from .utils import sm_to_gsm, gsm_to_sm



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
    data = np.reshape(data.ravel(), (s[2], s[1], s[0]), order='F')
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


def get_dipole_mesh_on_lfm_grid(lfm_hdf4_path):
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      lfm_hdf4_path: Path to LFM hdf4 file
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units o
        Gauss.
    """
    # Read LFM grid from HDF file and convert to units of km
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)
    X_grid = _fix_lfm_hdf4_array_order(hdf.select('X_grid').get())
    Y_grid = _fix_lfm_hdf4_array_order(hdf.select('Y_grid').get())
    Z_grid = _fix_lfm_hdf4_array_order(hdf.select('Z_grid').get())
    X_grid *= units.cm
    Y_grid *= units.cm
    Z_grid *= units.cm

    X_grid_km = X_grid.to(units.km).value
    Y_grid_km = Y_grid.to(units.km).value
    Z_grid_km = Z_grid.to(units.km).value

    # Compute cell centers
    # ------------------------------------------------------------------------
    cell_centers = (
        pyvista.StructuredGrid(X_grid_km, Y_grid_km, Z_grid_km)
        .cell_centers()
    )
    x_km = cell_centers.points[:, 0] * units.km
    y_km = cell_centers.points[:, 1] * units.km
    z_km = cell_centers.points[:, 2] * units.km

    # Calculate dipole model 
    # ------------------------------------------------------------------------
    # Dipole model, per Kivelson and Russel equations 6.3(a)-(c), page 165.
    r_re = np.sqrt(x_km**2 + y_km**2 + z_km**2)
    r_re = r_re.to(constants.R_earth).value
    
    x_re = x_km.to(constants.R_earth).value 
    y_re = y_km.to(constants.R_earth).value 
    z_re = z_km.to(constants.R_earth).value 
    
    B0 = 30e3 
    Bx = 3 * x_re * z_re * B0 / r_re**5
    By = 3 * y_re * z_re * B0 / r_re**5
    Bz = (3 * z_re**2 - r_re**2) * B0 / r_re**5

    Bx *= units.nT
    By *= units.nT
    Bz *= units.nT

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    X_grid_re = X_grid.to(constants.R_earth).value
    Y_grid_re = Y_grid.to(constants.R_earth).value
    Z_grid_re = Z_grid.to(constants.R_earth).value

    mesh = pyvista.StructuredGrid(X_grid_re, Y_grid_re, Z_grid_re)

    B = np.empty((mesh.n_cells, 3))
    B[:, 0] = Bx.to(units.G).value
    B[:, 1] = By.to(units.G).value
    B[:, 2] = Bz.to(units.G).value
    mesh['B'] = B

    mesh = mesh.cell_data_to_point_data()
    
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


def get_t96_mesh_on_lfm_grid(dynamic_pressure, Dst, By_imf, Bz_imf,
                             lfm_hdf4_path, time=datetime(1970, 1, 1),
                             n_jobs=-1, verbose=1000):
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.
l
    Args
      lfm_hdf4_path: Path to LFM hdf4 file      
      time: Time for T89 model, sets parameters
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    # Read LFM grid from HDF file and convert to units of km and re
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)
    X_hdf_grid = _fix_lfm_hdf4_array_order(hdf.select('X_grid').get())
    Y_hdf_grid = _fix_lfm_hdf4_array_order(hdf.select('Y_grid').get())
    Z_hdf_grid = _fix_lfm_hdf4_array_order(hdf.select('Z_grid').get())
    X_hdf_grid *= units.cm
    Y_hdf_grid *= units.cm
    Z_hdf_grid *= units.cm

    X_km_sm_grid = X_hdf_grid.to(units.km).value
    Y_km_sm_grid = Y_hdf_grid.to(units.km).value
    Z_km_sm_grid = Z_hdf_grid.to(units.km).value

    X_re_sm_grid = X_hdf_grid.to(constants.R_earth).value
    Y_re_sm_grid = Y_hdf_grid.to(constants.R_earth).value
    Z_re_sm_grid = Z_hdf_grid.to(constants.R_earth).value
    
    # Compute cell centers to evaluate T96 at
    # ------------------------------------------------------------------------
    cell_centers = (
        pyvista.StructuredGrid(X_km_sm_grid, Y_km_sm_grid, Z_km_sm_grid)
        .cell_centers()
    )
    x_km_sm = cell_centers.points[:, 0] * units.km
    y_km_sm = cell_centers.points[:, 1] * units.km
    z_km_sm = cell_centers.points[:, 2] * units.km
    num_cells = cell_centers.points.shape[0]
    
    x_re_sm = x_km_sm.to(constants.R_earth).value
    y_re_sm = y_km_sm.to(constants.R_earth).value
    z_re_sm = z_km_sm.to(constants.R_earth).value

    # Calibrate geopack using the specified time and convert grid to GSM using
    # dipole tilt
    # ------------------------------------------------------------------------
    epoch = datetime(1970, 1, 1)
    seconds = (time - epoch).total_seconds()
    dipole_tilt = geopack.recalc(seconds)

    x_re_gsm, y_re_gsm, z_re_gsm = sm_to_gsm(x_re_sm, y_re_sm, z_re_sm,
                                             dipole_tilt)
    
    # Calculate the internal (dipole) and external (t96) fields using the
    # geopack module
    # ------------------------------------------------------------------------    
    # Use joblib to process in parallel using the number of processes and
    # verbosity settings specified by caller
    params = (dynamic_pressure, Dst, By_imf, Bz_imf, 0, 0, 0, 0, 0, 0)
    tasks = []

    for i in range(num_cells):    
        task = delayed(_t96_parallel_helper)(
            i, params, x_re_gsm[i], y_re_gsm[i], z_re_gsm[i], dipole_tilt
        )
        tasks.append(task)
            
    results = Parallel(verbose=verbose, n_jobs=n_jobs,
                       backend='multiprocessing')(tasks)

    # Repopulate parallel results into single array and convert to SM
    # coordinates
    B_shape = (3, num_cells)
    B_internal = np.zeros(B_shape)          # GSM Coordinates
    B_external = np.zeros(B_shape)          # GSM Coordinates

    for i, internal_field_vec, external_field_vec in results:
        B_internal[:, i] = internal_field_vec
        B_external[:, i] = external_field_vec

    B_t96 = gsm_to_sm(*(B_internal + B_external), dipole_tilt)
    B_t96 *= units.nT

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_re_sm_grid, Y_re_sm_grid, Z_re_sm_grid)
    
    B = np.empty((mesh.n_cells, 3))
    B[:, 0] = B_t96[0, :].to(units.G).value
    B[:, 1] = B_t96[1, :].to(units.G).value
    B[:, 2] = B_t96[2, :].to(units.G).value
    mesh['B'] = B

    mesh = mesh.cell_data_to_point_data()
    
    # Return output
    return mesh



def _t96_parallel_helper(i, params, x_re_gsm, y_re_gsm, z_re_gsm, dipole_tilt):
    internal_field_vec = geopack.dip(x_re_gsm, y_re_gsm, z_re_gsm)
    external_field_vec = t96.t96(params, dipole_tilt,
                                 x_re_gsm, y_re_gsm, z_re_gsm)

    return i, internal_field_vec, external_field_vec
