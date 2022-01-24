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


def get_dipole_mesh_on_lfm_grid(lfm_hdf4_path):
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      lfm_hdf4_path: Path to LFM hdf4 file
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    X_grid, Y_grid, Z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

    x_re = X_grid.flatten(order='F')  # flat arrays, easier later
    y_re = Y_grid.flatten(order='F')
    z_re = Z_grid.flatten(order='F')

    # Calculate dipole model
    # ------------------------------------------------------------------------
    # Dipole model, per Kivelson and Russel equations 6.3(a)-(c), page 165.
    r_re = np.sqrt(x_re**2 + y_re**2 + z_re**2)

    B0 = 30e3
    Bx = 3 * x_re * z_re * B0 / r_re**5
    By = 3 * y_re * z_re * B0 / r_re**5
    Bz = (3 * z_re**2 - r_re**2) * B0 / r_re**5

    Bx *= units.nT
    By *= units.nT
    Bz *= units.nT

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_grid, Y_grid, Z_grid)

    B = np.empty((mesh.n_points, 3))
    B[:, 0] = Bx.to(units.G).value
    B[:, 1] = By.to(units.G).value
    B[:, 2] = Bz.to(units.G).value
    mesh.point_data['B'] = B

    # Return output
    return mesh


def _get_fixed_lfm_grid_centers(lfm_hdf4_path):
    """Loads LFM grid centers with singularity patched. 

    This code is adapted from Josh Murphy's GhostPy and converted to python
    (crudely).
    
    Args
      lfm_hdf4_path: Path to LFM HDF4 file
    Returns
      X_grid, Y_grid, Z_grid: arrays of LFM grid cell centers in units of Re
        with the x-axis singularity issue fixed.
    """
    # Read LFM grid from HDF file 
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)
    X_grid_raw = _fix_lfm_hdf4_array_order(hdf.select('X_grid').get())
    Y_grid_raw = _fix_lfm_hdf4_array_order(hdf.select('Y_grid').get())
    Z_grid_raw = _fix_lfm_hdf4_array_order(hdf.select('Z_grid').get())
    
    # This code implements Josh Murphy's point2CellCenteredGrid() function
    # ------------------------------------------------------------------------
    ni   = X_grid_raw.shape[0] - 1
    nip1 = X_grid_raw.shape[0]
    nip2 = X_grid_raw.shape[0] + 1

    nj   = X_grid_raw.shape[1] - 1
    njp1 = X_grid_raw.shape[1]
    njp2 = X_grid_raw.shape[1] + 1

    nk   = X_grid_raw.shape[2] - 1
    nkp1 = X_grid_raw.shape[2]
    nkp2 = X_grid_raw.shape[2] + 1

    X_grid = np.zeros((ni, njp2, nkp1))
    Y_grid = np.zeros((ni, njp2, nkp1))
    Z_grid = np.zeros((ni, njp2, nkp1))
    
    X_grid[:, 1:-1, :-1] = _calc_cell_centers(X_grid_raw)
    Y_grid[:, 1:-1, :-1] = _calc_cell_centers(Y_grid_raw)
    Z_grid[:, 1:-1, :-1] = _calc_cell_centers(Z_grid_raw)
    
    jAxis = 0
    axisCoord = np.zeros(3)
    xyz = np.zeros(3)    

    for j in range(0, njp2, njp1):
        jAxis = max(1, min(nj, j))
        for i in range(ni):
            xyz[:] = 0
            for k in range(nk):
                xyz[:] += [X_grid[i, jAxis, k],
                           Y_grid[i, jAxis, k],
                           Z_grid[i, jAxis, k]]

            xyz[0] /= nk
            for k in range(nk):
                X_grid[i, j, k] = xyz[0]
                Y_grid[i, j, k] = xyz[1]
                Z_grid[i, j, k] = xyz[2]

    for j in range(njp2):
        for i in range(ni):
            X_grid[i, j, nk] = X_grid[i, j, 0]
            Y_grid[i, j, nk] = Y_grid[i, j, 0]
            Z_grid[i, j, nk] = Z_grid[i, j, 0]

    # Convert to units of earth radii (Re)
    # ------------------------------------------------------------------------
    X_grid_re = (X_grid * units.cm).to(constants.R_earth).value
    Y_grid_re = (Y_grid * units.cm).to(constants.R_earth).value
    Z_grid_re = (Z_grid * units.cm).to(constants.R_earth).value

    return X_grid_re, Y_grid_re, Z_grid_re


def get_lfm_hdf4_data(lfm_hdf4_path):
    """Get a magnetic field data + grid from LDM output. Uses an LFM HDF4 file.

    Args
      lfm_hdf4_path: Path to LFM hdf4 file
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with LFM dynamic
        field values. Grid is in units of Re and magnetic field is is units o
        Gauss.
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    X_grid, Y_grid, Z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)
    
    # Read LFM B values from HDF file 
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)

    Bx_raw = _fix_lfm_hdf4_array_order(hdf.select('bx_').get())
    By_raw = _fix_lfm_hdf4_array_order(hdf.select('by_').get())
    Bz_raw = _fix_lfm_hdf4_array_order(hdf.select('bz_').get())

    # This code is Josh Murphy's point2CellCenteredVector() function converted
    # to python.
    # ------------------------------------------------------------------------
    ni   = Bx_raw.shape[0] - 1
    nip1 = Bx_raw.shape[0]
    nip2 = Bx_raw.shape[0] + 1

    nj   = Bx_raw.shape[1] - 1
    njp1 = Bx_raw.shape[1]
    njp2 = Bx_raw.shape[1] + 1

    nk   = Bx_raw.shape[2] - 1
    nkp1 = Bx_raw.shape[2]
    nkp2 = Bx_raw.shape[2] + 1

    Bx = np.zeros((ni, njp2, nkp1))
    By = np.zeros((ni, njp2, nkp1))
    Bz = np.zeros((ni, njp2, nkp1))

    offsetData = offsetCell = 0
    tuple = np.zeros(3)

    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                Bx[i, j+1, k] = Bx_raw[i, j, k]            
                By[i, j+1, k] = By_raw[i, j, k]
                Bz[i, j+1, k] = Bz_raw[i, j, k]

    tupleDbl = np.zeros(3)
    for j in range(0, njp2, njp1):
        jAxis = max(1, min(nj, j))
        for i in range(ni):
            tuple[:] = 0

            for k in range(nk):
                tuple[:] += [Bx[i, jAxis, k],
                             By[i, jAxis, k],
                             Bz[i, jAxis, k]]
            tuple /= nk

            for k in range(nk):
                Bx[i, j, k] = tuple[0]
                By[i, j, k] = tuple[1]
                Bz[i, j, k] = tuple[2]

    for j in range(njp2):
        for i in range(ni):
            Bx[i, j, nk] = Bx[i, j, 0]
            By[i, j, nk] = By[i, j, 0]
            Bz[i, j, nk] = Bz[i, j, 0]
            
    # Create PyVista structured grid.
    # ------------------------------------------------------------------------    
    mesh = pyvista.StructuredGrid(X_grid, Y_grid, Z_grid)
    
    B = np.empty((mesh.n_points, 3))
    B[:, 0] = Bx.flatten(order='F')
    B[:, 1] = By.flatten(order='F')
    B[:, 2] = Bz.flatten(order='F')
    mesh.point_data['B'] = B    
    
    # Return output
    return mesh

              
def _calc_cell_centers(A):
    """Calculates centers of cells on a 3D grid.

    Args
      3D grid holding grid positions on one of X, Y or Z for each grid 
      coordinate.
    Returns
      3D array of X, Y, or Z positions for grid coordinates
    """
    s = A.shape

    centers = np.zeros((s[0]-1, s[1]-1, s[2]-1))
    
    centers += A[:-1, :-1, :-1]  # i,   j,   k

    centers += A[1:, :-1, :-1]   # i+1, j,   k
    centers += A[:-1, 1:, :-1]   # i,   j+1, k
    centers += A[:-1, :-1, 1:]   # i,   j,   k+1

    centers += A[1:, 1:, :-1]    # i+1, j+1, k
    centers += A[:-1, 1:, 1:]    # i,   j+1, k+1
    centers += A[1:, :-1, 1:]    # i+1, j,   k+1
    
    centers += A[1:, 1:, 1:]     # i+1, j+1, k+1

    centers /= 8.0

    return centers

   
def get_t96_mesh_on_lfm_grid(dynamic_pressure, Dst, By_imf, Bz_imf,
                             lfm_hdf4_path, time=datetime(1970, 1, 1),
                             n_jobs=-1, verbose=1000):
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      lfm_hdf4_path: Path to LFM hdf4 file      
      time: Time for T89 model, sets parameters
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    X_re_sm_grid, Y_re_sm_grid, Z_re_sm_grid = (
        _get_fixed_lfm_grid_centers(lfm_hdf4_path)
    )
    
    x_re_sm = X_re_sm_grid.flatten(order='F')  # flat arrays, easier for later
    y_re_sm = Y_re_sm_grid.flatten(order='F')
    z_re_sm = Z_re_sm_grid.flatten(order='F')

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

    for i in range(x_re_gsm.shape[0]):
        task = delayed(_t96_parallel_helper)(
            i, params, x_re_gsm[i], y_re_gsm[i], z_re_gsm[i], dipole_tilt
        )
        tasks.append(task)
            
    results = Parallel(verbose=verbose, n_jobs=n_jobs,
                       backend='multiprocessing')(tasks)

    # Repopulate parallel results into single array and convert to SM
    # coordinates
    B_shape = (3, x_re_gsm.shape[0])
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
    
    B = np.empty((mesh.n_points, 3))
    B[:, 0] = B_t96[0, :].to(units.G).value
    B[:, 1] = B_t96[1, :].to(units.G).value
    B[:, 2] = B_t96[2, :].to(units.G).value
    mesh.point_data['B'] = B
    
    # Return output
    return mesh


def _t96_parallel_helper(i, params, x_re_gsm, y_re_gsm, z_re_gsm, dipole_tilt):
    internal_field_vec = geopack.dip(x_re_gsm, y_re_gsm, z_re_gsm)
    external_field_vec = t96.t96(params, dipole_tilt,
                                 x_re_gsm, y_re_gsm, z_re_gsm)

    return i, internal_field_vec, external_field_vec
