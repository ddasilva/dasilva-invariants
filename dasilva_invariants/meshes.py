"""Tools for obtaining meshes for use in calculating invariants.

Meshes are grids + magnetic field vectors at those grid points. They
are instances of pyvista.StructuredGrid. PyVista is used throughout
this project.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from ai import cs
from astropy import constants, units
from datetime import datetime, timedelta
import PyGeopack as gp
from joblib import Parallel, delayed
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
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

    B0 = -30e3
    Bx = 3 * x_re * z_re * B0 / r_re**5
    By = 3 * y_re * z_re * B0 / r_re**5
    Bz = (3 * z_re**2 - r_re**2) * B0 / r_re**5

    Bx *= units.nT
    By *= units.nT
    Bz *= units.nT

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_grid, Y_grid, Z_grid)

    _add_spherical_coords_to_mesh(mesh, X_grid, Y_grid, Z_grid)

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

    for j in range(0, njp2, njp1):
        jAxis = max(1, min(nj, j))
        for i in range(ni):
            X_grid[i, j, :-1] = X_grid[i, jAxis, :-1].mean()
            Y_grid[i, j, :-1] = Y_grid[i, jAxis, :-1].mean()
            Z_grid[i, j, :-1] = Z_grid[i, jAxis, :-1].mean()

    X_grid[:, :, nk] = X_grid[:, :, 0]
    Y_grid[:, :, nk] = Y_grid[:, :, 0]
    Z_grid[:, :, nk] = Z_grid[:, :, 0]

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

    Bx, By, Bz = _apply_murphy_lfm_grid_patch(Bx_raw, By_raw, Bz_raw)
    
    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_grid, Y_grid, Z_grid)
    
    _add_spherical_coords_to_mesh(mesh, X_grid, Y_grid, Z_grid)
    
    B = np.empty((mesh.n_points, 3))
    B[:, 0] = Bx.flatten(order='F')
    B[:, 1] = By.flatten(order='F')
    B[:, 2] = Bz.flatten(order='F')

    mesh.point_data['B'] = B
        
    return mesh


def _add_spherical_coords_to_mesh(mesh, X_grid, Y_grid, Z_grid):
    """Add pre-compute spherical coordinates of grid to mesh in place.
    
    Args
      mesh: Mesh to add to
      X_grid: three-dimensional grid of X coordinates
      Y_grid: three-dimensional grid of Y coordinates
      Z_grid: three-dimensional grid of Z coordinates
    """
    R_grid, Theta_grid, Phi_grid = cs.cart2sp(X_grid, Y_grid, Z_grid)
    mesh.point_data['R_grid'] = R_grid.flatten(order='F')
    mesh.point_data['Phi_grid'] = Phi_grid.flatten(order='F')
    mesh.point_data['Theta_grid'] = Theta_grid.flatten(order='F')


def _apply_murphy_lfm_grid_patch(Bx_raw, By_raw, Bz_raw):
    """Apply Josh Murphy's patch to the LFM grid.

    This code is Josh Murphy's point2CellCenteredVector() function converted
    to python.

    Args
     Bx_raw, By_raw, Bz_raw: The magnetic field in the raw grid.
    Returns
     Bx, By, Bz: The magnetic field in the patched grid
    """    
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

    Bx[:, 1:, :] = Bx_raw[:-1, :, :]
    By[:, 1:, :] = By_raw[:-1, :, :]
    Bz[:, 1:, :] = Bz_raw[:-1, :, :]

    for j in range(0, njp2, njp1):
        jAxis = max(1, min(nj, j))
        for i in range(ni):
            Bx[i, j, :-1] = Bx[i, jAxis, :-1].mean()
            By[i, j, :-1] = By[i, jAxis, :-1].mean()
            Bz[i, j, :-1] = Bz[i, jAxis, :-1].mean()

    Bx[:, :, nk] = Bx[:, :, 0]
    By[:, :, nk] = By[:, :, 0]
    Bz[:, :, nk] = Bz[:, :, 0]

    return Bx, By, Bz


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


def _get_tsyganenko_on_lfm_grid(
    model_name, params, time, lfm_hdf4_path, external_field_only=False
):
    """Internal helper function to get one of the tsyganenko fields on an LFM grid.

    Args
      model_name: Name of the model, either 'T96' or 'T04'
      params: Model parameters: tuple of 10 values
      time: Time for T89 model, sets parameters
      lfm_hdf4_path: Path to LFM HDF4 file to set grid.
      external_field_only: Set to True to not include the internal (dipole) model
      force_zero_tilt: Force a zero tilt when calculating the file
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

    # Call Geopack to get external fields
    # ------------------------------------------------------------------------    
    date = int(time.strftime('%Y%m%d')) 
    ut = int(time.strftime('%H')) + time.minute / 60

    Bx, By, Bz = (
        gp.ModelField(x_re_sm, y_re_sm, z_re_sm, Date=date, ut=ut,
                      Model=model_name, CoordIn='SM', CoordOut='SM', **params)
    )

    Bx = (Bx * units.nT).to(units.G).value
    By = (By * units.nT).to(units.G).value
    Bz = (Bz * units.nT).to(units.G).value

    # Calculate dipole field for internal model,
    # see Kivelson and Russel equations 6.3(a)-(c), page 165.
    # ----------------------------------------------------------------------
    if external_field_only:
        r_re_sm = np.sqrt(x_re_sm**2 + y_re_sm**2 + z_re_sm**2)

        B0 = -30e3
        Bx_int = 3 * x_re_sm * z_re_sm * B0 / r_re_sm**5
        By_int = 3 * y_re_sm * z_re_sm * B0 / r_re_sm**5
        Bz_int = (3 * z_re_sm**2 - r_re_sm**2) * B0 / r_re_sm**5

        Bx_int = (Bx_int * units.nT).to(units.G).value
        By_int = (By_int * units.nT).to(units.G).value
        Bz_int = (Bz_int * units.nT).to(units.G).value

        Bx -= Bx_int
        By -= By_int
        Bz -= Bz_int

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_re_sm_grid, Y_re_sm_grid, Z_re_sm_grid)

    _add_spherical_coords_to_mesh(
        mesh, X_re_sm_grid, Y_re_sm_grid, Z_re_sm_grid
    )

    B = np.empty((mesh.n_points, 3))
    B[:, 0] = Bx
    B[:, 1] = By
    B[:, 2] = Bz
    mesh.point_data['B'] = B

    # Return output
    return mesh


def get_tsyganenko_params(times, path, tell_params=True, __T_AUTO_DL_CACHE={}):
    """Get parameters for tsyganenko models.

    Path is location of the following zip file ond isk:
    http://virbo.org/ftp/QinDenton/hour/merged/latest/WGhour-latest.d.zip

    Args:
      times: List of times, or individual time
      path: Path to zip file (may be URL if network enabled)
      tell_params: If set to true, prints parameters to output
    Returns:
      dictionary mapping variable to array of parameters
    """
    try:
        iter(times)
    except TypeError:
        times = [times]

    if path in __T_AUTO_DL_CACHE:
        if tell_params:
            print(f'Getting {path} from cache')
        df = __T_AUTO_DL_CACHE[path]
    else:
        if tell_params:
            print(f'Loading {path}')
        df = pd.read_csv(path, index_col=False, sep='\s+', skiprows=1, names=[
           'Year', 'Day', 'Hr', 'By', 'Bz', 'V_SW', 'Den_P', 'Pdyn',
           'G1', 'G2', 'G3', '8_status', 'kp', 'akp3', 'SymH',
           'Bz1', 'Bz2', 'Bz3', 'Bz4', 'Bz5', 'Bz6',
           'W1', 'W2', 'W3', 'W4', 'W5', 'W6', '6_stat',
        ])
        __T_AUTO_DL_CACHE[path] = df

    min_year = min(time.year for time in times)
    max_year = max(time.year for time in times)

    mask = (df.Year > (min_year - 1)) & (df.Year < (max_year + 1))
    df = df[mask].copy()
    df['DateTime'] = [
        datetime(int(row.Year), 1, 1) +
        timedelta(days=row.Day - 1, hours=row.Hr)
        for _, row in df.iterrows()
    ]

    # Interpolate Tsyganenko parameters (some may be unused)
    cols = ['Pdyn', 'SymH', 'By', 'Bz', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    params_dict = {}

    for col in cols:
        params_dict[col] = np.interp(
            date2num(times), date2num(df.DateTime), df[col]
        )

    if tell_params:
        print(params_dict)

    return params_dict


def get_tsyganenko_on_lfm_grid_with_auto_params(
    model_name, time, lfm_hdf4_path, param_path, tell_params=True, **kwargs
):
    """Get a Tsyganenko field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      model_name: String name of model (T96 or T04)
      time: Time for model, sets auto omni paramteters
      lfm_hdf4_path: Path to LFM HDF4 file to set grid.
      params_path: Path to OMNI records file, can be URL to download.
      tell_params: Print OMNI paramteters retreived to standard output
      external_field_only: Set to True to not include the internal (dipole)
        model
      force_zero_tilt: Force a zero tilt when calculating the file
      n_jobs: Number of parallel processes to use (-1 for all available cores)
      verbose: Verbosity level (see joblib.Parallel documentation)
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    # Lookup params  -----------------------------------------------
    params = get_tsyganenko_params(time, param_path, tell_params=tell_params)

    # Call model using parameters
    return _get_tsyganenko_on_lfm_grid(
        model_name, params, time, lfm_hdf4_path, **kwargs
    )


