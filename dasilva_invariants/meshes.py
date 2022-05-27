"""Tools for obtaining meshes for use in calculating invariants.

Meshes are grids + magnetic field vectors at those grid points. They
are instances of pyvista.StructuredGrid. PyVista is used throughout
this project.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from ai import cs
from astropy import constants, units
from datetime import datetime
from geopack import geopack, t96, t04
from joblib import Parallel, delayed
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
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

    # Pre-compute spherical coordinates of grid
    # ------------------------------------------------------------------------
    R_grid, Phi_grid, Theta_grid = cs.cart2sp(X_grid, Y_grid, Z_grid)

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_grid, Y_grid, Z_grid)

    B = np.empty((mesh.n_points, 3))
    B[:, 0] = Bx.flatten(order='F')
    B[:, 1] = By.flatten(order='F')
    B[:, 2] = Bz.flatten(order='F')

    mesh.point_data['B'] = B
    mesh.point_data['R_grid'] = R_grid.flatten(order='F')
    mesh.point_data['Phi_grid'] = Phi_grid.flatten(order='F')
    mesh.point_data['Theta_grid'] = Theta_grid.flatten(order='F')

    # Return output
    return mesh


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
    model_name, params, lfm_hdf4_path, time=datetime(1970, 1, 1),
    external_field_only=False, force_zero_tilt=True, n_jobs=-1, verbose=1000
):
    """Internal helper function to get one of the tsyganenko fields on an LFM grid.

    Args
      model_name: Name of the model, either 'T96' or 'T04'
      params: Model parameters: tuple of 10 values
      time: Time for T89 model, sets parameters
      external_field_only: Set to True to not include the internal (dipole) model
      force_zero_tilt: Force a zero tilt when calculating the file
      n_jobs: Number of parallel processes to use (-1 for all available cores)
      verbose: Verbosity level (see joblib.Parallel documentation)
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

    if force_zero_tilt:
        dipole_tilt = 0.0

    x_re_gsm, y_re_gsm, z_re_gsm = sm_to_gsm(x_re_sm, y_re_sm, z_re_sm,
                                             dipole_tilt)

    # Calculate the internal (dipole) and external (t96) fields using the
    # geopack module
    # ------------------------------------------------------------------------    
    # Use joblib to process in parallel using the number of processes and
    # verbosity settings specified by caller
    tasks = []

    for i in range(x_re_gsm.shape[0]):
        task = delayed(_tsyganenko_parallel_helper)(
            model_name, i, params, x_re_gsm[i], y_re_gsm[i], z_re_gsm[i], dipole_tilt
        )
        tasks.append(task)

    if verbose:
        print(f'Total number of tasks: {len(tasks)}')

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

    if external_field_only:
        B_t = gsm_to_sm(*B_external, dipole_tilt)
    else:
        B_t = gsm_to_sm(*(B_internal + B_external), dipole_tilt)

    B_t *= units.nT

    # Create PyVista structured grid.
    # ------------------------------------------------------------------------
    mesh = pyvista.StructuredGrid(X_re_sm_grid, Y_re_sm_grid, Z_re_sm_grid)

    B = np.empty((mesh.n_points, 3))
    B[:, 0] = B_t[0, :].to(units.G).value
    B[:, 1] = B_t[1, :].to(units.G).value
    B[:, 2] = B_t[2, :].to(units.G).value
    mesh.point_data['B'] = B

    # Return output
    return mesh


def get_t96_mesh_on_lfm_grid(dynamic_pressure, Dst, By_imf, Bz_imf,
                             lfm_hdf4_path, **kwargs):
    """Get a T96 field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      dynamic_pressure: Dynamic Pressure of Solar Wind (nPA); parameter of
        T96 Model
      Dst: Disturbance storm time index; parameter of T96 model
      By_imf: Y component of IMF Field (nT); parameter of T96 Model
      Bz_imf: Z component of IMF Field (nT); parameter of T96 Model
      lfm_hdf4_path: Path to LFM hdf4 file      
      time: Time for T89 model, sets parameters
      external_field_only: Set to True to not include the internal (dipole) model
      force_zero_tilt: Force a zero tilt when calculating the file
      n_jobs: Number of parallel processes to use (-1 for all available cores)
      verbose: Verbosity level (see joblib.Parallel documentation)
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    params = (dynamic_pressure, Dst, By_imf, Bz_imf, 0, 0, 0, 0, 0, 0)
    return _get_tsyganenko_on_lfm_grid('T96', params, lfm_hdf4_path, **kwargs)


def get_tsyganenko_on_lfm_grid_with_auto_params(model_name, time, lfm_hdf4_path,
                                                tell_params=True, __T_AUTO_DL_CACHE={},
                                                **kwargs):
    """Get a T96 field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      dynamic_pressure: Dynamic Pressure of Solar Wind (nPA); parameter of
        T96 Model
      Dst: Disturbance storm time index; parameter of T96 model
      By_imf: Y component of IMF Field (nT); parameter of T96 Model
      Bz_imf: Z component of IMF Field (nT); parameter of T96 Model
      lfm_hdf4_path: Path to LFM hdf4 file      
      time: Time for T89 model, sets parameters
      external_field_only: Set to True to not include the internal (dipole) model
      force_zero_tilt: Force a zero tilt when calculating the file
      n_jobs: Number of parallel processes to use (-1 for all available cores)
      verbose: Verbosity level (see joblib.Parallel documentation)
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    # Lookup data from internet ------------------------------------------------------------
    year = '%4d' % time.year
    month = '%02d' % time.month
    day = '%02d' % time.day

    url = (
        f'https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/{year}/'
        f'QinDenton_{year}{month}{day}_5min.txt'
    )

    if url in __T_AUTO_DL_CACHE:
        if tell_params:
            print(f'Getting {url} from cache')
        df = __T_AUTO_DL_CACHE[url]
    else:
        if tell_params:
            print(f'Downloading {url}')
        col_names = [
            'DateTime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
            'ByIMF', 'BzIMF', 'Vsw', 'Den_P', 'Pdyn', 
            'G1', 'G2', 'G3',
            'ByIMF_status', 'BzIMF_status', 'Vsw_status', 'Den_P_status', 'Pdyn_status',
            'G1_status', 'G2_status', 'G3_status',
            'Kp', 'akp3', 'Dst',
            'Bz1', 'Bz2', 'Bz3', 'Bz4', 'Bz5', 'Bz6', 
            'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 
            'W1_status', 'W2_status', 'W3_status', 'W4_status', 'W5_status', 'W6_status', 
        ]
        df = pd.read_csv(url, index_col=False, names=col_names, sep='\s+', comment='#')
        __T_AUTO_DL_CACHE[url] = df

    # Interpolate Tsyganenko parameters (some may be unused)
    cols = ['Pdyn', 'Dst', 'ByIMF', 'BzIMF', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    params_dict = {}

    for col in cols:
        params_dict[col] = np.interp(date2num(time), date2num(df.DateTime), df[col])

    params = list(params_dict.values())

    if tell_params:
        print(f'Looked up parameters: {params_dict}')

    if model_name == 'T96':
        for i in range(6):                   # set last six elements to zero
            params[len(params)- 1 - i] = 0

    # Call model using parameters
    return _get_tsyganenko_on_lfm_grid(model_name, params, lfm_hdf4_path, **kwargs)


def get_t04_mesh_on_lfm_grid(dynamic_pressure, Dst, By_imf, Bz_imf, W_values,
                             lfm_hdf4_path, **kwargs):
    """Get a T04/T05 field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Args
      dynamic_pressure: Dynamic Pressure of Solar Wind (nPA); parameter of
        T96 Model
      Dst: Disturbance storm time index; parameter of T96 model
      By_imf: Y component of IMF Field (nT); parameter of T96 Model
      Bz_imf: Z component of IMF Field (nT); parameter of T96 Model
      W_values: Tuple of 6 W values as defined by model. They can be obtained from 
      https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/2013/
      lfm_hdf4_path: Path to LFM hdf4 file      
      time: Time for T89 model, sets parameters
      external_field_only: Set to True to not include the internal (dipole) model
      force_zero_tilt: Force a zero tilt when calculating the file
      n_jobs: Number of parallel processes to use (-1 for all available cores)
      verbose: Verbosity level (see joblib.Parallel documentation)
    Returns
      mesh: pyvista.StrucutredGrid instance, mesh on LFM grid with dipole
        field values. Grid is in units of Re and magnetic field is is units of
        Gauss.
    """
    params = (dynamic_pressure, Dst, By_imf, Bz_imf,) + tuple(W_values)

    # T04 and T05 are the same model; the paper was published in 2005 but the code was
    # published in 2004 so they are called different things in different places.
    return _get_tsyganenko_on_lfm_grid('T04', params, lfm_hdf4_path, **kwargs)


def _tsyganenko_parallel_helper(model_name, i, params, x_re_gsm, y_re_gsm, 
                                z_re_gsm, dipole_tilt):
    internal_field_vec = geopack.dip(x_re_gsm, y_re_gsm, z_re_gsm)

    if model_name == 'T96':
        tsyganenko_func = t96.t96
    elif model_name == 'T04':
        tsyganenko_func = t04.t04
    else:
        raise RuntimeError(f"Unknown tsyganenko model {model_name}")
    
    external_field_vec = tsyganenko_func(
        params, dipole_tilt, x_re_gsm, y_re_gsm, z_re_gsm
    )

    return i, internal_field_vec, external_field_vec
