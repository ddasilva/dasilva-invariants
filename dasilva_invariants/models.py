"""Tools for obtaining models for use in calculating invariants. Models
are grids + magnetic field vectors at those grid points. They
are instances of :py:class:`~MagneticFieldModel`.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast, Dict, List, Tuple, Union

from ai import cs
from astropy import constants, units
import cdflib
import h5py
from matplotlib.dates import date2num
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pyhdf.SD import SD, SDC
import pyvista as pv
import vtk

from .constants import EARTH_DIPOLE_B0, LFM_INNER_BOUNDARY
from .utils import nanoTesla2Gauss
from ._fortran import geopack2008, t96, ts05  # type:ignore


__all__ = [
    "MagneticFieldModel",
    "FieldLineTrace",
    "get_dipole_model_on_lfm_grid",
    "get_lfm_hdf4_model",
    "get_tsyganenko",
    "get_tsyganenko_on_lfm_grid",
    "get_tsyganenko_params",
    "get_swmf_cdf_model",
    "get_generic_hdf5_model",
    "get_model",
]


@dataclass
class FieldLineTrace:
    """Class to hold the results of a field line trace.

    Parameters
    ----------
    points : array, shape (n, 3)
       Positions along field line trace, in SM coordinate system and units of Re
    B : array, shape (n, 3)
       Magnetic field vector along field line trace, in SM coordinates and units
       of Gauss
    """

    points: NDArray[np.float64]
    B: NDArray[np.float64]


class MagneticFieldModel:
    """Represents a magnetic field model, with methods for sampling the
    magnetic field at an aribtrary point.

    Attributes
    ----------
    x : array of (m, n, p)
        X coordinates of data, in SM coordiantes and units of Re
    y : array of (m, n, p)
        Y coordinates of data, in SM coordiantes and units of Re
    z : array of (m, n, p)
        Z coordinates of data, in SM coordiantes and units of Re
    Bx : array of (m, n, p)
        Magnetic field X component, in SM coordinates and units of Gauss
    By : array of (m, n, p)
        Magnetic field Y component, in SM coordinates and units of Gauss
    Bz : array of (m, n, p)
        Magnetic field Z component, in SM coordinates and units of Gauss
    inner_boundary : float
        Minimum radius to be considered too close to the earth for model to
        cover.
    """

    def __init__(self, x, y, z, Bx, By, Bz, inner_boundary):
        self.x = x
        self.y = y
        self.z = z
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.inner_boundary = inner_boundary

        B = np.empty((Bx.size, 3))
        B[:, 0] = Bx.flatten(order="F")
        B[:, 1] = By.flatten(order="F")
        B[:, 2] = Bz.flatten(order="F")
        self._mesh = pv.StructuredGrid(x, y, z)
        self._mesh.point_data["B"] = B

        R_grid, theta_grid, phi_grid = cs.cart2sp(x, y, z)
        self._mesh.point_data["R_grid"] = R_grid.flatten(order="F")
        self._mesh.point_data["Theta_grid"] = theta_grid.flatten(order="F")
        self._mesh.point_data["Phi_grid"] = phi_grid.flatten(order="F")

    def trace_field_line(
        self, starting_point, step_size
    ) -> FieldLineTrace:
        """Perform a field line trace. Implements RK45 in both directions,
        stopping when outside the grid.

        Parameters
        ----------
        starting_point : tuple of floats
            Starting point of the field line trace, as (x, y, z) tuple of
            floats, in units of Re. Trace will go in both directions until it hits
            the model inner or outer boundary.
        step_size : float, optional
            Step size to use with the field line trace. If not sure, try 1e-3.

        Returns
        -------
        trace : :py:class:`FieldLineTrace`
            Coordinates and magnetic field vector along the field line trace
        """
        pv_trace = self._mesh.streamlines(
            "B",
            start_position=starting_point,
            terminal_speed=0.0,
            max_step_length=step_size,
            min_step_length=step_size,
            initial_step_length=step_size,
            step_unit="l",
            max_steps=1_000_000,
            interpolator_type="c",
        )

        return FieldLineTrace(points=pv_trace.points, B=pv_trace["B"])

    def interpolate(self, point, radius=0.1):
        """Interpolate mesh to find magnetic field at given point.

        Uses a distance-weighted average of neighboring points.

        Parameters
        ----------
        point: tuple
           Position tuple of (x, y, z)

        Returns
        -------
        B : tuple
            Interpolated value of the mesh at given point.
        """
        points_search = pv.PolyData(np.array([point]))

        interp = vtk.vtkPointInterpolator()  # linear interpolation
        interp.SetInputData(points_search)
        interp.SetSourceData(self._mesh)
        interp.GetKernel().SetRadius(radius)        
        interp.Update()

        interp_result = pv.PolyData(interp.GetOutput())
        B = tuple(np.array(interp_result["B"])[0])
        B = cast(Tuple[float, float, float], B)

        return B


def _fix_lfm_hdf4_array_order(data):
    """Apply fix to LFM data to account for strange format in HDF4 file.

    This must be called on all data read from an LFM HDF4 file.

    Adapted from pyLTR, which has the following comment:
       Data is stored in C-order, but shape is transposed in
       Fortran order!  Why?  Because LFM stores its arrays using
       the rather odd A++/P++ library.  Let's reverse this evil:

    Arguments
    ---------
    data : NDArray[np.float64]
        Scalar field over model grid

    Returns
    --------
    data : NDArray[np.float64]
        Scalar field over model grid, with dimensions fixed
    """
    s = data.shape
    data = np.reshape(data.ravel(), (s[2], s[1], s[0]), order="F")
    return data


def get_dipole_model_on_lfm_grid(lfm_hdf4_path) -> MagneticFieldModel:
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Parameters
    ----------
    lfm_hdf4_path : str
        Path to LFM file in HDF4 format

    Returns
    -------
    model : :py:class:`~MagneticFieldModel`
        Mesh on LFM grid with dipole field values. Grid is in units of Re and
        magnetic field is is units of Gauss
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    X_grid, Y_grid, Z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

    # Calculate dipole model
    # ------------------------------------------------------------------------
    # Dipole model, per Kivelson and Russel equations 6.3(a)-(c), page 165.
    R_grid = np.sqrt(X_grid**2 + Y_grid**2 + Z_grid**2)

    Bx = 3 * X_grid * Z_grid * EARTH_DIPOLE_B0 / R_grid**5
    By = 3 * Y_grid * Z_grid * EARTH_DIPOLE_B0 / R_grid**5
    Bz = (3 * Z_grid**2 - R_grid**2) * EARTH_DIPOLE_B0 / R_grid**5

    # Create magnetic field model
    # ------------------------------------------------------------------------
    return MagneticFieldModel(
        X_grid, Y_grid, Z_grid, Bx, By, Bz, inner_boundary=LFM_INNER_BOUNDARY
    )


def _get_fixed_lfm_grid_centers(lfm_hdf4_path: str):
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
    X_grid_raw = _fix_lfm_hdf4_array_order(hdf.select("X_grid").get())
    Y_grid_raw = _fix_lfm_hdf4_array_order(hdf.select("Y_grid").get())
    Z_grid_raw = _fix_lfm_hdf4_array_order(hdf.select("Z_grid").get())

    # This code implements Josh Murphy's point2CellCenteredGrid() function
    # ------------------------------------------------------------------------
    ni = X_grid_raw.shape[0] - 1
    # nip1 = X_grid_raw.shape[0]
    # nip2 = X_grid_raw.shape[0] + 1

    nj = X_grid_raw.shape[1] - 1
    njp1 = X_grid_raw.shape[1]
    njp2 = X_grid_raw.shape[1] + 1

    nk = X_grid_raw.shape[2] - 1
    nkp1 = X_grid_raw.shape[2]
    # nkp2 = X_grid_raw.shape[2] + 1

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
    # ignore typing here because MyPy doesn't work well with astropy
    X_grid_re = (X_grid * units.cm).to(constants.R_earth).value  # type: ignore
    Y_grid_re = (Y_grid * units.cm).to(constants.R_earth).value  # type: ignore
    Z_grid_re = (Z_grid * units.cm).to(constants.R_earth).value  # type: ignore

    return X_grid_re, Y_grid_re, Z_grid_re

        
def get_lfm_hdf4_model(lfm_hdf4_path) -> MagneticFieldModel:
    """Get a magnetic field data + grid from LFM output. Uses an LFM HDF4 file.

    Parameters
    -----------
    lfm_hdf4_path : str
        Path to LFM file in HDF4 format

    Returns
    --------
    model : :py:class:`~MagneticFieldModel`
        Mesh on LFM grid with LFM magnetic field values. Grid is in units of Re
        and magnetic field is is units of Gauss
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    X_grid, Y_grid, Z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

    # Read LFM B values from HDF file
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)

    Bx_raw = _fix_lfm_hdf4_array_order(hdf.select("bx_").get())
    By_raw = _fix_lfm_hdf4_array_order(hdf.select("by_").get())
    Bz_raw = _fix_lfm_hdf4_array_order(hdf.select("bz_").get())

    # Bx, By, Bz = Bx_raw, By_raw, Bz_raw
    Bx, By, Bz = _apply_murphy_lfm_grid_patch(Bx_raw, By_raw, Bz_raw)

    # Create Magnetic Field Model
    # ------------------------------------------------------------------------
    return MagneticFieldModel(
        X_grid,
        Y_grid,
        Z_grid,
        Bx,
        By,
        Bz,
        inner_boundary=LFM_INNER_BOUNDARY,
    )


def _apply_murphy_lfm_grid_patch(Bx_raw, By_raw, Bz_raw):
    """Apply Josh Murphy's patch to the LFM grid.

    This code is Josh Murphy's point2CellCenteredVector() function converted
    to python.

    Args
     Bx_raw, By_raw, Bz_raw: The magnetic field in the raw grid.
    Returns
     Bx, By, Bz: The magnetic field in the patched grid
    """
    ni = Bx_raw.shape[0] - 1
    # nip1 = Bx_raw.shape[0]
    # nip2 = Bx_raw.shape[0] + 1

    nj = Bx_raw.shape[1] - 1
    njp1 = Bx_raw.shape[1]
    njp2 = Bx_raw.shape[1] + 1

    nk = Bx_raw.shape[2] - 1
    nkp1 = Bx_raw.shape[2]
    # nkp2 = Bx_raw.shape[2] + 1

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

    Parameters
    ----------
    A : NDArray[np.float64]
        3D grid holding grid positions on one of X, Y or Z for each grid
        coordinate.

    Returns
    -------
    centers : NDArray[np.float64]
        3D array of X, Y, or Z positions for grid coordinates
    """
    s = A.shape

    centers = np.zeros((s[0] - 1, s[1] - 1, s[2] - 1))

    centers += A[:-1, :-1, :-1]  # i,   j,   k

    centers += A[1:, :-1, :-1]  # i+1, j,   k
    centers += A[:-1, 1:, :-1]  # i,   j+1, k
    centers += A[:-1, :-1, 1:]  # i,   j,   k+1

    centers += A[1:, 1:, :-1]  # i+1, j+1, k
    centers += A[:-1, 1:, 1:]  # i,   j+1, k+1
    centers += A[1:, :-1, 1:]  # i+1, j,   k+1

    centers += A[1:, 1:, 1:]  # i+1, j+1, k+1

    centers /= 8.0

    return centers


def get_tsyganenko(
        model_name: str,
        params: Dict[str, NDArray[np.float64]],
        time: datetime,
        x_re_sm_grid: NDArray[np.float64],
        y_re_sm_grid: NDArray[np.float64],
        z_re_sm_grid: NDArray[np.float64],
        inner_boundary: float,
        external_field_only: bool = False,
) -> MagneticFieldModel:
    """Internal helper function to get one of the tsyganenko fields on an LFM grid.

    Parameters
    -----------
    model_name : {'T96', 'TS05'}
        Name of the magnetic field model to use.
    params : dictionary of string to array
        Parameters to support Tsyganenko magnetic field mode
    time : datetime, no timezone
        Time to support the Tsyganenko magnetic field model
    x_re_sm_grid : array of shame (m, n, p)
        x coordinates
    y_re_sm_grid : array of shame (m, n, p)
        y coordinates
    z_re_sm_grid : array of shame (m, n, p)
        z coordinates
    inner_boundary : float
        Inner boundary of model
    external_field_only : bool
        Set to True to not include the internal (dipole) model

    Returns
    -------
    model : :py:class:`~MagneticFieldModel`
        Magnetic model on LFM grid with dipole field values. Grid is in units of
        Re and magnetic field is is units of Gauss.
    """
    x_re_sm = x_re_sm_grid.flatten()  # flat arrays, easier for later
    y_re_sm = y_re_sm_grid.flatten()
    z_re_sm = z_re_sm_grid.flatten()

    # Call compiled Tsyganenko fortran code  to get extenral field
    parmod = (
        params["Pdyn"],
        params["dst"],
        params["By"],
        params["Bz"]
    ) + tuple([params[f"W{i+1}"] for i in range(6)])

    time_tup = (
        time.year,
        int(time.strftime("%j")),
        time.hour,
        time.minute,
        time.second,
    )

    geopack2008.recalc(time_tup, (-400, 0.0, 0.0))

    if model_name.lower() == "t96":
        Bx, By, Bz = t96.t96numpy(parmod, 0.0, x_re_sm, y_re_sm, z_re_sm)
    elif model_name.lower() == "ts05":
        Bx, By, Bz = ts05.ts05numpy(parmod, 0.0, x_re_sm, y_re_sm, z_re_sm)
    else:
        raise ValueError(f"Invalid parameter model_name={repr(model_name)}")

    # Calculate dipole field for internal model,
    if not external_field_only:
        Bx_dip, By_dip, Bz_dip = geopack2008.dipnumpy(x_re_sm, y_re_sm, z_re_sm)
        Bx += Bx_dip
        By += By_dip
        Bz += Bz_dip

    # NaN out points under inner boundary
    r_re_sm = np.sqrt(x_re_sm**2 + y_re_sm**2 + z_re_sm**2)
    Bx[r_re_sm < inner_boundary] = np.nan
    By[r_re_sm < inner_boundary] = np.nan
    Bz[r_re_sm < inner_boundary] = np.nan
    
    # Convert from nT to Gauss
    Bx = nanoTesla2Gauss(Bx)
    By = nanoTesla2Gauss(By)
    Bz = nanoTesla2Gauss(Bz)

    # Create magnetic field model
    shape = x_re_sm_grid.shape
    Bx = Bx.reshape(shape)
    By = By.reshape(shape)
    Bz = Bz.reshape(shape)
    
    return MagneticFieldModel(
        x_re_sm_grid,
        y_re_sm_grid,
        z_re_sm_grid,
        Bx,
        By,
        Bz,
        inner_boundary=inner_boundary
    )


def get_tsyganenko_on_lfm_grid(
    model_name: str,
    params: Dict[str, NDArray[np.float64]],
    time: datetime,
    lfm_hdf4_path: str,
    external_field_only: bool = False,
) -> MagneticFieldModel:
    """Internal helper function to get one of the tsyganenko fields on an LFM grid.

    Parameters
    -----------
    model_name : {'T96', 'TS05'}
        Name of the magnetic field model to use.
    params : dictionary of string to array
        Parameters to support Tsyganenko magnetic field mode
    time : datetime, no timezone
        Time to support the Tsyganenko magnetic field model
    lfm_hdf4_path : str
        Path to LFM file in HDF4 format to provide grid.
    external_field_only : bool
        Set to True to not include the internal (dipole) model

    Returns
    -------
     model : :py:class:`~MagneticFieldModel`
        Magnetic model on LFM grid with dipole field values. Grid is in units of
        Re and magnetic field is is units of Gauss.
    """

    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    x_re_sm_grid, y_re_sm_grid, z_re_sm_grid = _get_fixed_lfm_grid_centers(
        lfm_hdf4_path
    )
    
    return get_tsyganenko(
        model_name,
        params,
        time,
        x_re_sm_grid,
        y_re_sm_grid,
        z_re_sm_grid,
        inner_boundary=LFM_INNER_BOUNDARY,
        external_field_only=external_field_only,
    )


def get_tsyganenko_params(
    times: Union[Sequence, datetime],
    path: str,
    skip_cache: bool = False,
    __T_AUTO_DL_CACHE: Dict[str, pd.DataFrame] = {},
) -> Dict[str, NDArray[np.float64]]:
    """Get parameters for tsyganenko models.

    Parameters
    -----------
    times : datetime or list of datetime (no timezones)
        Time(s) to get paramters for.
    path : str
        Path to zip file (may be URL if network enabled). It is fastest
        to download this file and save it to disk, but this URL may be passed
        automatically to download every time
        http://mag.gmu.edu/ftp/QinDenton/5min/merged/latest/WGparameters5min-latest.d.zip

    Returns
    -------
    params : dict, str to array
        dictionary mapping variable to array of parameters
    """
    times_list: List[datetime] = []

    try:
        iter(times)  # type: ignore
        times_list = times  # type: ignore
    except TypeError:
        assert isinstance(times, datetime)
        times_list = [times]

    if path in __T_AUTO_DL_CACHE and not skip_cache:
        df = __T_AUTO_DL_CACHE[path]
    else:
        # ignore typing here because pandas broken
        df = pd.read_csv(
            path,
            index_col=False,
            delim_whitespace=True,
            skiprows=1,
            names=[  # type:ignore
                "Year",
                "Day",
                "Hr",
                "Min",
                "By",
                "Bz",
                "V_SW",
                "Den_P",
                "Pdyn",
                "G1",
                "G2",
                "G3",
                "8_status",
                "kp",
                "akp3",
                "dst",
                "Bz1",
                "Bz2",
                "Bz3",
                "Bz4",
                "Bz5",
                "Bz6",
                "W1",
                "W2",
                "W3",
                "W4",
                "W5",
                "W6",
                "6_stat",
            ],
        )

        df = cast(pd.DataFrame, df)
        __T_AUTO_DL_CACHE[path] = df

    min_year = min(time.year for time in times_list)
    max_year = max(time.year for time in times_list)

    mask = (df["Year"] > (min_year - 1)) & (df["Year"] < (max_year + 1))
    df = df[mask].copy()
    df["DateTime"] = [
        datetime(int(row.Year), 1, 1)
        + timedelta(days=row.Day - 1, hours=row.Hr, minutes=row.Min)
        for _, row in df.iterrows()
    ]

    # Interpolate Tsyganenko parameters (some may be unused)
    cols = ["Pdyn", "dst", "By", "Bz", "W1", "W2", "W3", "W4", "W5", "W6"]
    params_dict = {}

    for col in cols:
        if len(times_list) == 1:
            (params_dict[col],) = np.interp(
                date2num(times_list), date2num(df.DateTime), df[col]
            )
        else:
            params_dict[col] = np.interp(
                date2num(times_list), date2num(df.DateTime), df[col]
            )

    return params_dict


def get_swmf_cdf_model(
    path: str,
    xaxis: NDArray[np.float64] =np.arange(-10, 10, .15),
    yaxis: NDArray[np.float64] = np.arange(-10, 10, .15),
    zaxis: NDArray[np.float64] = np.arange(-5, 5, .15)
):
    """Get a magnetic field data + grid from SWMF CDF output. This regrids it
    to a rectilinear grid.

    Parameters
    -----------
    path : str
        Path to SWMF file in CDF format
    xaxis: array
        x-axis of rectilinear grid (default -10:.15:10)
    yaxis: array
        y-axis of rectilinear grid (default -10:.15:10)
    zaxis: array
        z-axis of rectilinear grid  (default -5:.15:5)

    Returns
    --------
    model : :py:class:`~MagneticFieldModel`
        Data on rectilinear grid with SWMF magnetic field values. Grid is in units 
        of Re and magnetic field is is units of Gauss
    """    
    # Load data from CDF
    cdf = cdflib.CDF(path)
    
    x = cdf.varget('x').flatten()
    y = cdf.varget('y').flatten()
    z = cdf.varget('z').flatten()
    bx = nanoTesla2Gauss(cdf.varget('bx').flatten())
    by = nanoTesla2Gauss(cdf.varget('by').flatten())
    bz = nanoTesla2Gauss(cdf.varget('bz').flatten())
    
    # Calculate Dipole (data in file is external field)
    r = np.sqrt(x**2 + y**2 + z**2)
    bx_dipole = 3 * x * z * EARTH_DIPOLE_B0 / r**5
    by_dipole = 3 * y * z * EARTH_DIPOLE_B0 / r**5
    bz_dipole = (3 * z**2 - r**2) * EARTH_DIPOLE_B0 / r**5    
    
    # Interpolate onto rectilinear grid
    X, Y, Z = np.meshgrid(xaxis, yaxis, zaxis)

    point_cloud = pv.PolyData(np.transpose([x, y, z]))
    point_cloud['Bx'] = bx + bx_dipole
    point_cloud['By'] = by + by_dipole
    point_cloud['Bz'] = bz + bz_dipole

    points_search = pv.PolyData(np.transpose([X.flatten(), Y.flatten(), Z.flatten()]))
    interp = vtk.vtkPointInterpolator()  # linear interpolation
    interp.SetInputData(points_search)
    interp.SetSourceData(point_cloud)
    interp.GetKernel().SetRadius(0.1)
    interp.Update()

    interp_result = pv.PolyData(interp.GetOutput())
    
    # Make MagneticFieldModel
    x_grid = interp_result.points[:, 0].reshape(X.shape)
    y_grid = interp_result.points[:, 1].reshape(X.shape)
    z_grid = interp_result.points[:, 2].reshape(X.shape)
    r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

    Bx = interp_result['Bx'].reshape(X.shape)
    By = interp_result['By'].reshape(X.shape)
    Bz = interp_result['Bz'].reshape(X.shape)

    inner_bdy = LFM_INNER_BOUNDARY
    mask = r_grid < inner_bdy
    Bx[mask] = np.nan
    By[mask] = np.nan
    Bz[mask] = np.nan    

    return MagneticFieldModel(
        x_grid, y_grid, z_grid, Bx, By, Bz,
        inner_boundary=inner_bdy
    )


def get_generic_hdf5_model(path):
    """Load a :py:class:`~MagneticFieldModel` from a generic HDF5 file.
    
    This is meant to plug in your own data.
    
    The file should have (m, n, p) arrays named "x", "y", "z",
    "Bx", "By", "Bz", and a scalar key named "inner_boundary".

    
    Parameters
    ----------
    path : str
       Path to file on disk

    Returns
    --------
    model : :py:class:`~MagneticFieldModel`
       Grid and Magnetic field values on that grid.
    """
    hdf = h5py.File(path)
    x = hdf['x'][:]
    y = hdf['y'][:]
    z = hdf['z'][:]
    Bx = hdf['Bx'][:]
    By = hdf['By'][:]
    Bz = hdf['Bz'][:]
    inner_boundary = hdf['inner_boundary'][()]
    hdf.close()

    return MagneticFieldModel(
        x, y, z, Bx, By, Bz, inner_boundary
    )


def get_model(model_type, path, **kwargs):
    """Get a magnetic field mmodel_type, pathodel;

    For specific keyword arguments see other functions in this model
    that this common functions calls.

    Parameters
    ----------
    model_type : {"lfm_hdf4", "swmf_cdf", "generic_hdf5"}
       Type of the model (case insensitive)
    path : str
       Path to file on disk

    Returns
    --------
    model : :py:class:`~MagneticFieldModel`
       Grid and Magnetic field values on that grid.
    """
    model_type = model_type.lower()
    
    if model_type == "lfm_hdf4":
        return get_lfm_hdf4_model(path)
    elif model_type == "swmf_cdf":
        return get_swmf_cdf_model(path, **kwargs)
    elif model_type == "generic_hdf5":
        return get_generic_hdf5_model(path, **kwargs)
    else:
        raise TypeError(
            f"Unknown model type {repr(model_type)}"
        )
