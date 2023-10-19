"""Tools for obtaining meshes for use in calculating invariants.

Meshes are grids + magnetic field vectors at those grid points. They
are instances of :py:class:`~MagneticFieldModel`.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from typing import cast, Dict, List, Tuple, Union

from ai import cs
from astropy import constants, units
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from matplotlib.dates import date2num
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pyhdf.SD import SD, SDC
import pyvista
import vtk

from .constants import EARTH_DIPOLE_B0, LFM_INNER_BOUNDARY
from .utils import nanoTesla2Gauss
from ._fortran import _geopack2008, _t96, _ts05  # type:ignore


__all__ = [
    "MagneticFieldModel",
    "FieldLineTrace",
    "get_dipole_mesh_on_lfm_grid",
    "get_lfm_hdf4_data",
    "get_tsyganenko_on_lfm_grid_with_auto_params",
    "get_tsyganenko_params",
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

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        Bx: NDArray[np.float64],
        By: NDArray[np.float64],
        Bz: NDArray[np.float64],
        inner_boundary: float,
    ):
        self.x = x.copy()
        self.y = y.copy()
        self.z = z.copy()
        self.Bx = Bx.copy()
        self.By = By.copy()
        self.Bz = Bz.copy()
        self.inner_boundary = inner_boundary

        B = np.empty((Bx.size, 3))
        B[:, 0] = Bx.flatten(order="F")
        B[:, 1] = By.flatten(order="F")
        B[:, 2] = Bz.flatten(order="F")
        self._mesh = pyvista.StructuredGrid(x, y, z)
        self._mesh.point_data["B"] = B

        R_grid, theta_grid, phi_grid = cs.cart2sp(x, y, z)
        self._mesh.point_data["R_grid"] = R_grid.flatten(order="F")
        self._mesh.point_data["Theta_grid"] = theta_grid.flatten(order="F")
        self._mesh.point_data["Phi_grid"] = phi_grid.flatten(order="F")

    def trace_field_line(
        self, starting_point: Tuple[float, float, float], step_size: float = 1e-3
    ) -> FieldLineTrace:
        """Perform a field line trace. Implements RK45 in both directions,
        stopping when outside the grid.

        Parameters
        ----------
        starting_point : tuple of floats
            Starting point of the field line trace, as (x, y, z) tuple of
            floats, in units of Re. Trace will go in both directions.
        step_size : float, optional
            Step size to use with the field line trace

        Returns
        -------
        trace : :py:class:`FieldLineTrace`
            Coordinates and magnetic field vector along the field line trace
        """
        pyvista_trace = self._mesh.streamlines(
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

        return FieldLineTrace(points=pyvista_trace.points, B=pyvista_trace["B"])

    def interpolate(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Linearly interpolate mesh to find magnetic field at given point.

        Parameters
        ----------
        point: tuple of floats
           Position tuple of (x, y, z)

        Returns
        -------
        B : tuple of floats
            Interpolated value of the mesh at given point.
        """
        points_search = pyvista.PolyData(np.array([point]))

        interp = vtk.vtkPointInterpolator()  # linear interpolation
        interp.SetInputData(points_search)
        interp.SetSourceData(self._mesh)
        interp.Update()

        interp_result = pyvista.PolyData(interp.GetOutput())
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


def get_dipole_mesh_on_grid(x_grid, y_grid, z_grid, inner_boundary):
    """Get the dipole field on an arbitrary grid.

    Parameters
    ----------
    x_grid : array of (m, n, p)
        X coordinates of data, in SM coordiantes and units of Re
    y_grid : array of (m, n, p)
        Y coordinates of data, in SM coordiantes and units of Re
    z_grid : array of (m, n, p)
        Z coordinates of data, in SM coordiantes and units of Re
    inner_boundary : float
        Minimum radius to be considered too close to the earth for model to
        cover.

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        Mesh on provided grid with dipole field values. Grid is in units
        of Re and magnetic field is is units of Gauss.
    """
    # Calculate dipole model
    # ------------------------------------------------------------------------
    # Dipole model, per Kivelson and Russel equations 6.3(a)-(c), page 165.
    R_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

    Bx = 3 * x_grid * z_grid * EARTH_DIPOLE_B0 / R_grid**5
    By = 3 * y_grid * z_grid * EARTH_DIPOLE_B0 / R_grid**5
    Bz = (3 * z_grid**2 - R_grid**2) * EARTH_DIPOLE_B0 / R_grid**5

    # Create magnetic field model
    # ------------------------------------------------------------------------
    return MagneticFieldModel(
        x_grid, y_grid, z_grid, Bx, By, Bz, inner_boundary=inner_boundary
    )


def get_dipole_mesh_on_lfm_grid(lfm_hdf4_path: str) -> MagneticFieldModel:
    """Get a dipole field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Parameters
    ----------
    lfm_hdf4_path : str
        Path to LFM file in HDF4 format

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        Mesh on LFM grid with dipole field values. Grid is in units of Re and
        magnetic field is is units of Gauss
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    x_grid, y_grid, z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

    return get_dipole_mesh_on_grid(x_grid, y_grid, z_grid, LFM_INNER_BOUNDARY)


def _get_fixed_lfm_grid_centers(
    lfm_hdf4_path: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Loads LFM grid centers with singularity patched.

    This code is adapted from Josh Murphy's GhostPy and converted to python
    (crudely).

    Args
      lfm_hdf4_path: Path to LFM HDF4 file
    Returns
      x_grid, y_grid, z_grid: arrays of LFM grid cell centers in units of Re
        with the x-axis singularity issue fixed.
    """
    # Read LFM grid from HDF file
    # ------------------------------------------------------------------------
    hdf = SD(lfm_hdf4_path, SDC.READ)
    x_grid_raw = _fix_lfm_hdf4_array_order(hdf.select("X_grid").get())
    y_grid_raw = _fix_lfm_hdf4_array_order(hdf.select("Y_grid").get())
    z_grid_raw = _fix_lfm_hdf4_array_order(hdf.select("Z_grid").get())

    # This code implements Josh Murphy's point2CellCenteredGrid() function
    # ------------------------------------------------------------------------
    ni = x_grid_raw.shape[0] - 1
    # nip1 = x_grid_raw.shape[0]
    # nip2 = x_grid_raw.shape[0] + 1

    nj = x_grid_raw.shape[1] - 1
    njp1 = x_grid_raw.shape[1]
    njp2 = x_grid_raw.shape[1] + 1

    nk = x_grid_raw.shape[2] - 1
    nkp1 = x_grid_raw.shape[2]
    # nkp2 = x_grid_raw.shape[2] + 1

    x_grid = np.zeros((ni, njp2, nkp1))
    y_grid = np.zeros((ni, njp2, nkp1))
    z_grid = np.zeros((ni, njp2, nkp1))

    x_grid[:, 1:-1, :-1] = _calc_cell_centers(x_grid_raw)
    y_grid[:, 1:-1, :-1] = _calc_cell_centers(y_grid_raw)
    z_grid[:, 1:-1, :-1] = _calc_cell_centers(z_grid_raw)

    for j in range(0, njp2, njp1):
        jAxis = max(1, min(nj, j))
        for i in range(ni):
            x_grid[i, j, :-1] = x_grid[i, jAxis, :-1].mean()
            y_grid[i, j, :-1] = y_grid[i, jAxis, :-1].mean()
            z_grid[i, j, :-1] = z_grid[i, jAxis, :-1].mean()

    x_grid[:, :, nk] = x_grid[:, :, 0]
    y_grid[:, :, nk] = y_grid[:, :, 0]
    z_grid[:, :, nk] = z_grid[:, :, 0]

    # Convert to units of earth radii (Re)
    # ------------------------------------------------------------------------
    # ignore typing here because MyPy doesn't work well with astropy
    x_grid_re = (x_grid * units.cm).to(constants.R_earth).value  # type: ignore
    y_grid_re = (y_grid * units.cm).to(constants.R_earth).value  # type: ignore
    z_grid_re = (z_grid * units.cm).to(constants.R_earth).value  # type: ignore

    return x_grid_re, y_grid_re, z_grid_re


def get_lfm_hdf4_data(lfm_hdf4_path: str) -> MagneticFieldModel:
    """Get a magnetic field data + grid from LFM output. Uses an LFM HDF4 file.

    Parameters
    -----------
    lfm_hdf4_path : str
        Path to LFM file in HDF4 format

    Returns
    --------
    mesh : :py:class:`~MagneticFieldModel`
        Mesh on LFM grid with LFM magnetic field values. Grid is in units of Re
        and magnetic field is is units of Gauss
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    x_grid, y_grid, z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

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
        x_grid,
        y_grid,
        z_grid,
        Bx,
        By,
        Bz,
        inner_boundary=LFM_INNER_BOUNDARY,
    )


def _apply_murphy_lfm_grid_patch(
    Bx_raw: NDArray[np.float64],
    By_raw: NDArray[np.float64],
    Bz_raw: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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


def _calc_cell_centers(A: NDArray[np.float64]) -> NDArray[np.float64]:
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


def _get_tsyganenko_on_lfm_grid(
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
    external_field_only : boopl
        Set to True to not include the internal (dipole) model

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        Magnetic model on LFM grid with dipole field values. Grid is in units of
        Re and magnetic field is is units of Gauss.
    """

    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    X_re_sm_grid, Y_re_sm_grid, Z_re_sm_grid = _get_fixed_lfm_grid_centers(
        lfm_hdf4_path
    )

    x_re_sm = X_re_sm_grid.flatten()  # flat arrays, easier for later
    y_re_sm = Y_re_sm_grid.flatten()
    z_re_sm = Z_re_sm_grid.flatten()

    # Call compiled Tsyganenko fortran code  to get extenral field
    # ------------------------------------------------------------------------
    parmod = (params["Pdyn"], params["dst"], params["By"], params["Bz"]) + tuple(
        [params[f"W{i+1}"] for i in range(6)]
    )
    time_tup = (
        time.year,
        int(time.strftime("%j")),
        time.hour,
        time.minute,
        time.second,
    )

    _geopack2008.recalc(time_tup, (-400, 0.0, 0.0))

    if model_name == "T96":
        Bx, By, Bz = _t96.t96numpy(parmod, 0.0, x_re_sm, y_re_sm, z_re_sm)
    elif model_name == "TS05":
        Bx, By, Bz = _ts05.ts05numpy(parmod, 0.0, x_re_sm, y_re_sm, z_re_sm)
    else:
        raise ValueError(f"Invalid parameter model_name={repr(model_name)}")

    # Calculate dipole field for internal model,
    # ----------------------------------------------------------------------
    if not external_field_only:
        Bx_dip, By_dip, Bz_dip = _geopack2008.dipnumpy(x_re_sm, y_re_sm, z_re_sm)
        Bx += Bx_dip
        By += By_dip
        Bz += Bz_dip

    # Convert from nT to Gauss
    # ------------------------------------------------------------------------
    Bx = nanoTesla2Gauss(Bx)
    By = nanoTesla2Gauss(By)
    Bz = nanoTesla2Gauss(Bz)

    # Create magnetic field model
    # ------------------------------------------------------------------------
    shape = X_re_sm_grid.shape
    Bx = Bx.reshape(shape)
    By = By.reshape(shape)
    Bz = Bz.reshape(shape)

    return MagneticFieldModel(
        X_re_sm_grid,
        Y_re_sm_grid,
        Z_re_sm_grid,
        Bx,
        By,
        Bz,
        inner_boundary=LFM_INNER_BOUNDARY,
    )


def get_tsyganenko_params(
    times: Union[Sequence, datetime],
    path: str,
    tell_params: bool = True,
    __T_AUTO_DL_CACHE: Dict[str, pd.DataFrame] = {},
) -> Dict[str, NDArray[np.float64]]:
    """Get parameters for tsyganenko models.

    Parameters
    -----------
    times : List of datetime, no timezones
        Times to get paramters for
    path : str
        Path to zip file (may be URL if network enabled). It is fastest
        to download this file and save it to disk, but this URL may be passed
        automatically to download every time
        http://virbo.org/ftp/QinDenton/hour/merged/latest/WGhour-latest.d.zip
    tell_params : bool, optional
        If set to true, prints parameters to output

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

    if path in __T_AUTO_DL_CACHE:
        if tell_params:
            print(f"Getting {path} from cache")
        df = __T_AUTO_DL_CACHE[path]
    else:
        if tell_params:
            print(f"Loading {path}")
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

    if tell_params:
        print(params_dict)

    return params_dict


def get_tsyganenko_on_lfm_grid_with_auto_params(
    model_name: str,
    time: datetime,
    lfm_hdf4_path: str,
    param_path: str,
    tell_params: bool = True,
    **kwargs,
) -> MagneticFieldModel:
    """Get a Tsyganenko field on a LFM grid. Uses an LFM HDF4 file to obtain
    the grid.

    Parameters
    ----------
    model_name : {'T96', 'TS05'}
        Name of the magnetic field model to use.
    time : datetime, no timezone
        Time to support the Tsyganenko magnetic field model
    lfm_hdf4_path : str
        Path to LFM file in HDF4 format to provide grid.
    params_path : str
        Path to OMNI records file, can be URL to download. This file can be
        downloaded from http://virbo.org/ftp/QinDenton/hour/merged/latest/WGhour-latest.d.zip
    tell_params : bool, optional
        Print OMNI paramteters retreived to standard output
    external_field_only : boo, optional
        Set to True to not include the internal (dipole) model
    force_zero_tilt : bool, optional
        Force a zero tilt when calculating the magnetic field
    n_jobs : int, optional
        Number of parallel processes to use (-1 for all available cores)

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        Tsyganenko magnetic field model on LFM grid. Grid is in units of Re and magnetic
        field is is units of Gauss.
    """
    # Lookup params  -----------------------------------------------
    params = get_tsyganenko_params(time, param_path, tell_params=tell_params)

    # Call model using parameters
    return _get_tsyganenko_on_lfm_grid(
        model_name, params, time, lfm_hdf4_path, **kwargs
    )


def get_igrf_on_grid(time: datetime, x_grid, y_grid, z_grid, inner_boundary):
    """Evaluate the IGRF model on an arbitrary grid.

    Parameters
    ----------
    time : datetime
        Time to evaluate IGRF at.
    x_grid : array of (m, n, p)
        X coordinates of data, in SM coordiantes and units of Re
    y_grid : array of (m, n, p)
        Y coordinates of data, in SM coordiantes and units of Re
    z_grid : array of (m, n, p)
        Z coordinates of data, in SM coordiantes and units of Re
    inner_boundary : float
        Minimum radius to be considered too close to the earth for model to
        cover.

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        mesh with provided grid and IGRF field values. Grid is in units
        of Re and magnetic field is is units of Gauss.
    """
    # Call IGRF code
    # ------------------------------------------------------------------------
    time_tup = (
        time.year,
        int(time.strftime("%j")),
        time.hour,
        time.minute,
        time.second,
    )
    _geopack2008.recalc(time_tup, (-400, 0.0, 0.0))

    Bx, By, Bz = _geopack2008.igrfnumpy(
        x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    )

    shape = x_grid.shape
    Bx = nanoTesla2Gauss(Bx).reshape(shape)
    By = nanoTesla2Gauss(By).reshape(shape)
    Bz = nanoTesla2Gauss(Bz).reshape(shape)

    # Create magnetic field model
    # ------------------------------------------------------------------------
    return MagneticFieldModel(
        x_grid, y_grid, z_grid, Bx, By, Bz, inner_boundary=inner_boundary
    )


def get_igrf_on_lfm_grid(time: datetime, lfm_hdf4_path: str) -> MagneticFieldModel:
    """Calculate the IGRF magnetic field on an LFM grid. Uses an LFM HDF4 file
    to obtain the grid.

    Parameters
    ----------
    time : datetime
      Time to evaluate IGRF at
    lfm_hdf4_path : str
      Path to LFM file in HDF4 format

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        Mesh on LFM grid with IGRF field values. Grid is in units of Re and
        magnetic field is is units of Gauss
    """
    # Load LFM grid centers with singularity patched
    # ------------------------------------------------------------------------
    x_grid, y_grid, z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

    return get_igrf_on_grid(time, x_grid, y_grid, z_grid, LFM_INNER_BOUNDARY)


def get_igrf_on_rectangular_grid(
    time: datetime,
    x_range: Tuple[float, float] = (-LFM_INNER_BOUNDARY, LFM_INNER_BOUNDARY),
    y_range: Tuple[float, float] = (-LFM_INNER_BOUNDARY, LFM_INNER_BOUNDARY),
    z_range: Tuple[float, float] = (-LFM_INNER_BOUNDARY, LFM_INNER_BOUNDARY),
    nx: int = 128,
    ny: int = 128,
    nz: int = 128,
    range_padding: float = 0,
    inner_boundary: float = 1.0,
):
    """Evaluate the IGRF model on a rectangular grid.

    Parameters
    ----------
    time : datetime
        Time to evaluate IGRF at.
    x_range : tuple of (x_start, x_end)
        X-axis limits of the rectangular grid.
    y_range : tuple of (y_start, y_end)
        Y-axis limits of the rectangular grid.
    Z_range : tuple of (z_start, z_end)
        Z-axis limits of the rectangular grid.
    nx : int
        Number of x-axis points spanning the provided x_range.
    ny : int
        Number of y-axis points spanning the provided y_range.
    nz : int
        Number of z-axis points spanning the provided z_range.
    range_paadding : float
        Grows `x_range`, `y_range`, `z_range` by this amount.
    inner_boundary : float
        Inner boundary where field line traces should end.

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        mesh with provided grid and IGRF field values. Grid is in units
        of Re and magnetic field is is units of Gauss.
    """
    # Prepare grid
    x_axis = np.linspace(x_range[0] - range_padding, x_range[1] + range_padding, nx)
    y_axis = np.linspace(y_range[0] - range_padding, y_range[1] + range_padding, ny)

    z_axis = np.linspace(z_range[0] - range_padding, z_range[1] + range_padding, nz)

    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    # Call IGRF code
    # ------------------------------------------------------------------------
    time_tup = (
        time.year,
        int(time.strftime("%j")),
        time.hour,
        time.minute,
        time.second,
    )
    _geopack2008.recalc(time_tup, (-400, 0.0, 0.0))

    Bx, By, Bz = _geopack2008.igrfnumpy(
        x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    )

    shape = x_grid.shape
    Bx = nanoTesla2Gauss(Bx).reshape(shape)
    By = nanoTesla2Gauss(By).reshape(shape)
    Bz = nanoTesla2Gauss(Bz).reshape(shape)

    radius = np.linalg.norm([x_grid, y_grid, z_grid], axis=0)
    Bx[radius < inner_boundary] = np.nan
    By[radius < inner_boundary] = np.nan
    Bz[radius < inner_boundary] = np.nan

    # Return MagneticFieldModel
    return MagneticFieldModel(
        x_grid, y_grid, z_grid, Bx, By, Bz, inner_boundary=inner_boundary
    )


def sub_lfm_dipole_for_igrf(lfm_mesh, time):
    """Substitute the internal dipole field model of an LFM mesh with
    IGRF.

    lfm_mesh : :py:class:`~MagneticFieldModel`
      Mesh containning LFM field data on the LFM grid.
    time : datetime
      Time to evaluate IGRF at

    Returns
    -------
    mesh : :py:class:`~MagneticFieldModel`
        Mesh on same grid with dipole subtracted and replaced with IGRF
        internal fields.
    """
    dip_mesh = get_dipole_mesh_on_grid(
        lfm_mesh.x, lfm_mesh.y, lfm_mesh.z, lfm_mesh.inner_boundary
    )
    igrf_mesh = get_igrf_on_grid(
        time, lfm_mesh.x, lfm_mesh.y, lfm_mesh.z, lfm_mesh.inner_boundary
    )

    Bx = lfm_mesh.Bx - dip_mesh.Bx + igrf_mesh.Bx
    By = lfm_mesh.By - dip_mesh.By + igrf_mesh.By
    Bz = lfm_mesh.Bz - dip_mesh.Bz + igrf_mesh.Bz

    return MagneticFieldModel(
        lfm_mesh.x, lfm_mesh.y, lfm_mesh.z, Bx, By, Bz, lfm_mesh.inner_boundary
    )
