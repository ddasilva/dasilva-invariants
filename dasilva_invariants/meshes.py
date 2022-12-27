"""Tools for obtaining meshes for use in calculating invariants.

Meshes are grids + magnetic field vectors at those grid points. They
are instances of :py:class:`~MagneticFieldModel`.

In this module, all grids returned are in units of Re and all magnetic
fields are in units of Gauss.
"""
from typing import cast, Dict, List, Union, Optional, Tuple

from ai import cs
from astropy import constants, units
from collections.abc import Sequence
from datetime import datetime, timedelta

from matplotlib.dates import date2num
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import PyGeopack as gp
from pyhdf.SD import SD, SDC
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

from .constants import EARTH_DIPOLE_B0, LFM_INNER_BOUNDARY
from .utils import nanoTesla2Gauss

__all__ = [
    "MagneticFieldModel",
    "get_dipole_mesh_on_lfm_grid",
    "get_lfm_hdf4_data",
    "get_tsyganenko_on_lfm_grid_with_auto_params",
    "get_tsyganenko_params",
]


class MagneticFieldModel:
    """Represents a magnetic field model, with methods for sampling the
    magnetic field at an aribtrary point.

    The `interp_method` can be changed after the object is created. For
    calculating L* it is recommended to use "preprocess". In some limited
    situations, if only K needs to be calculated, "kdtree" may be faster.
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
        interp_method: str = "preprocess",
    ):
        """ "Create a magnetic field model.

        Parameters
        ----------
        x_grid : array of (m, n, p)
            X coordinates of data, in SM coordiantes and units of Re
        y_grid : array of (m, n, p)
            Y coordinates of data, in SM coordiantes and units of Re
        z_grid : array of (m, n, p)
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
        interp_method : {'preprocess', 'kdtree'}, optional
            Interpolation method. See class documentation for guidelines.

        Raises
        ------
        ValueError
            Invalid value for `interp_method` provided
        """
        self.x = x
        self.y = y
        self.z = z
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.inner_boundary = inner_boundary
        self.interp_method = interp_method

    @property
    def interp_method(self) -> str:
        """Controls the interpolation method used. 

        Returns
        -------
        Interpolation method selected
        """
        return self._interp_method

    @interp_method.setter
    def interp_method(self, value: str) -> None:
        """Controls the interpolation method used.

        Parameters
        ----------
        value : {'preprocess', 'kdtree'}
           New value for `interp_method`

        Raises
        ------
        ValueError
            Invalid value for `interp_method` provided
        """
        points = np.array([self.x.flatten(), self.y.flatten(), self.z.flatten()]).T
        values = np.array([self.Bx.flatten(), self.By.flatten(), self.Bz.flatten()]).T

        if value == "preprocess":
            mask = np.linalg.norm(points, axis=1) < 30
            self._interp = LinearNDInterpolator(points[mask], values[mask])
            #self._interp = LinearNDInterpolator(points, values)
        else:
            raise ValueError(f"Invalid value for interp_method {repr(value)}")

        self._interp_method = value

    def interpolate(self, x: Tuple[float, float, float]) -> Tuple[float]:
        """Find the magnetic field at a given point

        Parameters
        ----------
        point : tuple of float
            Coordinates (x, y, z) to interpolate at. SM Coordinate system,
            units of Re.
        """
        return self._interp(x.T).T


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
    X_grid, Y_grid, Z_grid = _get_fixed_lfm_grid_centers(lfm_hdf4_path)

    x_re = X_grid.flatten(order="F")  # flat arrays, easier later
    y_re = Y_grid.flatten(order="F")
    z_re = Z_grid.flatten(order="F")

    # Calculate dipole model
    # ------------------------------------------------------------------------
    # Dipole model, per Kivelson and Russel equations 6.3(a)-(c), page 165.
    r_re = np.sqrt(x_re**2 + y_re**2 + z_re**2)
    n_points = r_re.size

    Bx = nanoTesla2Gauss(3 * x_re * z_re * EARTH_DIPOLE_B0 / r_re**5)
    By = nanoTesla2Gauss(3 * y_re * z_re * EARTH_DIPOLE_B0 / r_re**5)
    Bz = nanoTesla2Gauss((3 * z_re**2 - r_re**2) * EARTH_DIPOLE_B0 / r_re**5)

    # Create magnetic field model
    # ------------------------------------------------------------------------
    return MagneticFieldModel(
        X_grid, Y_grid, Z_grid, Bx, By, Bz, inner_boundary=LFM_INNER_BOUNDARY
    )


def _get_fixed_lfm_grid_centers(
    lfm_hdf4_path: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    # X_grid_re = (X_grid_raw * units.cm).to(constants.R_earth).value  # type: ignore
    # Y_grid_re = (Y_grid_raw * units.cm).to(constants.R_earth).value  # type: ignore
    # Z_grid_re = (Z_grid_raw * units.cm).to(constants.R_earth).value  # type: ignore

    # return X_grid_re, Y_grid_re, Z_grid_re

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
    force_zero_tilt : bool
        Force a zero tilt when calculating the magnetic field

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

    x_re_sm = X_re_sm_grid.flatten(order="F")  # flat arrays, easier for later
    y_re_sm = Y_re_sm_grid.flatten(order="F")
    z_re_sm = Z_re_sm_grid.flatten(order="F")

    # Call Geopack to get external fields
    # ------------------------------------------------------------------------
    date = int(time.strftime("%Y%m%d"))
    ut = int(time.strftime("%H")) + time.minute / 60
    
    gp_tmp = gp.ModelField(
        x_re_sm,
        y_re_sm,
        z_re_sm,
        Date=date,
        ut=ut,
        Model=model_name,
        CoordIn="SM",
        CoordOut="SM",
        **params,
    )
    gp_tmp = cast(
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], gp_tmp
    )

    Bx, By, Bz = gp_tmp

    Bx = nanoTesla2Gauss(Bx)
    By = nanoTesla2Gauss(By)
    Bz = nanoTesla2Gauss(Bz)

    # Calculate dipole field for internal model,
    # see Kivelson and Russel equations 6.3(a)-(c), page 165.
    # ----------------------------------------------------------------------
    if external_field_only:
        r_re_sm = np.sqrt(x_re_sm**2 + y_re_sm**2 + z_re_sm**2)

        Bx_int = 3 * x_re_sm * z_re_sm * EARTH_DIPOLE_B0 / r_re_sm**5
        By_int = 3 * y_re_sm * z_re_sm * EARTH_DIPOLE_B0 / r_re_sm**5
        Bz_int = (3 * z_re_sm**2 - r_re_sm**2) * EARTH_DIPOLE_B0 / r_re_sm**5

        Bx_int = nanoTesla2Gauss(Bx_int)
        By_int = nanoTesla2Gauss(By_int)
        Bz_int = nanoTesla2Gauss(Bz_int)

        Bx -= Bx_int
        By -= By_int
        Bz -= Bz_int

    # Create magnetic field model
    # ------------------------------------------------------------------------
    return MagneticFieldModel(
        X_re_sm_grid,
        Y_re_sm_grid,
        Z_re_sm_grid,
        Bx,
        Bz,
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
                "SymH",
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
        datetime(int(row.Year), 1, 1) + timedelta(days=row.Day - 1, hours=row.Hr)
        for _, row in df.iterrows()
    ]

    # Interpolate Tsyganenko parameters (some may be unused)
    cols = ["Pdyn", "SymH", "By", "Bz", "W1", "W2", "W3", "W4", "W5", "W6"]
    params_dict = {}

    for col in cols:
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
