"""Numerical calculation of adiabatic invariants.

The public interface functions are as follows, each of which return dataclasses
defined specifically for the function. These functions raise a documented set of
exceptions, which should be caught to detect problems which may occur during
processing.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys

from ai import cs
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from scipy.integrate import RK45

from . import utils
from .meshes import MagneticFieldModel


__all__ = [
    "CalculateKResult",
    "CalculateLStarResult",
    "calculate_K",
    "calculate_LStar",
    "FieldLineTraceInsufficient",
    "DriftShellSearchDoesntConverge",
    "DriftShellLinearSearchDoesntConverge",
    "DriftShellBisectionDoesntConverge",
]


@dataclass
class FieldLineTrace:
    """Class to hold the results of a field line trace.

    Parameters
    ----------
    points : array
       Positions along field line trace, in SM coordinate system and units of Re
    B : array
       Magnetic field vector along field line trace, in SM coordinates and units
       of Gauss
    """

    points: NDArray[np.float64]
    B: NDArray[np.float64]


@dataclass
class CalculateKResult:
    """Class to hold the return value of calculate_K(). This class also provides
    information to inspect the bounce path determined while calculating K.

    All arrays are sorted by magnetic latitude.

    Parameters
    ----------
    K : float
        The second adiabatic invariant, in units of sqrt(G) * Re
    Bm : float
        Magnetic field strength at mirroring point, in units of Gauss
    Bmin : float
        Minimum magnetic field strength along the bounce path, in units of Gauss
    mirror_latitude : float, optional
        If mirroring point was specified in terms of a magnetic latitude, this
        is that magnetic latitude. Units of degrees in SM coordinate system.
    starting_point : tuple of floats
        Starting point of the field line trace used to determine the bounce
        path. Units of Re in SM coordinate system.
    trace_points : NDArray[np.float64]
        Array of cartesian coordinates along the bounce path
    trace_latitude : NDArray[np.float64]
        Array of magnetic latitudes along the bounce path, in units of radians
    trace_field_strength : NDArray[np.float64]
        Array of magnetic field strengths along the bounce path, in units of
        Gauss.
    integral_axis_latitude : NDArray[np.float64]
        Corresponds to the intergration used to find K. These are the magnetic
        latitudes across the integration domain (bounce path)
    integral_integrand : NDArray[np.float64]
        Corresponds to the intergration used to find K. These are the integrand
        values across the integration domain (bounce path)
    """

    K: float
    Bm: float
    Bmin: float
    mirror_latitude: Optional[float]
    starting_point: Tuple[float, float, float]

    trace_points: NDArray[np.float64]
    trace_latitude: NDArray[np.float64]
    trace_field_strength: NDArray[np.float64]

    integral_axis: NDArray[np.float64]
    integral_axis_latitude: NDArray[np.float64]
    integral_integrand: NDArray[np.float64]

    _trace: FieldLineTrace


@dataclass
class CalculateLStarResult:
    """Class to hold the return value of calculate_LStar(). This includes the
    LStar adiabatic invariant as well as all details required to reconstruct
    the full drift shell.

    Parameters
    ----------
    LStar : float
        Third adiabatic invariant (L*), unitless
    drift_local_times : NDArray[np.float64]
        Array of magnetic local times around the drive shell.
    drift_rvalues : NDArray[np.float64]
        Array of drift shell radius at each local time, to be paired with
        `drift_local_times`
    drift_K : NDArray[np.float64]
        Array of second adiabatic invariant (K) at each local time, to be
        paired with `drift_local_times`
    drift_K_results : List[:py:class:`~CalculateKResult`]
        Array of :py:class:`~CalculateKResult` instances at each local time,
        to be paired with `drift_local_times. Through this object one can
        obtain a bounce motion path at each local time.
    drift_is_closed : bool
        Boolean whether the drift shell was detected to be closed. For more
        information, see da Silva et al., 2023
    integral_axis : NDArray[np.float64]
        Corresponding to the integral used to calculate LStar, this is the
        integration axis in local time. Units of radians
    integral_theta : NDArray[np.float64]
        Corresponding to the integral used to calculate LStar, this is the
        variable integrated over. Units of radians
    integral_integrand : NDArray[np.float64]
        Corresponding to the integral used to calculate LStar, this is the
        integrand
    """

    # Third adiabatic invariant (L*)
    LStar: float
    # magnetic local times of drift shell
    drift_local_times: NDArray[np.float64]
    # radius drift shell at local time
    drift_rvalues: NDArray[np.float64]
    # drift shell K values, shorthand
    drift_K: NDArray[np.float64]
    # Comes from calculate_K()
    drift_K_results: List[CalculateKResult]
    # Whether drift shell is closed
    drift_is_closed: bool
    # integaxis local time (radians)
    integral_axis: NDArray[np.float64]
    # intragral theta variable (variable integrated over)
    integral_theta: NDArray[np.float64]
    # integral integrand
    integral_integrand: NDArray[np.float64]


class FieldLineTraceInsufficient(RuntimeError):
    """Raised when a field line trace is performed but the result is empty or
    is too small."""


class DriftShellSearchDoesntConverge(RuntimeError):
    """Raised when search to determine drift shell doesn't converge"""


class DriftShellLinearSearchDoesntConverge(DriftShellSearchDoesntConverge):
    """Raised when linear search to determine the drift shell doesn't
    converge."""


class DriftShellBisectionDoesntConverge(DriftShellSearchDoesntConverge):
    """Raised when bisection serach to determine the drift shell doesn't
    converge."""


def calculate_K(
    mesh: MagneticFieldModel,
    starting_point: Tuple[float, float, float],
    mirror_latitude: Optional[float] = None,
    Bm: Optional[float] = None,
    pitch_angle: Optional[float] = None,
    step_size: Optional[float] = None,
    reuse_trace: Optional[FieldLineTrace] = None,
) -> CalculateKResult:
    """Calculate the third adiabatic invariant, K.

    Either `mirror_latitude`, `Bm`, or `pitch_angle` must be specified.

    Parameters
    ----------
    mesh : MagneticFieldModel
        Grid and magnetic field, loaded using meshes module
    starting_point : tuple of floats
        Starting point of the field line trace, as (x, y, z) tuple of
        floats, in units of Re.
    mirror_latitude : float, optional
        Magnetic latitude in degrees to use for the mirroring point, to
        specify bounce path
    Bm : float, optional
        Magnetic field strength at mirroring point, to specify bounce path
    pitch_angle : float, optional
        Local pitch angle at starting point, to specify the bounce path. In
        units of degrees
    step_size ; float, optional
        Step size to use with the field line trace

    Returns
    -------
    result : :py:class:`~CalculateKResult`
         Calcualte K variable and related bounce path information

    Raises
    ------
    FieldLineTraceInsufficient
        field line trace empty or too small
    """
    # Validate function arguments
    # ------------------------------------------------------------------------
    optional_params = [mirror_latitude, Bm, pitch_angle]
    num_specified = len({val for val in optional_params if val is not None})

    if num_specified == 0:
        raise ValueError(
            'One of the keyword arguments "mirror_latitude", "Bm", or '
            '"pitch_angle" must be specified.'
        )
    elif num_specified > 1:
        raise ValueError(
            'Only one of the keyword arguments "mirror_latitude", "Bm", or '
            '"pitch_angle" may be specified.'
        )

    # Calculate field line trace
    # ------------------------------------------------------------------------
    if step_size is None:
        step_size = 1e-1

    if reuse_trace is None:
        trace = trace_field_line(mesh, starting_point, step_size)

    if len(trace.points) == 0:
        r = np.linalg.norm(starting_point)
        raise FieldLineTraceInsufficient(
            f"Trace returned empty from {starting_point}, r={r:.1f}"
        )
    if len(trace.points) < 50:
        raise FieldLineTraceInsufficient(
            f"Trace too small ({len(trace.points)} points)"
        )

    trace_field_strength = np.linalg.norm(trace.B, axis=1)

    # Get the trace latitudes and Bm if not specified
    # ------------------------------------------------------------------------
    _, trace_latitude, _ = cs.cart2sp(
        x=trace.points[:, 0], y=trace.points[:, 1], z=trace.points[:, 2]
    )
    trace_sorter = np.argsort(trace_latitude)
    Bmin = trace_field_strength.min()

    if mirror_latitude is not None:
        tmp = np.array([mirror_latitude])
        (Bm,) = np.interp(
            x=np.deg2rad(tmp),
            xp=trace_latitude[trace_sorter],
            fp=trace_field_strength[trace_sorter],
        )
    elif pitch_angle is not None:
        tmp = np.array([pitch_angle])
        (Bm,) = Bmin / np.sin(np.deg2rad(tmp)) ** 2
    elif Bm is None:
        raise RuntimeError("This code should not be reachable")

    assert Bm is not None, "Bm should be non-None here"

    # Sort field line trace points
    # ------------------------------------------------------------------------
    trace_latitude_sorted = trace_latitude[trace_sorter]
    trace_points_sorted = trace.points[trace_sorter]
    trace_field_strength_sorted = trace_field_strength[trace_sorter]

    _, unique_inds = np.unique(trace_latitude_sorted, return_index=True)
    trace_latitude_sorted = trace_latitude_sorted[unique_inds]
    trace_points_sorted = trace_points_sorted[unique_inds]
    trace_field_strength_sorted = trace_field_strength_sorted[unique_inds]

    # Find mask for deepest |B| Well
    # ------------------------------------------------------------------------
    Bm_mask = trace_field_strength_sorted < Bm

    # Calculate Function Values
    # ------------------------------------------------------------------------
    ds_vec = np.diff(trace_points_sorted[Bm_mask], axis=0)
    ds_scalar = np.linalg.norm(ds_vec, axis=1)

    integral_axis_latitude = np.rad2deg(trace_latitude_sorted[Bm_mask])
    integral_axis = np.array([0] + np.cumsum(ds_scalar).tolist())
    integral_integrand = np.sqrt(Bm - trace_field_strength_sorted[Bm_mask])

    K = np.trapz(integral_integrand, integral_axis)

    # Return results
    # ------------------------------------------------------------------------
    return CalculateKResult(
        K=K,
        Bm=Bm,
        Bmin=Bmin,
        mirror_latitude=mirror_latitude,
        starting_point=starting_point,
        _trace=trace,
        trace_points=trace_points_sorted,
        trace_latitude=trace_latitude_sorted,
        trace_field_strength=trace_field_strength_sorted,
        integral_axis=integral_axis,
        integral_axis_latitude=integral_axis_latitude,
        integral_integrand=integral_integrand,
    )


def calculate_LStar(
    mesh: MagneticFieldModel,
    starting_point: Tuple[float, float, float],
    mode: str = "linear",
    num_local_times: int = 4,
    starting_mirror_latitude: Optional[float] = None,
    Bm: Optional[float] = None,
    starting_pitch_angle: Optional[float] = None,
    major_step: float = 0.05,
    minor_step: float = 0.01,
    interval_size_threshold: float = 0.05,
    rel_error_threshold: float = 0.01,
    max_iters: int = 300,
    trace_step_size: Optional[float] = None,
    interp_local_times: bool = True,
    interp_npoints: int = 1024,
    verbose: bool = False,
) -> CalculateLStarResult:
    """Calculate the third adiabatic invariant, L*

    Can be run in two modes, 'linear' and 'bisection'. Linear mode is
    slower, but gives better results for distributed magnetic fields.

    Parameters
    ----------
    mesh : :py:class:`~MagneticFieldModel`
        grid and magnetic field, loaded using meshes module
    starting_point : tuple of floats
        Starting point of the field line integration, as
        (x, y, z) tuple of floats, in units of Re
    mode : {'linear', 'bisection'}, optional
        Linear is more suiltable for non-quiet fields, but bisection may be
        faster. Defaults to 'linear'
    num_local_times : int, optional
        Number of local times spaced evenly around the drift shell to solve
        for with bisection
    starting_mirror_latitude : float, optional
        Latitude in degrees to use for the mirroring point for the local time
        associated with the starting point to calculate drift shell
    Bm : float, optional
        Magnetic field strength at mirror point used to calculate drift
        shell
    starting_pitch_angle : float, optional
        Pitch angle at Bmin at starting point to calcualte the drift shell.
        If set to 90.0, then a special path will be taken through the code
        where the drift shell is used by searching for isolines of Bmin instead
        of K
    major_step : float, optional
        Only used by mode='linear'. Size of large step (float, units of Re)
        to find rough location of drift shell radius
    minor_step : float, optional
        Only used by mode='linear'. Size of small step (float, units of Re)
        to refine drift shell radius
    interval_size_threshold : float, optional
        Only used by mode='bisection'. Bisection threshold before linearly
        interpolating
    max_iters : int, optional
        Used by all modes. Maximum iterations before raising exception.
    trace_step_size : float, optional
        Used by all modes. step size used to trace field lines
    interp_local_times : bool, optional
        Interpolate intersection latitudes for local times with cubic splines
        to allow for less local time calculation
    interp_npoints : int, optional
        Number of points to use in interplation, only active if
        interp_local_times=True
    verbose : bool, optional
        Set to true to enable logging messages to console (stdout)

    Returns
    -------
      result: :py:class:`~CalculateLStarResult`
    """
    # Determine list of local times we will search
    # ------------------------------------------------------------------------
    _, _, starting_phi = utils.cart2sp_point(*starting_point)

    starting_rvalue, _, _ = utils.cart2sp_point(
        x=starting_point[0], y=starting_point[1], z=0
    )

    drift_local_times = starting_phi + (
        2 * np.pi * np.arange(num_local_times) / num_local_times
    )

    # Calculate K at the current local time.
    # ------------------------------------------------------------------------
    if verbose:
        print(f"Calculating drift radius 1/{drift_local_times.size}")

    if Bm is not None:
        kwargs = {"Bm": Bm}
    elif starting_mirror_latitude is not None:
        kwargs = {"mirror_latitude": starting_mirror_latitude}
    elif starting_pitch_angle is not None:
        kwargs = {"pitch_angle": starting_pitch_angle}
    else:
        raise RuntimeError(
            "Must specify one of Bm=, starting_mirror_latitude=, or " "pitch_angle="
        )

    starting_result = calculate_K(
        mesh, starting_point, step_size=trace_step_size, **kwargs  # type: ignore
    )

    # Estimate radius of equivalent K/Bm at other local times using method
    # based on whether the particle is equitorial mirroring or not (if it is,
    # a trace is not required and can be skipped).
    # ------------------------------------------------------------------------
    drift_shell_results = [(starting_rvalue, starting_result)]

    for i, local_time in enumerate(drift_local_times):
        if i == 0:
            continue
        if verbose:
            print(f"Calculating drift radius {i+1}/{drift_local_times.size}")

        last_rvalue, _ = drift_shell_results[-1]

        if mode == "linear":
            if starting_pitch_angle == 90.0:
                output = _linear_search_rvalue_by_Bmin(
                    mesh,
                    starting_result.Bm,
                    last_rvalue,
                    local_time,
                    max_iters,
                    trace_step_size,
                    major_step,
                    minor_step,
                )
            else:
                output = _linear_search_rvalue_by_K(
                    mesh,
                    starting_result.K,
                    starting_result.Bm,
                    last_rvalue,
                    local_time,
                    max_iters,
                    trace_step_size,
                    major_step,
                    minor_step,
                )
        elif mode == "bisection":
            if starting_pitch_angle == 90.0:
                raise NotImplementedError(
                    "_bisect_search_rvalue_by_Bmin() not implemented"
                )
                # output = _bisect_search_rvalue_by_Bmin(
                #    mesh, starting_result.Bm, starting_rvalue, local_time,
                #    max_iters, trace_step_size, major_step, minor_step
                # )
            else:
                output = _bisect_rvalue_by_K(
                    mesh,
                    starting_result.K,
                    starting_result.Bm,
                    last_rvalue,
                    local_time,
                    max_iters,
                    interval_size_threshold,
                    rel_error_threshold,
                    trace_step_size,
                )
        else:
            raise RuntimeError("Code should never reach here")

        drift_shell_results.append(output)

    # Extract drift shell results into arrays which correspond in index to
    # drift_local_times
    drift_rvalues: List[float] = []
    drift_K_results: List[CalculateKResult] = []

    for i, (tmp_rvalue, tmp_result) in enumerate(drift_shell_results):
        drift_rvalues.append(tmp_rvalue)
        drift_K_results.append(tmp_result)

    # Calculate L*
    # This method assumes a dipole below the innery boundary, and integrates
    # around the local times using stokes law with B = curl A.
    # -----------------------------------------------------------------------
    inner_rvalue = mesh.inner_boundary
    surface_rvalue = 1

    trace_north_latitudes = np.array(
        [result.trace_latitude.max() for result in drift_K_results], dtype=float
    )

    if interp_local_times:
        # Interpolate with cubic spline with periodic boundary condition
        # that forces the 1st and 2nd derivatives to be equal at the first
        # and last points
        spline_x = np.zeros(len(drift_K_results) + 1)
        spline_y = np.zeros(len(drift_K_results) + 1)

        spline_x[:-1] = drift_local_times
        spline_x[-1] = drift_local_times[0] + 2 * np.pi

        spline_y[:-1] = trace_north_latitudes
        spline_y[-1] = trace_north_latitudes[0]

        spline = interpolate.CubicSpline(spline_x, spline_y, bc_type="periodic")

        integral_axis = np.linspace(spline_x.min(), spline_x.max(), interp_npoints)
        # colatitude
        integral_theta = np.pi / 2 - spline(integral_axis).astype(float)
    else:
        integral_axis = np.zeros(drift_local_times.size + 1)
        integral_axis[:-1] = drift_local_times
        integral_axis[-1] = integral_axis[0] + 2 * np.pi

        integral_theta = np.zeros(len(drift_K_results) + 1)
        integral_theta[:-1] = np.pi / 2 - trace_north_latitudes  # colatitude
        integral_theta[-1] = integral_theta[0]

    integral_integrand = np.sin(integral_theta) ** 2.0
    integral = np.trapz(integral_integrand, integral_axis)
    LStar = 2 * np.pi * (inner_rvalue / surface_rvalue) / integral

    # Return results
    # ------------------------------------------------------------------------
    drift_K = np.array([result.K for result in drift_K_results], dtype=float)
    drift_rvalues_arr = np.array(drift_rvalues)
    drift_is_closed = _test_drift_is_closed(drift_rvalues_arr)

    if not drift_is_closed:
        LStar = np.nan

    return CalculateLStarResult(
        LStar=LStar,
        drift_local_times=drift_local_times,
        drift_rvalues=drift_rvalues_arr,
        drift_K=drift_K,
        drift_K_results=drift_K_results,
        drift_is_closed=drift_is_closed,
        integral_axis=integral_axis,
        integral_theta=integral_theta,
        integral_integrand=integral_integrand,
    )


def trace_field_line(
    mesh: MagneticFieldModel,
    starting_point: Tuple[float, float, float],
    step_size: float,
) -> FieldLineTrace:
    """ "Perform a field line trace. Implements RK45 in both directions, stopping
    when `mesh.innery_boundary` is crossed.

    Parameters
    ----------
    mesh : MagneticFieldModel
        Grid and magnetic field, loaded using meshes module
    starting_point : tuple of floats
        Starting point of the field line trace, as (x, y, z) tuple of
        floats, in units of Re.
    step_size : float, optional
        Step size to use with the field line trace

    Returns
    -------
    trace : :py:class:`FieldLineTrace`
        Coordinates and magnetic field vector along the field line trace
    """

    def _fun(t, y):
        B = mesh.interpolate(y)
        direction = B / np.linalg.norm(B)
        return direction

    # Integrate forwards ----------------------------------------------------
    rk45 = RK45(
        _fun,
        t0=0,
        y0=starting_point,
        first_step=step_size,
        max_step=step_size,
        t_bound=sys.float_info.max,
    )

    forward_points = [starting_point]

    while True:
        rk45.step()
        next_point = rk45.y
        # print(next_point)
        if np.linalg.norm(next_point) < mesh.inner_boundary:
            break
        forward_points.append(next_point)

    # Integrate backwards ----------------------------------------------------
    rk45 = RK45(
        _fun,
        t0=0,
        y0=starting_point,
        first_step=step_size,
        max_step=step_size,
        t_bound=-sys.float_info.max,
    )

    backward_points = [starting_point]

    while True:
        rk45.step()
        next_point = rk45.y
        # print(next_point)
        if np.linalg.norm(next_point) < mesh.inner_boundary:
            break
        backward_points.append(next_point)

    # return FieldLineTrace object  ------------------------------------------
    points = np.array(backward_points + forward_points)
    B = np.array([mesh.interpolate(point) for point in points])

    trace = FieldLineTrace(points=points, B=B)

    return trace


def _bisect_rvalue_by_K(
    mesh: MagneticFieldModel,
    target_K: float,
    Bm: float,
    starting_rvalue: float,
    local_time: float,
    max_iters: int,
    interval_size_threshold: float,
    rel_error_threshold: float,
    step_size: Optional[float],
) -> Tuple[float, CalculateKResult]:
    """Internal helper function to calculate_LStar(). Applies bisection method
    to find an radius with an equal K, stopping when either relative error
    is sufficiently small or interval being searched is sufficiently small to
    interpolate.

    Parameters
    ----------
    mesh : MagneticFieldModel
        Grid and magnetic field, loaded using meshes module
    target_K : float
        K value to search for (float)
    Bm : float
        Magnetic mirroring point, parameter used to estimte K
    starting_rvalue : float
        Initial radius of search
    local_time : float
        Current local time in radians
    max_iters : int
        Maximum number of iterations before erroring out (int)
    interval_size_threshold : float
        Bisection threshold before linearly interpolating
    rel_error_threshold : float
        Relative error threshold to consider two K's equal, between [0, 1]

    Returns
    -------
    rvalue : float
        Radius at given local time which produces the same K on he given
        mesh
    calculate_K_result : :py:class:`~CalculateKResult`
        Value of K and field line trac information corresponding to final field
        line

    Raises
    -------
    DriftShellBisectionDoesntConverge
        Maximum number of iterations reached
    """
    # Perform bisection method searching for K(Bm, r)
    # ------------------------------------------------------------------------
    upper_rvalue = starting_rvalue * 2
    lower_rvalue = mesh.inner_boundary
    current_rvalue = starting_rvalue
    history = []

    for _ in range(max_iters):
        # Check for interval size stopping condition -------------------------
        interval_size = upper_rvalue - lower_rvalue

        if interval_size < interval_size_threshold:
            # Calculate K and upper and lower bounds
            lower_starting_point = utils.sp2cart_point(
                r=lower_rvalue, phi=-local_time, theta=0
            )
            upper_starting_point = utils.sp2cart_point(
                r=upper_rvalue, phi=-local_time, theta=0
            )

            lower_result = calculate_K(
                mesh, lower_starting_point, Bm=Bm, step_size=step_size
            )
            upper_result = calculate_K(
                mesh, upper_starting_point, Bm=Bm, step_size=step_size
            )

            # Interpolate between upper and lower bounds to find final rvalue
            (final_rvalue,) = np.interp(
                [target_K],
                [upper_result.K, lower_result.K],
                [upper_rvalue, lower_rvalue],
            )
            final_starting_point = utils.sp2cart_point(
                r=final_rvalue, phi=-local_time, theta=0
            )

            final_result = calculate_K(
                mesh, final_starting_point, Bm=Bm, step_size=step_size
            )

            return final_rvalue, final_result

        # Check for relative error stopping condition ------------------------
        current_starting_point = utils.sp2cart_point(
            r=current_rvalue, phi=-local_time, theta=0
        )

        current_result = calculate_K(
            mesh, current_starting_point, Bm=Bm, step_size=step_size
        )

        rel_error = abs(target_K - current_result.K) / max(
            np.abs(target_K), np.abs(current_result.K)
        )

        if rel_error < rel_error_threshold:
            # match found
            return current_rvalue, current_result

        # Continue iterating by halving the interval -------------------------
        if current_result.K < target_K:
            # too low
            history.append((interval_size, "too_low", current_rvalue))

            current_rvalue, lower_rvalue = (
                (upper_rvalue + current_rvalue) / 2,
                current_rvalue,
            )
        else:
            # too high
            history.append((interval_size, "too_high", current_rvalue))

            current_rvalue, upper_rvalue = (
                (lower_rvalue + current_rvalue) / 2,
                current_rvalue,
            )

    # If the code reached this point, the maximum number of iterations
    # was exhausted.
    # ------------------------------------------------------------------------
    raise DriftShellBisectionDoesntConverge(
        f"Maximum number of iterations {max_iters} reached for local time "
        f"{local_time:.1f} during bisection " + repr(history)
    )


def _linear_search_rvalue_by_Bmin(
    mesh: MagneticFieldModel,
    target_Bmin: float,
    initial_rvalue: float,
    local_time: float,
    max_iters: int,
    step_size: Optional[float],
    major_step: float,
    minor_step: float,
) -> Tuple[float, CalculateKResult]:
    """Internal helper function to calculate_LStar(). Steps in large and then
    small increments to search for a drift shell radius that has the target
    Bmin.

    Args
      mesh: reference to grid and magnetic field
      target_Bmin: floating point Bmin value to search for
      initial_rvalue: initial point rvalue (float)
      local_time: initial local time (radians, float)
      max_iters: Maximum number of iterations before erroring out (int)
      step_size: step size used to trace field lines
      major_step: Size of large step (Re) to find rough location of drift shell
        radius.
      minor_step: Size of small step (Re) to refine drift shell radius.

    Returns
      rvalue: radius at given local time which produces the same K on
        the given mesh (float)
      calculate_K_result: instance of CalculateKResult corresponding to
        radius and calculate_K().
    Raises
      DriftShellBisectionDoesntConverge: maximu number of iterations reached
    """
    # Decide which direction to iterate --------------------------------------
    initial_point = utils.sp2cart_point(r=initial_rvalue, phi=-local_time, theta=0)
    initial_result = calculate_K(
        mesh, initial_point, pitch_angle=90, step_size=step_size
    )
    initial_Bmin = initial_result.Bmin

    if initial_Bmin < target_Bmin:
        direction = -1  # too small, walk inward
    else:
        direction = 1  # too big, walk outward

    # Major step iteration, scan with low resolution
    #
    # Walks outward/inward in increments of `major_step`. Stops when finds
    # Bmin that overshoots.
    # ------------------------------------------------------------------------
    history_rvalue = [initial_rvalue]
    history_Bmin = [initial_Bmin]

    clean_finish = False
    minor_start_rvalue = -1.0
    minor_start_Bmin = -1.0

    for i in range(1, max_iters + 1):
        current_rvalue = initial_rvalue + i * major_step * direction
        current_point = utils.sp2cart_point(r=current_rvalue, phi=-local_time, theta=0)
        current_result = calculate_K(
            mesh, current_point, pitch_angle=90, step_size=step_size
        )
        current_Bmin = current_result.Bmin

        history_rvalue.append(current_rvalue)
        history_Bmin.append(current_Bmin)

        tmp = sorted(history_Bmin[-2:])
        in_interval = tmp[0] < target_Bmin < tmp[1]

        if in_interval:
            # Proceed to minor step iteration
            clean_finish = True
            minor_start_rvalue = history_rvalue[-2]
            minor_start_Bmin = history_Bmin[-2]
            break

    if not clean_finish:
        raise DriftShellBisectionDoesntConverge(
            f"Maximum number of iterations {max_iters} reached during major "
            f"iteration for local time {local_time:.1f} during iteration "
            + repr(list(zip(history_rvalue, history_Bmin)))
            + ","
            + repr((initial_Bmin, target_Bmin))
        )

    # Minor step iteration, scan with finer resolution
    # ------------------------------------------------------------------------
    history_rvalue = [minor_start_rvalue]
    history_Bmin = [minor_start_Bmin]

    for i in range(1, max_iters + 1):
        current_rvalue = minor_start_rvalue + i * minor_step * direction
        current_point = utils.sp2cart_point(r=current_rvalue, phi=-local_time, theta=0)
        current_result = calculate_K(
            mesh, current_point, pitch_angle=90, step_size=step_size
        )
        current_Bmin = current_result.Bmin

        history_rvalue.append(current_rvalue)
        history_Bmin.append(current_Bmin)

        tmp = sorted(history_Bmin[-2:])
        in_interval = tmp[0] < target_Bmin < tmp[1]

        if in_interval:
            # Interpolate between upper and lower bounds to find final rvalue
            (final_rvalue,) = np.interp(
                [target_Bmin ** (1 / 3)],
                np.array(history_Bmin[-2:]) ** (1 / 3),
                history_rvalue[-2:],
            )
            final_initial_point = utils.sp2cart_point(
                r=final_rvalue, phi=-local_time, theta=0
            )
            final_result = calculate_K(
                mesh, final_initial_point, pitch_angle=90, step_size=step_size
            )

            return final_rvalue, final_result

    raise DriftShellLinearSearchDoesntConverge(
        f"Maximum number of iterations {max_iters} reached during minor "
        f"iteration for local time {local_time:.1f} during iteration "
        + repr(list(zip(history_rvalue, history_Bmin)))
        + ","
        + repr((initial_Bmin, target_Bmin))
    )


def _linear_search_rvalue_by_K(
    mesh: MagneticFieldModel,
    target_K: float,
    Bm: float,
    initial_rvalue: float,
    local_time: float,
    max_iters: int,
    step_size: Optional[float],
    major_step: float,
    minor_step: float,
) -> Tuple[float, CalculateKResult]:
    """Internal helper function to calculate_LStar(). Steps in large and then
    small increments to search for a drift shell radius that has the target
    K with the given Bmin.

    Args
      mesh: reference to grid and magnetic field
      target_K: floating point K value to search for
      Bm: strength of magnetic field at mirror point, used to calcualte K
      initial_rvalue: initial point rvalue (float)
      local_time: initial local time (radians, float)
      max_iters: Maximum number of iterations before erroring out (int)
      step_size: step size used to trace field lines
      major_step: Size of large step (Re) to find rough location of drift shell
        radius.
      minor_step: Size of small step (Re) to refine drift shell radius.

    Returns
      rvalue: radius at given local time which produces the same K on
        the given mesh (float)
      calculate_K_result: instance of CalculateKResult corresponding to
        radius and calculate_K().
    Raises
      DriftShellBisectionDoesntConverge: maximum number of iterations reached
    """
    # Decide which direction to iterate
    #
    # The main point here is that in a small neighborhood, K shrinks with
    # increasing rvalue.
    # ---------------------------------------------------------------------
    initial_point = utils.sp2cart_point(r=initial_rvalue, phi=-local_time, theta=0)
    initial_result = calculate_K(mesh, initial_point, Bm=Bm, step_size=step_size)
    initial_K = initial_result.K

    if Bm < initial_result.Bmin or initial_K == 0:
        direction = 1
    elif initial_K < target_K:
        direction = 1
    else:
        direction = -1

    # Major step iteration, scan with low resolution
    #
    # Walks outward/inward in increments of `major_step`. Stops when finds
    # Bmin that overshoots.
    # ------------------------------------------------------------------------
    history_rvalue = [initial_rvalue]
    history_K = [initial_K]

    clean_finish = False
    minor_start_rvalue = -1.0
    minor_start_K = -1.0

    for i in range(1, max_iters + 1):
        current_rvalue = initial_rvalue + i * major_step * direction
        current_point = utils.sp2cart_point(r=current_rvalue, phi=-local_time, theta=0)
        current_result = calculate_K(mesh, current_point, Bm=Bm, step_size=step_size)
        current_K = current_result.K

        history_rvalue.append(current_rvalue)
        history_K.append(current_K)

        tmp = sorted(history_K[-2:])
        in_interval = tmp[0] < target_K < tmp[1]

        if in_interval:
            # Proceed to minor step iteration
            clean_finish = True
            minor_start_rvalue = history_rvalue[-2]
            minor_start_K = history_K[-2]
            break

    if not clean_finish:
        raise DriftShellLinearSearchDoesntConverge(
            f"Maximum number of iterations {max_iters} reached during major "
            f"iteration for local time {local_time:.1f} during iteration "
            + repr(list(zip(history_rvalue, history_K)))
            + ","
            + repr((initial_K, target_K))
        )

    # Minor step iteration, scan with finer resolution
    # ------------------------------------------------------------------------
    history_rvalue = [minor_start_rvalue]
    history_K = [minor_start_K]

    for i in range(1, max_iters + 1):
        current_rvalue = minor_start_rvalue + i * minor_step * direction
        current_point = utils.sp2cart_point(r=current_rvalue, phi=-local_time, theta=0)
        current_result = calculate_K(mesh, current_point, Bm=Bm, step_size=step_size)
        current_K = current_result.K

        history_rvalue.append(current_rvalue)
        history_K.append(current_K)

        tmp = sorted(history_K[-2:])
        in_interval = tmp[0] < target_K < tmp[1]

        if in_interval:
            # Interpolate between upper and lower bounds to find final rvalue
            (final_rvalue,) = np.interp([target_K], history_K[-2:], history_rvalue[-2:])
            final_point = utils.sp2cart_point(r=final_rvalue, phi=-local_time, theta=0)
            final_result = calculate_K(mesh, final_point, Bm=Bm, step_size=step_size)

            return final_rvalue, final_result

    raise DriftShellLinearSearchDoesntConverge(
        f"Maximum number of iterations {max_iters} reached during minor "
        f"iteration for local time {local_time:.1f} during iteration "
        + repr(list(zip(history_rvalue, history_K)))
        + ","
        + repr((initial_K, target_K))
    )


def _test_drift_is_closed(drift_rvalues: NDArray[np.float64]) -> bool:
    """Test whether a drift shell is closed.

    Does so by checking wehether the different in radius between the final
    step and second to final step is no more than X% bigger than any other
    two consecutive steps in the drift shell.

    Args
      drift_rvalues: List of drift radii at local times. Must be in order
        with final step last.
    Returns
      true/false whether drift shell is closed
    """
    delta_rvalue_threshold = 1.5 * np.max(np.abs(np.diff(drift_rvalues)))
    delta_rvalue_final = np.abs(drift_rvalues[-1] - drift_rvalues[0])
    is_closed = delta_rvalue_final < delta_rvalue_threshold

    return is_closed
