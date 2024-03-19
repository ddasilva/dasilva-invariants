"""Numerical calculation of adiabatic invariants."""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ai import cs
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from scipy.integrate import solve_ivp

from . import utils
from .models import MagneticFieldModel, FieldLineTrace


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
    drift_K_results : dictionary of float to :py:class:`~CalculateKResult`
        Dict mapping local time to  :py:class:`~CalculateKResult`,
        to be paired with `drift_local_times`. Through this object one can
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
    # result of scipy.integrate.ivp_result
    ivp_result: Any


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
    model: MagneticFieldModel,
    starting_point, mirror_latitude=None,
    Bm=None, pitch_angle=None,
    step_size=None, reuse_trace=None,
) -> CalculateKResult:
    """Calculate the third adiabatic invariant, K.

    Either `mirror_latitude`, `Bm`, or `pitch_angle` must be specified.

    Parameters
    ----------
    model : :py:class:`~MagneticFieldModel`
        Grid and magnetic field, loaded using models module
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
        step_size = 1e-3

    if reuse_trace is None:
        trace = model.trace_field_line(starting_point, step_size)
    else:
        trace = reuse_trace

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
    model: MagneticFieldModel,
    starting_point,
    mode="normal",
    starting_mirror_latitude=None,
    Bm=None,
    starting_pitch_angle=None,
    integrand_atol=0.01,
    integrand_rtol=0.01,
    num_local_times=16,
    first_mlt_step=2 * np.pi / 16,
    max_mlt_step=2 * np.pi / 4,
    major_step=0.05,
    minor_step=0.01,
    interval_size_threshold=0.05,
    rel_error_threshold=0.01,
    max_iters=300,
    trace_step_size=None,
    interp_local_times=True,
    interp_npoints=1024,
    verbose=False,
) -> CalculateLStarResult:
    """Calculate the third adiabatic invariant, L*

    Can be run in two modes, 'normal and 'equitorial'. The normal mode searches
    for a drift shell by selecting a field line that matches K at each local
    time; the equitorial mode matches Bmin at each local time.

    Parameters
    ----------
    model : :py:class:`~MagneticFieldModel`
        grid and magnetic field, loaded using models module
    starting_point : tuple of floats
        Starting point of the field line integration, as
        (x, y, z) tuple of floats, in units of Re
    mode : {'normal', 'equitorial'}, optional
        Mode to run drift shell search in. Equitorial mode does special
        search using Bmin, which is faster. Overrides starting_mirror_latitude,
        Bm, and starting_pitch_angle.
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
        Size of large step (float, units of Re) to find rough location of drift
        shell radius
    num_local_times : 'adaptive' or int
        Number of local times to use. If 'adaptive' is passed, uses a Runge-Kutta
        method.
    minor_step : float, optional
        Size of small step (float, units of Re) to refine drift shell radius
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
        Number of points to usein interplation, only active if
        interp_local_times=True

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

    # Calculate K at the current local time.
    # ------------------------------------------------------------------------
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
        model, starting_point, step_size=trace_step_size, **kwargs  # type: ignore
    )

    # Estimate radius of equivalent K/Bm at other local times at each local
    # time determined by the RK45 method.
    # ------------------------------------------------------------------------
    func_args = (
        starting_phi, starting_rvalue, starting_result,
        starting_pitch_angle,
        num_local_times, major_step, max_iters, minor_step,
        model, mode, trace_step_size, first_mlt_step, max_mlt_step,
        integrand_atol, integrand_rtol,
        interval_size_threshold,
        rel_error_threshold,    
    )

    if isinstance(num_local_times, int):
        drift_shell_results, drift_local_times, ivp_result = (
            _do_basic_drift_shell_calc(*func_args)
        )
    else:
        drift_shell_results, drift_local_times, ivp_result = (
            _do_adaptive_drift_shell_calc(*func_args)
        )

    # Extract drift shell results into arrays which correspond in index to
    # drift_local_times
    drift_rvalues: List[float] = []
    drift_K_results: List[CalculateKResult] = []

    for local_time in drift_local_times:
        drift_rvalues.append(drift_shell_results[local_time][0])
        drift_K_results.append(drift_shell_results[local_time][1])

    # Calculate L*
    # This method assumes a dipole below the innery boundary, and integrates
    # around the local times using stokes law with B = curl A.
    # -----------------------------------------------------------------------
    inner_rvalue = model.inner_boundary
    surface_rvalue = 1

    trace_north_latitudes = np.array(
        [result.trace_latitude.max() for result in drift_K_results], dtype=float
    )

    if ivp_result:
        # Use of adaptive integration supercedes any interpolation 
        integral_axis = ivp_result.t
        integral_theta = None
        integral_integrand = ivp_result.y
        integral = ivp_result.y[0, -1]
    elif interp_local_times:
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
        integral_integrand = np.sin(integral_theta) ** 2.0
        integral = np.trapz(integral_integrand, integral_axis)
    else:
        # Not adaptive, not with interpolation 
        integral_axis = np.zeros(drift_local_times.size + 1)
        integral_axis[:-1] = drift_local_times
        integral_axis[-1] = integral_axis[0] + 2 * np.pi
        
        integral_theta = np.zeros(len(drift_K_results) + 1)
        integral_theta[:-1] = np.pi / 2 - trace_north_latitudes  # colatitude
        integral_theta[-1] = integral_theta[0]

    LStar = 2 * np.pi * (inner_rvalue / surface_rvalue) / integral

    # Return results
    # ------------------------------------------------------------------------
    drift_K = np.array([result.K for result in drift_K_results], dtype=float)
    drift_rvalues_arr = np.array(drift_rvalues)
    drift_is_closed = _test_drift_is_closed(drift_rvalues_arr)

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
        ivp_result=ivp_result,
    )


def _bisect_rvalue_by_K(
    model: MagneticFieldModel,
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
    model : :py:class:`~MagneticFieldModel`
        Grid and magnetic field, loaded using models module
    target_K : float
        K value to search for
    Bm : float
        Magnetic mirroring point, parameter used to estimate K
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
        model
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
    lower_rvalue = model.inner_boundary
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
                model, lower_starting_point, Bm=Bm, step_size=step_size
            )
            upper_result = calculate_K(
                model, upper_starting_point, Bm=Bm, step_size=step_size
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
                model, final_starting_point, Bm=Bm, step_size=step_size
            )

            return final_rvalue, final_result

        # Check for relative error stopping condition ------------------------
        current_starting_point = utils.sp2cart_point(
            r=current_rvalue, phi=-local_time, theta=0
        )

        current_result = calculate_K(
            model, current_starting_point, Bm=Bm, step_size=step_size
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
    model: MagneticFieldModel,
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
      model: reference to grid and magnetic field
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
        the given model (float)
      calculate_K_result: instance of CalculateKResult corresponding to
        radius and calculate_K().
    Raises
      DriftShellBisectionDoesntConverge: maximu number of iterations reached
    """
    # Decide which direction to iterate --------------------------------------
    initial_point = utils.sp2cart_point(r=initial_rvalue, phi=-local_time, theta=0)
    initial_result = calculate_K(
        model, initial_point, pitch_angle=90, step_size=step_size
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
            model, current_point, pitch_angle=90, step_size=step_size
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
            model, current_point, pitch_angle=90, step_size=step_size
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
                model, final_initial_point, pitch_angle=90, step_size=step_size
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
    model: MagneticFieldModel,
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
      model: reference to grid and magnetic field
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
        the given model (float)
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
    initial_result = calculate_K(model, initial_point, Bm=Bm, step_size=step_size)
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
        current_result = calculate_K(model, current_point, Bm=Bm, step_size=step_size)
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
        current_result = calculate_K(model, current_point, Bm=Bm, step_size=step_size)
        current_K = current_result.K

        history_rvalue.append(current_rvalue)
        history_K.append(current_K)

        tmp = sorted(history_K[-2:])
        in_interval = tmp[0] < target_K < tmp[1]

        if in_interval:
            # Interpolate between upper and lower bounds to find final rvalue
            (final_rvalue,) = np.interp([target_K], history_K[-2:], history_rvalue[-2:])
            final_point = utils.sp2cart_point(r=final_rvalue, phi=-local_time, theta=0)
            final_result = calculate_K(model, final_point, Bm=Bm, step_size=step_size)

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


def _ivp_target_fun(local_time, current_state, extra_args):
    """TODO: document this function"""
    # Extract variables from function parameters
    # ------------------------------------------------------------------------
    Bm = extra_args["Bm"]
    drift_shell_results = extra_args["drift_shell_results"]
    K = extra_args["K"]
    major_step = extra_args["major_step"]
    max_iters = extra_args["max_iters"]
    minor_step = extra_args["minor_step"]
    model = extra_args["model"]
    mode = extra_args["mode"]
    trace_step_size = extra_args["trace_step_size"]

    if drift_shell_results:
        start_rvalue, _ = drift_shell_results[max(drift_shell_results.keys())]
    else:
        start_rvalue = extra_args["default_rvalue"]

    # Perform core search for drift shell at this local time
    # ------------------------------------------------------------------------
    if mode == "normal":
        output = _linear_search_rvalue_by_K(
            model,
            K,
            Bm,
            start_rvalue,
            local_time,
            max_iters,
            trace_step_size,
            major_step,
            minor_step,
        )
    elif mode == "equitorial":
        output = _linear_search_rvalue_by_Bmin(
            model,
            Bm,
            start_rvalue,
            local_time,
            max_iters,
            trace_step_size,
            major_step,
            minor_step,
        )
    else:
        raise RuntimeError(f"Code should never reach here: {mode}")

    # Add to reference that accumulates CalculateKResult's (output variable)
    # and return change in integral.
    # ------------------------------------------------------------------------
    drift_shell_results[local_time] = output

    trace_north_latitude = output[1].trace_latitude.max()
    integrand = np.sin(np.pi / 2 - trace_north_latitude) ** 2.0

    return integrand


def _do_basic_drift_shell_calc(
        starting_phi, starting_rvalue, starting_result,
        starting_pitch_angle,
        num_local_times, major_step, max_iters, minor_step,
        model, mode, trace_step_size, first_mlt_step, max_mlt_step,
        integrand_atol, integrand_rtol,
        interval_size_threshold,
        rel_error_threshold,    
):
    drift_local_times = starting_phi + (
        2 * np.pi * np.arange(num_local_times) / num_local_times
    )

    drift_shell_results = {
        drift_local_times[0]: (starting_rvalue, starting_result)
    }

    for i, local_time in enumerate(drift_local_times):
        if i == 0:
            continue

        last_rvalue, _ = drift_shell_results[drift_local_times[i - 1]]

        if mode == "normal":            
            if starting_pitch_angle == 90.0:
                output = _linear_search_rvalue_by_Bmin(
                    model,
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
                    model,
                    starting_result.K,
                    starting_result.Bm,
                    last_rvalue,
                    local_time,
                    max_iters,
                    trace_step_size,
                    major_step,
                    minor_step,
                )
        elif mode == "equitorial":
            if starting_pitch_angle == 90.0:
                raise NotImplementedError(
                    "_bisect_search_rvalue_by_Bmin() not implemented"
                )
                # output = _bisect_search_rvalue_by_Bmin(
                #    model, starting_result.Bm, starting_rvalue, local_time,
                #    max_iters, trace_step_size, major_step, minor_step
                # )
            else:
                output = _bisect_rvalue_by_K(
                    model,
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

        drift_shell_results[local_time] = output
    
    return (
        drift_shell_results,
        drift_local_times,
        None,
    )


def _do_adaptive_drift_shell_calc(
        starting_phi, starting_rvalue, starting_result,
        starting_pitch_angle,
        num_local_times, major_step, max_iters, minor_step,
        model, mode, trace_step_size, first_mlt_step, max_mlt_step,
        integrand_atol, integrand_rtol,
        interval_size_threshold,
        rel_error_threshold,    
):
    drift_shell_results = {}

    extra_args = dict(
        Bm=starting_result.Bm,
        default_rvalue=starting_rvalue,
        drift_shell_results=drift_shell_results,
        K=starting_result.K,
        major_step=major_step,
        max_iters=max_iters,
        minor_step=minor_step,
        model=model,
        mode=mode,
        trace_step_size=trace_step_size,
    )
    ivp_result = solve_ivp(
        fun=_ivp_target_fun,
        t_span=(0, 2 * np.pi),
        y0=(0,),
        method="RK45",
        args=(extra_args,),
        first_step=first_mlt_step,
        max_step=max_mlt_step,
        atol=integrand_atol,
        rtol=integrand_rtol,
    )
    drift_local_times = ivp_result.t

    return (
        drift_shell_results,
        drift_local_times,
        ivp_result
    )
