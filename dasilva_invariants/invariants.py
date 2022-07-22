"""Calculation of adiabatic invariants.

The following methods are key:
  - calculate_K()
  - calculate_LStar()
"""

from dataclasses import dataclass
from typing import Tuple

from ai import cs
import numpy as np
import pyvista
from scipy import interpolate
import vtk


@dataclass
class CalculateKResult:
    """Class to hold the return value of calculate_K()

    All arrays are sorted by magnetic latitude.
    """
    K: float                           # Second adiabatic invariant (K)
    Bm: float                          # magnetic mirror strength used
    mirror_latitude: float             # MLAT at which particle mirrors
    starting_point: Tuple[float, float, float]

    trace_points: np.array             # trace points
    trace_latitude: np.array           # trace latitude
    trace_field_strength: np.array     # field strenth along trace
    integral_axis: np.array            # axis of K integration
    integral_axis_latitude: np.array   # latitude corresponding to integ axis
    integral_integrand: np.array       # integrand of K calculation


@dataclass
class CalculateLStarResult:
    """Class to hold the return value of calculate_LStar()"""
    LStar: float                       # Third adiabatic invariant (L*)       
    drift_local_times: np.array        # magnetic local times of drift shell
    drift_rvalues: np.array            # radius drift shell at local time
    drift_K: np.array                  # drift shell K values, shorthand
    drift_K_results: np.array          # drift shell results from calculate_K()
    integral_axis: np.array            # integration axis local time (radians) 
    integral_theta: np.array           # integration theta variable
    integral_integrand: np.array       # integration integran


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
    mesh, starting_point, mirror_latitude=None, Bm=None, pitch_angle=None,
    step_size=None
):
    """Calculate the K adiabatic invariant.

    Either mirror_latitude, Bm, or pitch_angle must be specified.

    Arguments
      mesh: grid and magnetic field, loaded using meshes module
      starting_point: Starting point of the field line integration, as
        (x, y, z) tuple of floats, in units of Re.
      mirror_latitude: Lattitude in degrees to use for the mirroring point
      Bm: magnetic field strength at mirroring point
      pitch_angle: pitch angle in degrees
    Returns
      result: instance of CalculateKResult
    Raises
      FieldLineTraceInsufficient: field line trace empty or too small
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
        max_step_length = 1e-2
        min_step_length = 1e-2
        initial_step_length = 1e-2
    else:
        max_step_length = step_size
        min_step_length = step_size
        initial_step_length = step_size

    trace = mesh.streamlines(
        'B',
        start_position=starting_point,
        terminal_speed=0.0,
        max_step_length=max_step_length,
        min_step_length=min_step_length,
        initial_step_length=initial_step_length,
        step_unit='l',
        max_steps=1_000_000,
        interpolator_type='c'
    )

    if trace.n_points == 0:
        raise FieldLineTraceInsufficient('Trace returned empty')
    if trace.n_points < 50:
        raise FieldLineTraceInsufficient(
            f'Trace too small ({trace.n_points} points)'
        )

    trace_field_strength = np.linalg.norm(trace['B'], axis=1)

    # Get the trace latitudes and Bm if not specified
    # ------------------------------------------------------------------------
    _, trace_latitude, _ = cs.cart2sp(x=trace.points[:, 0],
                                      y=trace.points[:, 1],
                                      z=trace.points[:, 2])
    trace_sorter = np.argsort(trace_latitude)

    if mirror_latitude is not None:
        Bm = np.interp(x=np.deg2rad(mirror_latitude),
                       xp=trace_latitude[trace_sorter],
                       fp=trace_field_strength[trace_sorter])
    elif pitch_angle is not None:
        Bmin = trace_field_strength.min()
        Bm = Bmin / np.sin(np.deg2rad(pitch_angle))**2

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
    Bm_mask = (trace_field_strength_sorted < Bm)

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
        mirror_latitude=mirror_latitude,
        starting_point=starting_point,

        trace_points=trace_points_sorted,
        trace_latitude=trace_latitude_sorted,
        trace_field_strength=trace_field_strength_sorted,
        integral_axis=integral_axis,
        integral_axis_latitude=integral_axis_latitude,
        integral_integrand=integral_integrand,
    )


def calculate_LStar(
    mesh, starting_point, starting_mirror_latitude=None, Bm=None,
    starting_pitch_angle=None, num_local_times=4, interp_local_times=True,
    interp_npoints=1024, interval_size_threshold=0.1,
    rel_error_threshold=0.01, max_iters=100, trace_step_size=None,
    verbose=1000
):
    """Calculate the L* adiabatic invariant.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      starting_point: Starting point of the field line integration, as
        (x, y, z) tuple of floats, in units of Re.
      starting_mirror_latitude: Lattitude in degrees to use for the
        mirroring point for the local time associated with the starting
        point (float)
      num_local_times: Number of local times spaced evenly around the
        drift shell to solve for with bisection (int)
      interp_local_times: Interpolate intersection latitudes for local times
        with cubic splines to allow for less local time calculation.
      interp_npoints: Number of points to use in interplation, only active
        if interp_local_times=True.
      interval_size_threshold: bisection threshold before linearly
        interpolating
      max_iters: maximum iterations before erroring out (int)
      trace_step_size: undocumented
      verbose: verbosity level, see joblib.Parallel for more info
    Returns
      result: instance of CalculateLStarResult
    """
    # Determine list of local times we will search
    # ------------------------------------------------------------------------
    starting_r, starting_theta, starting_phi = cs.cart2sp(*starting_point)
    drift_local_times = starting_phi + (
        2 * np.pi * np.arange(num_local_times) / num_local_times
    )

    # Calculate K at the current local time.
    # ------------------------------------------------------------------------
    if verbose:
        print(f'Calculating drift radius 1/{drift_local_times.size}')

    starting_rvalue, _, _ = cs.cart2sp(
        x=starting_point[0], y=starting_point[1], z=0
    )

    if Bm is not None:
        kwargs = {'Bm': Bm}
    elif starting_mirror_latitude is not None:
        kwargs = {'mirror_latitude': starting_mirror_latitude}
    elif starting_pitch_angle is not None:
        kwargs = {'pitch_angle': starting_pitch_angle}

    starting_result = calculate_K(
        mesh, starting_point, step_size=trace_step_size, **kwargs
    )

    # Estimate radius of equivalent K/Bm at other local times using method
    # based on whether the particle is equitorial mirroring or not (if it is,
    # a trace is not required and can be skipped).
    # ------------------------------------------------------------------------
    drift_shell_results = [(starting_rvalue, starting_result)]

    for i, local_time in enumerate(drift_local_times):
        if i == 0:
            continue

        starting_rvalue, _ = drift_shell_results[-1]

        result = _bisect_rvalue_by_K(
            starting_result.K, starting_result.Bm,
            starting_rvalue, local_time, starting_theta,
            max_iters, interval_size_threshold, rel_error_threshold,
            trace_step_size,
            mesh=mesh
        )

        drift_shell_results.append(result)

    # Extract drift shell results into arrays which correspond in index to
    # drift_local_times
    drift_rvalues = np.zeros_like(drift_local_times)
    drift_K_results = np.zeros_like(drift_local_times, dtype=object)

    for i, result in enumerate(drift_shell_results):
        drift_rvalues[i], drift_K_results[i] = result

    # Calculate L*
    # This method assumes a dipole below the innery boundary, and integrates
    # around the local times using stokes law with B = curl A.
    # -----------------------------------------------------------------------
    inner_rvalue = np.linalg.norm(mesh.points, axis=1).min()
    surface_rvalue = 1

    trace_north_latitudes = np.array(
        [result.trace_latitude.max() for result in drift_K_results],
        dtype=float
    )

    if interp_local_times:
        # Interpolate with cubic spline with periodic boundary condition
        # that forces the 1st and 2nd derivatives to be equal at the first
        # and last points
        spline_x = np.zeros(drift_K_results.size + 1)
        spline_y = np.zeros(drift_K_results.size + 1)

        spline_x[:-1] = drift_local_times
        spline_x[-1] = drift_local_times[0] + 2 * np.pi

        spline_y[:-1] = trace_north_latitudes
        spline_y[-1] = trace_north_latitudes[0]

        spline = interpolate.CubicSpline(spline_x, spline_y, bc_type='periodic')

        integral_axis = np.linspace(spline_x.min(),
                                    spline_x.max(),
                                    interp_npoints)
        integral_theta = np.pi/2 - spline(integral_axis)  # colatitude
    else:
        integral_axis = np.zeros(drift_local_times.size + 1)
        integral_axis[:-1] = drift_local_times
        integral_axis[-1] = integral_axis[0] + 2 * np.pi

        integral_theta = np.zeros(drift_K_results.size + 1)
        integral_theta[:-1] = np.pi/2 - trace_north_latitudes  # colatitude
        integral_theta[-1] = integral_theta[0]

    integral_integrand = np.sin(integral_theta)**2
    integral = np.trapz(integral_integrand, integral_axis)
    LStar = 2 * np.pi * (inner_rvalue / surface_rvalue) / integral

    # Return results
    # ------------------------------------------------------------------------
    drift_K = np.array([result.K for result in drift_K_results], dtype=float)

    return CalculateLStarResult(
        LStar=LStar,
        drift_local_times=drift_local_times,
        drift_rvalues=drift_rvalues,
        drift_K=drift_K,
        drift_K_results=drift_K_results,
        integral_axis=integral_axis,
        integral_theta=integral_theta,
        integral_integrand=integral_integrand
    )


def _bisect_rvalue_by_K(
    target_K, Bm, starting_rvalue, local_time, starting_theta, max_iters,
    interval_size_threshold, rel_error_threshold, step_size, mesh=None,
    mesh_tempfile=None
):
    """Internal helper function to calculate_LStar(). Applies bisection method
    to find an radius with an equal K, stopping when either relative error
    is sufficiently small or interval being searched is sufficiently small to
    interpolate.

    Only one of mesh or mesh_tempfile is required. Use tempfiles for mutli-
    processing, using direct reference (mesh) for threading.

    Args
      target_K: floating point K value to search for (float)
      Bm: magnetic mirroring point; parameter used to estimte K (float)
      starting_rvalue: starting point rvalue (float)
      local_time: starting local time (radians, float)
      starting_theta: starting latitude (radians, float)
      max_iters: Maximum number of iterations before erroring out (int)
      interval_size_threshold: bisection threshold before linearly
        interpolating
      rel_error_threshold: Relative error threshold to consider two K's equal
        (float between [0, 1]).
      mesh: reference to grid and magnetic field
      mesh_tempfile: path to grid and magnetic field, loaded using pyvsita
    Returns
      rvalue: radius at given local time which produces the same K on 
        the given mesh (float)
      calculate_k_result: instance of CalculateKResult corresponding to
        radius and calculate_K().
    Raises
      DriftShellBisectionDoesntConverge: maximum number of iterations reached
    """
    # Perform bisection method searching for K(Bm, r)
    # ------------------------------------------------------------------------
    assert (mesh is not None) or (mesh_tempfile is not None), \
        'One of mesh= or mesh_tempfile= is required'

    if mesh_tempfile:
        mesh = pyvista.read(mesh_tempfile)

    upper_rvalue = starting_rvalue * 2
    lower_rvalue = np.linalg.norm(mesh.points, axis=1).min()
    current_rvalue = starting_rvalue
    history = []

    for _ in range(max_iters):
        # Check for interval size stopping condition -------------------------
        interval_size = upper_rvalue - lower_rvalue

        if interval_size < interval_size_threshold:
            # Calcualte K and upper and lower bounds
            upper_starting_point = cs.sp2cart(
                r=upper_rvalue, phi=local_time, theta=starting_theta)
            upper_result = calculate_K(
                mesh, upper_starting_point, Bm=Bm, step_size=step_size)
            lower_starting_point = cs.sp2cart(
                r=lower_rvalue, phi=local_time, theta=starting_theta)
            lower_result = calculate_K(
                mesh, lower_starting_point, Bm=Bm, step_size=step_size)

            # Interpolate between upper and lower bounds to find final rvalue
            final_rvalue = np.interp(
                target_K,
                [upper_result.K, lower_result.K],
                [upper_rvalue, lower_rvalue])
            final_starting_point = cs.sp2cart(
                r=final_rvalue, phi=local_time, theta=starting_theta)
            final_result = calculate_K(
                mesh, final_starting_point, Bm=Bm, step_size=step_size)

            return final_rvalue, final_result\

        # Check for relative error stopping condition ------------------------
        current_starting_point = cs.sp2cart(
            r=current_rvalue, phi=local_time, theta=starting_theta)
        current_result = calculate_K(
            mesh, current_starting_point, Bm=Bm, step_size=step_size)

        if target_K == 0 and current_result.K == 0:
            rel_error = 0
        else:
            rel_error = (
                abs(target_K - current_result.K) /
                max(np.abs(target_K), np.abs(current_result.K))
            )

        if rel_error < rel_error_threshold:
            # match found
            return current_rvalue, current_result

        # Continue iterating by halving the interval -------------------------
        if current_result.K < target_K:
            # too low
            history.append((interval_size, 'too_low', current_rvalue))

            current_rvalue, lower_rvalue = \
                ((upper_rvalue + current_rvalue) / 2, current_rvalue)
        else:
            # too high
            history.append((interval_size, 'too_high', current_rvalue))

            current_rvalue, upper_rvalue = \
                ((lower_rvalue + current_rvalue) / 2, current_rvalue)

    # If the code reached this point, the maximum number of iterations
    # was exhausted.
    # ------------------------------------------------------------------------
    raise DriftShellBisectionDoesntConverge(
        f'Maximum number of iterations {max_iters} reached for local time '
        f'{local_time:.1f} during bisection ' + repr(history)
    )
