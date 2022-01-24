"""Calculation of adiabatic invariants.

The following methods are key:
  - calculate_K()
  - calculate_LStar()
"""

from dataclasses import dataclass
from typing import Tuple

from ai import cs
import numpy as np


@dataclass
class CalculateKResult:
    """Class to hold the return value of calculate_K()

    All arrays are sorted by magnetic latitude.
    """
    K: float                           # Second invariant
    Bm: float                          # magnetic mirror strength
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
    drift_local_times: np.array        # magnetic local times of drift shell
    drift_rvalues: np.array            # radius drift shell at local time
    drift_K: np.array                  # drift shell K values 

    
class FieldLineTraceReturnedEmpty(RuntimeError):
    """Raised when a field line trace is performed but the result is empty."""

    
class DriftShellBisectionDoesntConverge(RuntimeError):
    """Raised when Bisection to determine the drift shell doesn't converge."""
    
    
def calculate_K(mesh, starting_point, mirror_latitude=None, Bm=None,
                step_size=None):
    """Calculate the K adiabatic invariant.

    Either mirror_latitude or Bm must be specified.
    
    Arguments
      mesh: grid and magnetic field, loaded using meshes module
      starting_point: Starting point of the field line integration, as
        (x, y, z) tuple of floats, in units of Re.
      mirror_latitude: Lattitude in degrees to use for the mirroring point
      Bm: magnetic field strength at mirroring point
    Returns
      result: instance of CalculateKResult
    """
    # Validate function arguments
    # ------------------------------------------------------------------------
    if (mirror_latitude is None) and (Bm is None):
        raise ValueError('Either of the keyword argumets "mirror_latitude" '
                         'or "Bm" must be specified.')
    elif (mirror_latitude is not None) and (Bm is not None):
        raise ValueError('Only one of the keyword arguments '
                         '"mirror_latitude" or "Bm" must be specified.')

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
        raise FieldLineTraceReturnedEmpty('Trace returned empty')
    
    trace_field_strength = np.linalg.norm(trace['B'], axis=1)

    # Get the trace latitudes and Bm if not specified
    # ------------------------------------------------------------------------
    _, trace_latitude, _  = cs.cart2sp(x=trace.points[:, 0],
                                       y=trace.points[:, 1],
                                       z=trace.points[:, 2])
    trace_sorter = np.argsort(trace_latitude)

    if Bm is None:
        Bm = np.interp(x=np.deg2rad(mirror_latitude),
                       xp=trace_latitude[trace_sorter],
                       fp=trace_field_strength[trace_sorter])
        
    # Sort field line trace points
    # ------------------------------------------------------------------------
    trace_latitude_sorted = trace_latitude[trace_sorter]
    trace_points_sorted = trace.points[trace_sorter]
    trace_field_strength_sorted = trace_field_strength[trace_sorter]    
    
    # Calculate Function Values
    # ------------------------------------------------------------------------    
    Bm_mask = (trace_field_strength_sorted < Bm)

    ds_vec = np.diff(trace_points_sorted[Bm_mask], axis=0)
    ds_scalar = np.linalg.norm(ds_vec, axis=1)

    integral_axis_latitude = np.rad2deg(trace_latitude_sorted[Bm_mask])
    integral_axis = np.array([0] + np.cumsum(ds_scalar).tolist())
    integral_integrand = (Bm - trace_field_strength_sorted[Bm_mask])**(0.5)

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


def calculate_LStar(mesh, starting_point, starting_mirror_latitude,
                    num_local_times=4, rel_error_threshold=0.03,
                    max_iters=100, trace_step_size=None, verbose=False):
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
      rel_error_threshold: bisection search pararameter in [0, 1] (float)
      max_iters: maximum iterations befor erroring out (int)
    Returns
      result: instance of CalculateLStarResult
    """
    # Determine list of local times we will search
    # ------------------------------------------------------------------------
    starting_r, starting_phi, starting_theta = cs.cart2sp(*starting_point)
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
    starting_result = calculate_K(
        mesh, starting_point, mirror_latitude=starting_mirror_latitude,
        step_size=trace_step_size,
    )
    
    # Estimate radius of equivalent K at other local times using bisection
    # method. The first element in the drift_rvalues array is not in the
    # loop because it is not done with bisection.
    # ------------------------------------------------------------------------
    drift_rvalues = np.zeros_like(drift_local_times)
    drift_K = np.zeros_like(drift_local_times)
    
    drift_rvalues[0] = starting_rvalue
    drift_K[0] = starting_result.K
    
    for i, local_time in enumerate(drift_local_times):
        if i == 0:
            continue
        if verbose:
            print(f'Calculating drift radius {i+1}/{drift_rvalues.size}')

        drift_rvalues[i], drift_K[i] = _bisect_rvalue_by_K(
            mesh, starting_result.K, starting_result.Bm,
            starting_rvalue, local_time, starting_theta,
            max_iters, rel_error_threshold, trace_step_size,
        )

    # Return results
    # ------------------------------------------------------------------------    
    return CalculateLStarResult(
        drift_local_times=drift_local_times,
        drift_rvalues=drift_rvalues,
        drift_K=drift_K
    )
    

def _bisect_rvalue_by_K(mesh, target_K, Bm, starting_rvalue, local_time,
                        starting_theta, max_iters, rel_error_threshold,
                        step_size):
    """Internal helper function to calculate_LStar(). Applies bisection method
    to find an radius with an equal K.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      target_K: floating point K value to search for (float)
      Bm: magnetic mirroring point; parameter used to estimte K (float)
      starting_rvalue: starting point rvalue (float)
      local_time: starting local time (radians, float)
      starting_theta: starting latitude (radians, float)
      max_iters: Maximum number of iterations before erroring out (int)
      rel_error_threshold: Relative error threshold to consider two K's equal
        (float between [0, 1]).
    Returns
      rvalue: lshell number at given local time which produces the same K on 
        the given mesh (float)
    Raises
      RuntimeError: maximum number of iterations reached
    """
    # Perform bisection method. If you are not sure what this is, it is
    # advised you read about bisection on wikipedia first.
    # ------------------------------------------------------------------------
    upper_rvalue = starting_rvalue * 2
    lower_rvalue = np.linalg.norm(mesh.points, axis=1).min()
    current_rvalue = starting_rvalue
    rel_errors = []
    
    for _ in range(max_iters):
        # print(lower_rvalue, upper_rvalue, current_rvalue)
        
        current_starting_point = cs.sp2cart(
            r=current_rvalue, phi=local_time, theta=starting_theta
        )
        current_result = calculate_K(mesh, current_starting_point, Bm=Bm,
                                     step_size=step_size)
        rel_error = (
            abs(target_K - current_result.K) /
            target_K
        )
        
        if rel_error < rel_error_threshold:
            # match found!
            return current_rvalue, current_result.K
        elif current_result.K < target_K:
            # too low!
            rel_errors.append((rel_error, 'too_low', current_rvalue))
                               
            current_rvalue, lower_rvalue = \
                ((upper_rvalue + current_rvalue) / 2, current_rvalue)
            # print('Too Low!')
        else:
            # too high!
            rel_errors.append((rel_error, 'too_high', current_rvalue))

            current_rvalue, upper_rvalue = \
                ((lower_rvalue + current_rvalue) / 2, current_rvalue)
            # print('Too high!')
        
    # If the code reached this point, the maximum number of iterations
    # was exhausted.
    # ------------------------------------------------------------------------
    raise DriftShellBisectionDoesntConverge(
        f'Maximum number of iterations {max_iters} reached for local time '
        f'{local_time:.1f} during bisection ' + repr(rel_errors)
    )    
