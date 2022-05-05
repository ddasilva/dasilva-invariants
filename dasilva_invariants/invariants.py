"""Calculation of adiabatic invariants.

The following methods are key:
  - calculate_K()
  - calculate_LStar()
"""

from dataclasses import dataclass
import os
import tempfile
from typing import Tuple

from ai import cs
from joblib import delayed, Parallel
import numpy as np
import pyvista
from scipy import interpolate
import vtk


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
    LStar: float                       # Third adiabatic invariant (L*)       
    drift_local_times: np.array        # magnetic local times of drift shell
    drift_rvalues: np.array            # radius drift shell at local time
    drift_K: np.array                  # drift shell K values, shorthand
    drift_K_results: np.array          # drift shell results from calculate_K()
    integral_axis: np.array            # integration axis local time (radians) 
    integral_theta: np.array           # integration theta variable
    integral_integrand: np.array       # integration integran

    
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
                    num_local_times=12, interp_local_times=True,
                    interp_npoints=1024, rel_error_threshold=0.03, n_jobs=-1,
                    max_iters=100, trace_step_size=None, verbose=1000):
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
      n_jobs: number of jobs to run in parallel (-1 for all cores). doesn't
        parallelize that well.
      rel_error_threshold: bisection search pararameter in [0, 1] (float)       
      max_iters: maximum iterations befor erroring out (int)
      trace_step_siz
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
    starting_result = calculate_K(
        mesh, starting_point, mirror_latitude=starting_mirror_latitude,
        step_size=trace_step_size,
    )
    
    # Estimate radius of equivalent K/Bm at other local times using bisection
    # method. The first element in the drift_rvalues array is not in the
    # loop because it is not done with bisection.
    # ------------------------------------------------------------------------
    # Configure following code based on whether parallel processing is used
    # For multirpocessing, avoid pickling (serialize with pyvsita), otherwise
    # faster to just send reference to thread.
    if n_jobs == 1:
        mesh_kwargs = {'mesh': mesh}
        parallel_processor = Parallel(
            verbose=verbose, n_jobs=1, batch_size=1, backend='threading'
        )        
    else:
        # avoid pickling meshes; instead pass filename        
        _, mesh_tempfile = tempfile.mkstemp(suffix='.vtk')
        mesh.save(mesh_tempfile)
        mesh_kwargs = {'mesh_tempfile': mesh_tempfile}
        parallel_processor = Parallel(
            verbose=verbose, n_jobs=n_jobs, batch_size=1,
            backend='multiprocessing'
        )

    # Create tasks -----------------------------------------------------------
    tasks = [] 
    
    for i, local_time in enumerate(drift_local_times):
        if i == 0:
            continue

        if starting_mirror_latitude == 0:
            # Special case for equitorial mirroring particles-- only need
            # to search for Bm
            task = delayed(_search_rvalue_by_Bm)(
                starting_result.Bm, starting_rvalue, local_time, max_iters,
                rel_error_threshold, trace_step_size,                
                **mesh_kwargs
            )
        else:            
            task = delayed(_bisect_rvalue_by_K)(
                starting_result.K, starting_result.Bm,
                starting_rvalue, local_time, starting_theta,
                max_iters, rel_error_threshold, trace_step_size,
                **mesh_kwargs
            )

        tasks.append(task)

    parallel_results = parallel_processor(tasks)

    # Parallel processing cleanup --------------------------------------------
    if n_jobs > 1:
        os.remove(mesh_tempfile)
    
    # Extract parallel results into arrays which correspond in index to
    # drift_local_times
    drift_rvalues = np.zeros_like(drift_local_times)
    drift_K_results = np.zeros_like(drift_local_times, dtype=object)

    drift_rvalues[0] = starting_rvalue
    drift_K_results[0] = starting_result

    for i, parallel_result in enumerate(parallel_results):
        drift_rvalues[i + 1], drift_K_results[i + 1] = parallel_result

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


def _search_rvalue_by_Bm(
        target_Bm, starting_rvalue, local_time, max_iters, rel_error_threshold,
        step_size, mesh=None, mesh_tempfile=None):
    """Internal helper function to calculate_LStar(). Applies linear search method
    to find an radius with an B(r) = Bm for equitorial mirroring particles. 

    Args

    Returns
      rvalue: radius at given local time which produces the same K on 
        the given mesh (float)
      calculate_k_result: instance of CalculateKResult corresponding to
        radius and calculate_K().
    Raises
      RuntimeError: maximum number of iterations reached
    """
    # Perform bisection method searching for Bm(r)
    # ------------------------------------------------------------------------
    assert (mesh is not None) or (mesh_tempfile is not None), \
        'One of mesh= or mesh_tempfile= is required'
    
    if mesh_tempfile:
        mesh = pyvista.read(mesh_tempfile)

    # Interpolate points 25% inside and 50% farther out the nominal rvalue
    # and search for closest B
    rvalues = np.arange(0.75 * starting_rvalue,
                        1.25 * starting_rvalue,
                        0.001 * starting_rvalue)
    local_times = np.array([local_time] * rvalues.size)
    latitudes = np.array([0] * rvalues.size)
    
    points_search = pyvista.PolyData(np.array(cs.sp2cart(
        r=rvalues, phi=local_times, theta=latitudes
    )).T)

    interp = vtk.vtkPointInterpolator()  # uses linear interpolation by default
    interp.SetInputData(points_search)
    interp.SetSourceData(mesh)
    interp.Update()

    points_interp = pyvista.PolyData(interp.GetOutput())

    # Search for closet point and return trace at that point
    B_search = np.linalg.norm(points_interp['B'], axis=1)
    i = np.argmin(np.abs(target_Bm - B_search))

    rel_error = np.abs(B_search[i] - target_Bm) / target_Bm

    if rel_error > rel_error_threshold:
        raise DriftShellBisectionDoesntConverge('Could not find Bm!')

    rvalue = np.linalg.norm(points_interp.points[i, :])    
    result = calculate_K(mesh, points_interp.points[i, :], Bm=target_Bm,
                         step_size=step_size)

    return rvalue, result


def _bisect_rvalue_by_K(target_K, Bm, starting_rvalue, local_time,
                        starting_theta, max_iters, rel_error_threshold,
                        step_size, mesh=None, mesh_tempfile=None):
    """Internal helper function to calculate_LStar(). Applies bisection method
    to find an radius with an equal K.

    Only one of mesh or mesh_tempfile is required. Use tempfiles for mutli-
    processing, using direct reference (mesh) for threading.

    Args
      target_K: floating point K value to search for (float)
      Bm: magnetic mirroring point; parameter used to estimte K (float)
      starting_rvalue: starting point rvalue (float)
      local_time: starting local time (radians, float)
      starting_theta: starting latitude (radians, float)
      max_iters: Maximum number of iterations before erroring out (int)
      rel_error_threshold: Relative error threshold to consider two K's equal
        (float between [0, 1]).
      mesh: reference to grid and magnetic field
      mesh_tempfile: path to grid and magnetic field, loaded using 
        pyvsita
    Returns
      rvalue: radius at given local time which produces the same K on 
        the given mesh (float)
      calculate_k_result: instance of CalculateKResult corresponding to
        radius and calculate_K().
    Raises
      RuntimeError: maximum number of iterations reached
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
    rel_errors = []
    
    for _ in range(max_iters):
        # print(lower_rvalue, upper_rvalue, current_rvalue)  
        current_starting_point = cs.sp2cart(
            r=current_rvalue, phi=local_time, theta=starting_theta
        )
        current_result = calculate_K(mesh, current_starting_point, Bm=Bm,
                                     step_size=step_size)

        if target_K == 0 and current_result.K == 0:
            rel_error = 0
        else:
            rel_error = (
                abs(target_K - current_result.K) /
                max(np.abs(target_K), np.abs(current_result.K))
            )
        
        if rel_error < rel_error_threshold:
            # match found!
            return current_rvalue, current_result
        elif current_result.K < target_K:
            # too low!
            rel_errors.append((rel_error, 'too_low', current_rvalue))
                               
            current_rvalue, lower_rvalue = \
                ((upper_rvalue + current_rvalue) / 2, current_rvalue)
        else:
            # too high!
            rel_errors.append((rel_error, 'too_high', current_rvalue))

            current_rvalue, upper_rvalue = \
                ((lower_rvalue + current_rvalue) / 2, current_rvalue)
        
    # If the code reached this point, the maximum number of iterations
    # was exhausted.
    # ------------------------------------------------------------------------
    raise DriftShellBisectionDoesntConverge(
        f'Maximum number of iterations {max_iters} reached for local time '
        f'{local_time:.1f} during bisection ' + repr(rel_errors)
    )    
