"""Calculation of adiabatic invariants.

The following methods are key:
  - calculate_K()
"""

from dataclasses import dataclass
from typing import Tuple

from ai import cs
import numpy as np
import pyvista


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
    trace_field_strength: np.array     # field strenth along trace
    integral_axis: np.array            # axis of K integration
    integral_axis_latitude: np.array   # latitude corresponding to integ axis
    integral_integrand: np.array       # integrand of K calculation
    

def calculate_K(mesh, starting_point, mirror_latitude):
    """Calculate the K adiabatic invariants.
    
    Arguments
      mesh: grid and magnetic field, loaded using meshes module
      starting_point: Starting point of the field line integration, as
        (x, y, z) tuple of floats, in units of Re.
      mirror_latitude: Lattitude in degrees to use for the mirroring point
    """
    # Calculate field line trace
    # ------------------------------------------------------------------------
    trace = mesh.streamlines(
        'B', source_center=starting_point,
        terminal_speed=0.0, n_points=1, source_radius=0.1,
        max_step_length=0.005,
        min_step_length=0.005,
        initial_step_length=0.005
    )
    trace_field_strength = np.linalg.norm(trace['B'], axis=1)

    # Get value of magnetic field strength at mirror point, Bm
    # ------------------------------------------------------------------------
    _, trace_latitude, _  = cs.cart2sp(x=trace.points[:, 0],
                                       y=trace.points[:, 1],
                                       z=trace.points[:, 2])
    trace_sorter = np.argsort(trace_latitude)
    Bm = np.interp(x=np.deg2rad(mirror_latitude),
                   xp=trace_latitude[trace_sorter],
                   fp=trace_field_strength[trace_sorter])

    # Sort field line trace points
    # ------------------------------------------------------------------------
    trace_points_sorted = trace.points[trace_sorter]
    trace_field_strength_sorted = trace_field_strength[trace_sorter]    
    
    # Calculate Function Values
    # ------------------------------------------------------------------------    
    Bm_mask = (trace_field_strength_sorted < Bm)

    ds_vec = np.diff(trace_points_sorted[Bm_mask], axis=0)
    ds_scalar = np.linalg.norm(ds_vec, axis=1)

    integral_axis_latitude = np.rad2deg(trace_latitude[trace_sorter][Bm_mask])
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
        trace_field_strength=trace_field_strength_sorted,
        integral_axis=integral_axis,
        integral_axis_latitude=integral_axis_latitude,
        integral_integrand=integral_integrand,
    )
