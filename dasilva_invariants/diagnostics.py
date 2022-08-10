"""Diagnostic routines with visualizations for quick look analysis at
magnetic field models.

To run all of these, use the do_all() function.
"""
from typing import Any, Dict, Tuple

from astropy import constants, units
from matplotlib.colors import LogNorm, Normalize
import pylab as plt
import numpy as np
from numpy.typing import NDArray
import pyvista
import seaborn as sns

from . import invariants, utils
from .constants import EARTH_DIPOLE_B0

# Default limits of the magnetosphere cut plots (equitorial and meridional).
# Note that the "X" and "Y" here are the axis of the matplotlib plot, not SM.
DEFAULT_MSPHERE_XLIM = (15, -15)
DEFAULT_MSPHERE_YLIM = (-11.5, 11.5)


def tail_traces(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    th: float = 180.0,
    r_min: float = 3.0,
    r_max: float = 9.0
) -> None:
    """Visualizes traces in the magnetotail in the X/Z plane.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      th: Rotation in degrees of the sun-earth line
      r_min: Radius of first trace
      r_max: Radius of last trace
    """
    plt.figure(figsize=(8, 5))
    rs = np.arange(r_min, r_max + 1)

    for r in rs:
        x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
        result = invariants.calculate_K(mesh, (x, y, 0), 7.5)

        plt.scatter(x=result.trace_points[:, 0],
                    y=result.trace_points[:, 2],
                    c=np.log10(result.trace_field_strength),
                    s=.3)
    
        i = np.argmin(result.trace_field_strength)
        plt.plot(result.trace_points[i, 0],
                 result.trace_points[i, 2], 'kx')

    cb = plt.colorbar()
    cb.set_label('Log10(B) (Log-Gauss)')
    plt.xlabel('X (Re)')
    plt.ylabel('Z (Re)')
    plt.grid(color='#ccc', linestyle='dashed')
    plt.title(f'{model_title}\nField Line Trace between '
              f'R={rs[0]} and R={rs[-1]}')
    plt.xlim(plt.xlim()[::-1])


def tail_K_vs_radius(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    th: float = 180.0,
    pitch_angle: float = 30.0,
    starting_radius: float = 8.0
) -> None:
    """Visualize K versus R as the radius is stepped out in the tail.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      th: Rotation in degrees of the sun-earth line    
    """
    rs = np.linspace(3, 15, 100)
    Ks = np.zeros_like(rs)

    x = np.cos(np.deg2rad(-th)) * starting_radius
    y = np.sin(np.deg2rad(-th)) * starting_radius
    Bm = invariants.calculate_K(mesh, (x, y, 0), pitch_angle=pitch_angle).Bm

    for i, r in enumerate(rs):
        x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
        Ks[i] = invariants.calculate_K(mesh, (x, y, 0), Bm=Bm).K

    plt.plot(rs, Ks, '.-', label=f'Pitch angle {pitch_angle}')
    plt.title(f'{model_title}\nK vs Radius in Tail')
    plt.xlabel('Radius (Re)')
    plt.ylabel('K(r)')


def tail_Bmin_vs_radius(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    th: float = 180.0
) -> None:
    """Visualize K versus R as the radius is stepped out in the tail.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      th: Rotation in degrees of the sun-earth line    
    """
    rs = np.linspace(3, 15, 100)
    Bmins = np.zeros_like(rs)

    for i, r in enumerate(rs):
        x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
        Bmins[i] = invariants.calculate_K(mesh, (x, y, 0), pitch_angle=90).Bm

    plt.plot(rs, Bmins, '.-')
    plt.title(f'{model_title}\nBmin vs Radius in Tail')
    plt.xlabel('Radius (Re)')
    plt.ylabel('Bmin')

    
def drift_shells(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    th: float = 180.0,
    r_min: float = 3.0,
    r_max: float = 9.0,
    pa: float = 90.0,
    num_local_times: int = 50
) -> None:
    """Visualize drift shells calculated from calculate_Lstar.
    
    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      th: Rotation in degrees of the sun-earth line for starting point.
      r_min: Radius of first drift stell starting point
      r_max: Radius of last drift shell starting point
    """
    # Calculate Drift Shells
    rs = np.arange(r_min, r_max + 1)
    results = []

    for r in rs:
        x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
        result = invariants.calculate_LStar(
            mesh, (x, y, 0), starting_pitch_angle=pa,
            num_local_times=num_local_times, verbose=False
        )
        results.append((r, result))

    # Plot drift shells
    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap('viridis')

    for i, (r, result) in enumerate(results):
        x = np.cos(result.drift_local_times) * result.drift_rvalues
        y = np.sin(result.drift_local_times) * result.drift_rvalues

        x = x.tolist() + [x[0]]
        y = y.tolist() + [y[0]]

        plt.plot(x, y, 'o-', label=f'R = {r:.1f} Re', color=cmap(i/len(results)))
        plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(color='#ccc', linestyle='dashed')
        plt.title(f'{model_title}\nDrift Shells between R={rs[0]} '
                  f'and R={rs[-1]}')
        plt.xlabel('X (Re)')
        plt.ylabel('Y (Re)')
        plt.xlim(plt.xlim()[::-1])
        circle = plt.Circle((0, 0), 1, color='b')
        plt.gca().add_patch(circle)
        plt.tight_layout()


def dayside_field_intensities(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    th: float = 0.0,
    r_min: float = 3.0,
    r_max: float = 7.0
) -> None:
    """Do field line traces at increasing radiuses and plot field intensity versus
    position in trace.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      th: Rotation in degrees of the sun-earth line
      r_min: Radius of first trace
      r_max: Radius of last trace
    """    
    plt.figure(figsize=(9, 6))
    rs = np.arange(r_min, r_max + 1)
    cmap = plt.get_cmap('viridis')

    for i, r in enumerate(rs):
        x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
        result = invariants.calculate_K(mesh, (x, y, 0), 7.5, step_size=None)
        rel_position = np.arange(result.trace_latitude.size, dtype=float)
        rel_position /= result.trace_latitude.size

        plt.plot(rel_position, result.trace_field_strength, ',-',
                 label=f'r={r}', color=cmap(i/rs.size))

    plt.xlabel('Relative Position in Trace')

    plt.ylabel('|B| (Gauss)')
    plt.title(f'{model_title}\n'
              f'with Bm at 7.5 deg')
    plt.yscale('log')
    plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(color='#ccc', linestyle='dashed')

    
def equitorial_plot_of_intensity(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    arr_name: str = 'B',
    norm: Any = LogNorm(vmin=1e-5, vmax=1e-3),
    cmap: Any = 'viridis',
    cbar_label='|B| (G)',
    xlim: Tuple[float, float] = DEFAULT_MSPHERE_XLIM,
    ylim: Tuple[float, float] = DEFAULT_MSPHERE_YLIM
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot equitorial plot of scalar quantity or normed vector quantity.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      arr_name: Name of array in mesh. If scalar, will be used as is. If
        vector, will be normed.
      norm: Matplotlib LogNorm instance (or None) to set colorbar limits
      cmap: string colormap or matplotlib colormap instance
      cbar_label: string colorbar label
    Returns
      fig: matplotlib figure generated
      ax: matplotlib axes generated
    """
    Xeq = utils.lfm_get_eq_slice(mesh.x)
    Yeq = utils.lfm_get_eq_slice(mesh.y)

    if len(mesh[arr_name].shape) == 1:
        field = mesh[arr_name]
    else:
        field = np.linalg.norm(mesh[arr_name], axis=1)

    s = mesh.x.shape
    F = np.reshape(field.ravel(), s, order='F')   # field
    Feq = utils.lfm_get_eq_slice(F)

    plt.figure(figsize=(12, 7))
    plt.title(f'{model_title}\nEquitorial Slice', fontsize=18,
              fontweight='bold')
    plt.pcolor(Xeq, Yeq, Feq, norm=norm, cmap=cmap)

    plt.xlabel('X SM (Re)', fontsize=16)
    plt.ylabel('Y SM (Re)', fontsize=16)
    plt.colorbar().set_label(cbar_label, fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)

    return plt.gcf(), plt.gca()
    

def meridional_plot_of_intensity(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    arr_name: str = 'B',
    norm: Any = LogNorm(vmin=5e-5, vmax=1e-3),
    cmap: Any = 'viridis',
    cbar_label: str = '|B| (G)',
    xlim: Tuple[float, float] = DEFAULT_MSPHERE_XLIM,
    ylim: Tuple[float, float] =DEFAULT_MSPHERE_YLIM
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot meridional plot of scalar quantity or normed vector quantity.
    
    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      arr_name: Name of array in mesh. If scalar, will be used as is. If
        vector, will be normed.
      norm: Matplotlib LogNorm instance (or None) to set colorbar limits
      cmap: string colormap or matplotlib colormap instance
      cbar_label: string colorbar label
      include_traces: Include l-shell traces for these L numbers
    Returns
      fig: matplotlib figure generated
      ax: matplotlib axes generated
    """
    Xmer = utils.lfm_get_mer_slice(mesh.x)
    Zmer = utils.lfm_get_mer_slice(mesh.z)

    if len(mesh[arr_name].shape) == 1:
        field = mesh[arr_name]
    else:
        field = np.linalg.norm(mesh[arr_name], axis=1)
    
    s = mesh.x.shape
    F = np.reshape(field.ravel(), s, order='F')   # field
    Fmer = utils.lfm_get_mer_slice(F)
    
    plt.figure(figsize=(12, 7))
    plt.pcolor(Xmer, Zmer, Fmer, norm=norm, cmap=cmap)
                
    plt.title(f'{model_title} - Meridional Slice', fontsize=18, fontweight='bold')    
    plt.xlabel('X SM (Re)', fontsize=16)
    plt.ylabel('Z SM (Re)', fontsize=16)
    plt.colorbar().set_label(cbar_label, fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)

    return plt.gcf(), plt.gca()
    

def K_integrand_plot(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    r: float = 7.0,
    th: float = 180.0
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot K integrand versus integration axis.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      r: starting radius
      th: starting angle (clockwise)
    Returns
      fig: matplotlib figure generated
      ax: matplotlib axes generated
    """
    mirror_deg = 7.5
    x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
    result = invariants.calculate_K(mesh, (x, y, 0), mirror_deg)

    plt.figure(figsize=(8, 4))
    plt.plot(result.integral_axis_latitude, result.integral_integrand, 'k.-')
    plt.fill_between(result.integral_axis_latitude,
                     result.integral_integrand.min(),
                     result.integral_integrand)  # type: ignore
    plt.title(f'{model_title}\n'
              f'K = {result.K:.6f} Re Sqrt(G) = '
              r'$\int_{s_m}^{s_m}\sqrt{B_m - B(s)}ds$'
              f'\n'
              f'Bm = {result.Bm:.6f} G\n'
              f'Mirror at {mirror_deg:.1f} deg',
              fontsize=20)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel(r'$\sqrt{B_m - B(s)}$', fontsize=20)

    return plt.gcf(), plt.gca()

    
def LStar_integrand_plot(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    r: float = 7.0,
    th: float = 0,
    LStar_kwargs: Dict[str, Any] = {'mirror_deg': 7.5}
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot LStar integrand versus integration axis.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
    Returns
      fig: matplotlib figure generated
      ax: matplotlib axes generated
    """
    x, y = np.cos(np.deg2rad(-th)) * r, np.sin(np.deg2rad(-th)) * r
    result = invariants.calculate_LStar(
        mesh, (x, y, 0), **LStar_kwargs)

    plt.figure(figsize=(8, 4))
    plt.plot(result.integral_axis, result.integral_integrand, 'k.-')
    
    plt.fill_between(
        result.integral_axis, [plt.ylim()[0]],
        result.integral_integrand)  # type: ignore

    trace_north_latitudes = np.array(
        [res.trace_latitude.max() for res in result.drift_K_results],
        dtype=float
    )

    for i, local_time in enumerate(result.drift_local_times):
        plt.plot(local_time, np.sin(np.pi/2 - trace_north_latitudes[i])**2, 'x', color='lime',
                 markersize=10)

    plt.title(f'{model_title}\n'
              f'L* = {result.LStar:.6f} = '
              r'$2 \pi (R_{in}/R_E) / \int_{0}^{2\pi} sin^2(\theta) d\phi$',
              fontsize=20)
    plt.xlabel('Local time (radians about sun-earth line)', fontsize=20)
    plt.ylabel(r'$sin^2(\theta)$', fontsize=20)

    return plt.gcf(), plt.gca()


def meridional_plot_of_current(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    empty_theta_region_threshold: float = 2.0,
    xlim: Tuple[float, float] = DEFAULT_MSPHERE_XLIM,
    ylim: Tuple[float, float] = DEFAULT_MSPHERE_YLIM,
    cmap: Any = sns.color_palette("mako", as_cmap=True),
    cbar_label: str = 'Current Density ($nA/m^2$)',
) -> Tuple[plt.Figure, plt.Axes]:
    """Produces a series of plots visualizing the current indensity.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      empty_theta_region_threshold: Display points with theta < this number of
        degrees as empty, to avoid displaying points with grid artifacts. May
        be None.
    Returns
      ax: matplotlib axes generated
    """    
    mesh_curlB = mesh.compute_derivative('B', gradient=False, vorticity='curlB')
    J = (
        mesh_curlB['curlB'] 
        * (units.G/constants.R_earth) / constants.mu0  # type:ignore
    )
    mesh_curlB['J'] = J.to(units.nA / units.m**2).value  # type: ignore
    mesh_curlB['Jy'] = mesh_curlB['J'][:, 1]

    if empty_theta_region_threshold is not None:
        theta_grid_deg = np.rad2deg(mesh['Theta_grid'])
        mask = (np.abs(theta_grid_deg) < empty_theta_region_threshold)
        mesh_curlB['J'][mask] = np.nan

    # Current Density Strength
    meridional_plot_of_intensity(
        mesh_curlB, model_title, arr_name='J', norm=LogNorm(1, 500),
        cbar_label=cbar_label, cmap=cmap,
    )
    plt.xlim(xlim)
    plt.ylim(ylim)

    return plt.gcf(), plt.gca()


def equitorial_plot_of_current(
    mesh: pyvista.StructuredGrid,
    model_title: str,
    xlim: Tuple[float, float] = DEFAULT_MSPHERE_XLIM,
    ylim: Tuple[float, float] =DEFAULT_MSPHERE_YLIM
) -> Tuple[plt.Figure, plt.Axes]:
    """Produces a series of plots visualizing the current indensity.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
    Returns
      ax: matplotlib axes generated
    """
    mesh_curlB = mesh.compute_derivative('B', gradient=False, vorticity='curlB')
    J = (
        mesh_curlB['curlB']
        * (units.G/constants.R_earth) / constants.mu0  # type: ignore
    )
    mesh_curlB['J'] = J.to(units.nA / units.m**2).value  # type: ignore
    mesh_curlB['Jy'] = mesh_curlB['J'][:, 1]

    equitorial_plot_of_intensity(
        mesh_curlB, model_title, arr_name='J',
        norm=Normalize(0, 15),
        cbar_label='Current Density Strength ($nA/m^2$)'
    )
    plt.xlim(xlim)
    plt.ylim(ylim)

    return plt.gcf(), plt.gca()


def add_field_line_traces_meridional_plot(
    ax: plt.Axes,
    mesh: pyvista.StructuredGrid,
    lshells: NDArray[np.float64] = np.arange(3, 6.5, .5),
    color: str  = 'black'
) -> None:
    """Helper function to add field line traces to a meridional plot.

    Args
      ax: Matplotlib axes, eg returned by meridional_plot_of_intensity() or
        meridional_plot_of_current()
      mesh: Mesh holding magnetic field to derive traces from
      lshells: sets magnetic latitudes of starting points of field line traces.
        magnetic latitude corresponds to field line with this lshell in dipole
      color: Color of lines
    """
    inner_rvalue = 1.05 * np.linalg.norm(mesh.points, axis=1).min()
    mlats = np.arccos(np.sqrt(inner_rvalue / lshells))

    for mlat in mlats:
        for flip in [-1, 1]:
            x = inner_rvalue * np.cos(mlat) * flip
            y = 0
            z = inner_rvalue * np.sin(mlat)

            try:
                res = invariants.calculate_K(mesh, (x, y, z), 7.5)
            except invariants.FieldLineTraceInsufficient:
                continue

            # Skip broken traces: these tend to jump around too far
            ds_vec = np.diff(res.trace_points, axis=0)
            ds_scalar = np.linalg.norm(ds_vec, axis=1)

            if not np.any(ds_scalar > 0.1):
                ax.plot(res.trace_points[:, 0], res.trace_points[:, 2], color=color)


def add_field_isolines_to_equitorial_plot(
    ax: plt.Axes, mesh: pyvista.StructuredGrid, levels: int = 20
) -> None:
    """Helper function to field isolines to a equitorial plot.

    Isolines are plot at Bmax / level_num**3 to account for cubically-
    decreasing field strength.

    Args
      ax: Matplotlib axes, eg returned by equitorial_plot_of_intensity() or
        equitorial_plot_of_current()
      mesh: Mesh holding magnetic field to derive isolines from
      levels: number of levels
    """
    field = np.linalg.norm(mesh['B'], axis=1)
    Xeq = utils.lfm_get_eq_slice(mesh.x)
    Yeq = utils.lfm_get_eq_slice(mesh.y)
    s = mesh.x.shape
    F = np.reshape(field.ravel(), s, order='F')   # field
    Feq = utils.lfm_get_eq_slice(F)
    levels_list = sorted( * EARTH_DIPOLE_B0 / np.arange(1, levels + 1)**3.0)

    ax.contour(Xeq, Yeq, Feq, levels=levels_list, colors='black')


def add_field_isolines_to_meridional_plot(
    ax: plt.Axes, mesh: pyvista.StructuredGrid, levels: int = 20
) -> None:
    """Helper function to field isolines to a meridional plot.

    Isolines are plot at Bmax / level_num**3 to account for cubically-
    decreasing field strength.

    Args
      ax: Matplotlib axes, eg returned by meridional_plot_of_intensity() or
        meridional_plot_of_current()
      mesh: Mesh holding magnetic field to derive isolines from
      levels: number of levels
    """
    field = np.linalg.norm(mesh['B'], axis=1)
    Xeq = utils.lfm_get_mer_slice(mesh.x)
    Zeq = utils.lfm_get_mer_slice(mesh.z)
    s = mesh.x.shape
    F = np.reshape(field.ravel(), s, order='F')   # field
    Feq = utils.lfm_get_mer_slice(F)
    levels_list = sorted( * EARTH_DIPOLE_B0 / np.arange(1, levels + 1)**3.0)

    ax.contour(Xeq, Zeq, Feq, levels=levels_list, colors='black')
