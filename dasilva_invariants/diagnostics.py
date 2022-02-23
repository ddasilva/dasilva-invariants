"""Diagnostic routines with visualizations for quick look analysis at
magnetic field models.

To run all of these, use the do_all() function.
"""
from . import invariants

from matplotlib.colors import LogNorm
import pylab as plt
import numpy as np


def do_all(mesh, model_title):
    """Perform all diagnostics.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot    
    """
    funcs = [
        tail_traces,
        tail_K_vs_radius,
        drift_shells,
        dayside_field_intensities,
        equitorial_plot_of_intensity,
        K_integrand_plot,
    ]

    for func in funcs:
        func(mesh, model_title)

    
def tail_traces(mesh, model_title, th=180, r_min=3, r_max=9):
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
        x, y = np.cos(np.deg2rad(th)) * r, np.sin(np.deg2rad(th)) * r
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


def tail_K_vs_radius(mesh, model_title, th=180):
    """Visualize K versus R as the radius is stepped out in the tail.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
      th: Rotation in degrees of the sun-earth line    
    """
    rs = np.linspace(3, 15, 100)
    Ks = np.zeros_like(rs)

    for i, r in enumerate(rs):
        x, y = np.cos(np.deg2rad(th)) * r, np.sin(np.deg2rad(th)) * r
        Ks[i] = invariants.calculate_K(mesh, (x, y, 0), 7.5).K
    
    plt.plot(rs, Ks, '.-')
    plt.title(f'{model_title}\nK vs Radius in Tail')
    plt.xlabel('Radius (Re)')
    plt.ylabel('K(r)')
    plt.ylim(0, .03)


def drift_shells(mesh, model_title, th=180, r_min=3, r_max=9):
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
        x, y = np.cos(np.deg2rad(th)) * r, np.sin(np.deg2rad(th)) * r
        
        result = invariants.calculate_LStar(mesh, (x, y, 0), 7.5,
                                            num_local_times=50,
                                            verbose=False)
        results.append((r, result))
        
    # Plot drift shells
    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap('viridis')
    
    for i, (r, result) in enumerate(results):
        x = np.cos(result.drift_local_times) * result.drift_rvalues
        y = np.sin(result.drift_local_times) * result.drift_rvalues
        
        x = x.tolist() + [x[0]]
        y = y.tolist() + [y[0]]
        
        plt.plot(x, y, label=f'R = {r:.1f} Re', color=cmap(i/len(results)))
        plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(color='#ccc', linestyle='dashed')
        plt.title(f'{model_title}\nDrift Shells between R={rs[0]} '
                  f'and R={rs[-1]}')
        plt.xlabel('X (Re)')
        plt.ylabel('Y (Re)')
        plt.xlim(r_max * 1.25, -r_max * 1.25)
        plt.ylim(-r_max * 1.25, r_max * 1.25)        
        
        circle = plt.Circle((0, 0), 1, color='b')
        plt.gca().add_patch(circle)
        plt.tight_layout()


def dayside_field_intensities(mesh, model_title, th=0, r_min=3, r_max=7):
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
        x, y = np.cos(np.deg2rad(th)) * r, np.sin(np.deg2rad(th)) * r
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


def equitorial_plot_of_intensity(mesh, model_title):
    """Plot equitorial plot of field indensity.
    
    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
    """
    def _get_eq_slice(data):
        # Adapted from PyLTR
        nk = data.shape[2]-1
        dusk = data[:, :, 0]
        dawn = data[:, :, nk//2]
        dawn = dawn[:, ::-1]
        eq = np.hstack((dusk, dawn[:, 1:]))
        eq_c = 0.25*(eq[:-1, :-1] + eq[:-1, 1:] + eq[1:, :-1] + eq[1:, 1:])
        eq_c = np.append(eq_c.transpose(), [eq_c[:, 0]], axis=0).transpose()
        return eq_c

    Xeq = _get_eq_slice(mesh.x)
    Yeq = _get_eq_slice(mesh.y)

    B = np.linalg.norm(mesh['B'], axis=1)
    s = mesh.x.shape
    B = np.reshape(B.ravel(), s, order='F')
    Beq = _get_eq_slice(B)

    plt.figure(figsize=(12, 7))
    plt.title(f'{model_title}\nEquitorial Slice')
    plt.pcolor(Xeq, Yeq, Beq, norm=LogNorm(vmin=1e-5, vmax=1e-1))
    plt.xlabel('X SM (Re)')
    plt.ylabel('Y SM (Re)')
    plt.colorbar()
    plt.xlim(20, -70)
    plt.ylim(-40, 40)
    

def meridional_plot_of_intensity(mesh, model_title):
    """Plot meridional plot of field intensity.
    
    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
    """
    def get_mer_slice(data):
        # Adapted from pyLTR
        nk = data.shape[2] - 1
        north = data[:, :, nk//4]
        south = data[:, :, 3*nk//4]
        south = south[:, ::-1] # reverse the j-index
        mer = np.hstack((north, south[:,1:]))
        mer_c = 0.25*(mer[:-1,:-1] + mer[:-1,1:] + mer[1:,:-1] + mer[1:,1:])
        mer_c = np.append(mer_c.transpose(),[mer_c[:,0]],axis=0).transpose()
        return mer_c

    Xmer = get_mer_slice(mesh.x)
    Zmer = get_mer_slice(mesh.z)

    B = np.linalg.norm(mesh['B'], axis=1)
    s = mesh.x.shape
    B = np.reshape(B.ravel(), s, order='F')
    Bmer = get_mer_slice(B)
    
    plt.figure(figsize=(12, 7))
    plt.pcolor(Xmer, Zmer, Bmer, norm=LogNorm(vmin=1e-5, vmax=1e-1))
    plt.title(f'{model_title}\nMeridional Slice')
    
    plt.xlabel('X SM (Re)')
    plt.ylabel('Z SM (Re)')
    plt.colorbar()
    plt.xlim(20, -70)
    plt.ylim(-40, 40)
    
    
def K_integrand_plot(mesh, model_title, r=7, th=180):
    """Plot K integrand versus integration axis.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
    """
    mirror_deg = 7.5
    x, y = np.cos(np.deg2rad(th)) * r, np.sin(np.deg2rad(th)) * r
    result = invariants.calculate_K(mesh, (x, y, 0), mirror_deg)

    plt.figure(figsize=(8, 4))
    plt.plot(result.integral_axis_latitude, result.integral_integrand, 'k.-')
    plt.fill_between(result.integral_axis_latitude,
                     result.integral_integrand.min(),
                     result.integral_integrand)
    plt.title(f'{model_title}\n'
              f'K = {result.K:.6f} Re Sqrt(G) = '
              r'$\int_{s_m}^{s_m}\sqrt{B_m - B(s)}ds$'
              f'\n'
              f'Bm = {result.Bm:.6f} G\n'
              f'Mirror at {mirror_deg:.1f} deg',
              fontsize=20)
    plt.xlabel('Latitude (deg)', fontsize=20)
    plt.ylabel(r'$\sqrt{B_m - B(s)}$', fontsize=20)


def LStar_integrand_plot(mesh, model_title, r=7, th=0, LStar_kwargs={}):
    """Plot LStar integrand versus integration axis.

    Args
      mesh: grid and magnetic field, loaded using meshes module
      model_title: Title of magnetic field model, used in title of plot
    """
    mirror_deg = 7.5
    x, y = np.cos(np.deg2rad(th)) * r, np.sin(np.deg2rad(th)) * r
    result = invariants.calculate_LStar(mesh, (x, y, 0), mirror_deg,
                                        **LStar_kwargs)

    plt.figure(figsize=(8, 4))
    plt.plot(result.integral_axis, result.integral_integrand, 'k.-')
    plt.ylim([.2, .4])
    plt.fill_between(result.integral_axis,
                     plt.ylim()[0],
                     result.integral_integrand)

    for delta_local_time in [0, np.pi, 2 * np.pi]:
        plt.axvline(result.integral_axis.min() + delta_local_time,
                    color='black', linestyle='dashed')
        
    plt.title(f'{model_title}\n'
              f'L* = {result.LStar:.6f} = '
              r'$2 \pi (R_{in}/R_E) / \int_{0}^{2\pi} sin^2(\theta) d\phi$'
              f'\n'
              f'Mirror at {mirror_deg:.1f} deg',
              fontsize=20)
    plt.xlabel('Local time (radians about sun-earth line)', fontsize=20)
    plt.ylabel(r'$sin^2(\theta)$', fontsize=20)
    
