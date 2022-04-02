#!/usr/bin/env python
"""Create per-timestep visualizations of LFM output and visualize.

Requires ffmpeg to visualize.
"""
import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
import glob
import shutil
import subprocess
import pylab as plt
from joblib import Parallel, delayed
from dasilva_invariants import diagnostics, meshes


def process_hdf_path(hdf_path, out_dir, base_title):
    """Process one HDF file, corresponding to a single timestep of model
    output.
    
    Args
       hdf_path: string path to HDF file
       out_dir: base path to output plots to (each type of plot type will have
         its own directory)
       base_title: First line of the title of each plot
    """
    title = base_title + '\n' + os.path.basename(hdf_path)
    file_short_name = os.path.basename(hdf_path).replace('.hdf', '').replace('.h5', '')
    
    mesh_lfm = meshes.get_lfm_hdf4_data(hdf_path)
    mesh_dip = meshes.get_dipole_mesh_on_lfm_grid(hdf_path)

    mesh = mesh_lfm.copy()
    mesh['B'] = mesh_lfm['B'] - mesh_dip['B']
    
    plot_eq_intensity(mesh, mesh_lfm, title, out_dir, file_short_name)
    plot_eq_current(mesh, mesh_lfm, title, out_dir, file_short_name)
    plot_mer_intensity(mesh, mesh_lfm, title, out_dir, file_short_name)
    plot_mer_current(mesh, mesh_lfm, title, out_dir, file_short_name)
    

def plot_eq_intensity(mesh, mesh_lfm, title, out_dir, file_short_name):
    """Generate equitorial plot of intensity and save to disk.
    
    Args
      mesh: PyVista mesh holding the dipole subtracted field
      mesh_lfm: PyVista mesh holding the non-subtracted LFM output
      title: plot title to pass to diagnostics module
      out_dir: base path to output plots to (each type of plot type will have
        its own directory)
      file_short_name: The base name of the HDF file without file extension   
    """
    ax = diagnostics.equitorial_plot_of_intensity(mesh, title)
    diagnostics.add_field_isolines_to_equitorial_plot(ax, mesh_lfm)
    
    out_img = os.path.join(
        out_dir, 'eq_intensity', file_short_name + '_eq_intensity.png'
    )

    fig = ax.get_figure()
    fig.savefig(out_img)
    plt.close(fig)

    
def plot_mer_intensity(mesh, mesh_lfm, title, out_dir, file_short_name):
    """Generate meridional plot of intensity
    
    Args
      mesh: PyVista mesh holding the dipole subtracted field
      mesh_lfm: PyVista mesh holding the non-subtracted LFM output
      title: plot title to pass to diagnostics module
      out_dir: base path to output plots to (each type of plot type will have
        its own directory)
      file_short_name: The base name of the HDF file without file extension   
    """
    ax = diagnostics.meridional_plot_of_intensity(mesh, title)
    diagnostics.add_field_line_traces_meridional_plot(ax, mesh_lfm)

    out_img = os.path.join(
        out_dir, 'mer_intensity', file_short_name + '_mer_intensity.png'
    )
    
    fig = ax.get_figure()
    fig.savefig(out_img)
    plt.close(fig)

    
def plot_eq_current(mesh, mesh_lfm, title, out_dir, file_short_name):
    """Generate equitorial plot of intensity
    
    Args
      mesh: PyVista mesh holding the dipole subtracted field
      mesh_lfm: PyVista mesh holding the non-subtracted LFM output
      title: plot title to pass to diagnostics module
      out_dir: base path to output plots to (each type of plot type will have
        its own directory)
      file_short_name: The base name of the HDF file without file extension   
    """
    ax = diagnostics.equitorial_plot_of_current(mesh, title)
    diagnostics.add_field_isolines_to_equitorial_plot(ax, mesh_lfm)

    out_img = os.path.join(
        out_dir, 'eq_current', file_short_name + '_eq_current.png'
    )
    
    fig = ax.get_figure()
    fig.savefig(out_img)
    plt.close(fig)

    
def plot_mer_current(mesh, mesh_lfm, title, out_dir, file_short_name):
    """Generate meridional plot of current.
    
    Args
      mesh: PyVista mesh holding the dipole subtracted field
      mesh_lfm: PyVista mesh holding the non-subtracted LFM output
      title: plot title to pass to diagnostics module
      out_dir: base path to output plots to (each type of plot type will have
        its own directory)
      file_short_name: The base name of the HDF file without file extension
    """
    ax = diagnostics.meridional_plot_of_current(mesh, title)
    diagnostics.add_field_line_traces_meridional_plot(ax, mesh_lfm)

    out_img = os.path.join(
        out_dir, 'mer_current', file_short_name + '_mer_current.png'
    )
    
    fig = ax.get_figure()
    fig.savefig(out_img)
    plt.close(fig)

    
def run_ffmpeg_animation(plot_dir, mp4_output):
    """Use FFMPEG to animate a series of PNG's found in the plot_dir argument,
    and write to the mp4_output file (templated by %d for FPS variation).
    
    Args
       plot_dir: Path to directory holding PNG files
       mp4_output: Path to MP4 file to write, templated by %d for FPS variation
    """
    mp4_output = os.path.join(os.getcwd(), mp4_output)
    
    renamed_dir = os.path.join(plot_dir, 'renamed')
    os.makedirs(renamed_dir, exist_ok=True)

    png_files = glob.glob(f'{plot_dir}/*.png')
    png_files.sort()
    
    for i, png_file in enumerate(png_files):
        os.symlink(png_file, os.path.join(renamed_dir, '%05d.png' % i))
    
    cur_dir = os.getcwd()
    os.chdir(renamed_dir)
    
    fps15_mp4_output = mp4_output % 15
    fps30_mp4_output = mp4_output % 30
    
    subprocess.check_call(
        f'ffmpeg -r 15 -f image2  -start_number 1 -i %05d.png -vframes 1000 -vcodec libx264 '
        f'-crf 25 -pix_fmt yuv420p {fps15_mp4_output}',
        shell=True
    )
    subprocess.check_call(
        f'ffmpeg -r 30 -f image2  -start_number 1 -i %05d.png -vframes 1000 -vcodec libx264 '
        f'-crf 25 -pix_fmt yuv420p {fps30_mp4_output}',
        shell=True
    )
    
    os.chdir(cur_dir)
    shutil.rmtree(renamed_dir)
    

def main():
    """Main function of the program."""
    # Parser comomand line arguments ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--data_dir', default='/glade/scratch/danieldas/data')
    
    args = parser.parse_args()
    
    # Ensure that ffmpeg is present -----------------------------------------
    if shutil.which('ffmpeg') is None:
        print('ffmpeg is required. If running on cheyenne/casper, run '
              '"module load ffmpeg', file=sys.stderr)
        sys.exit(1)
    
    # Create plot output directories ----------------------------------------     
    base_title = args.run_name
    out_dir = f'{args.data_dir}/{args.run_name}/anim'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/eq_intensity', exist_ok=True)
    os.makedirs(out_dir + '/eq_current', exist_ok=True)
    os.makedirs(out_dir + '/mer_intensity', exist_ok=True)
    os.makedirs(out_dir + '/mer_current', exist_ok=True)
    
    # Lookup list of model output (one file per timestep) -------------------
    hdf_dir = f'{args.data_dir}/{args.run_name}/*mhd*.hdf'
    hdf_paths = glob.glob(hdf_dir)
    hdf_paths.sort()

    # Create joblib tasks and process in parallel ---------------------------
    tasks = []
    
    for hdf_path in hdf_paths:
        tasks.append(delayed(process_hdf_path)(hdf_path, out_dir, base_title))

    par = Parallel(n_jobs=args.n_jobs, backend='multiprocessing', verbose=50000)
    par(tasks)

    # Animate output --------------------------------------------------------
    run_ffmpeg_animation(out_dir + '/eq_intensity', 'eq_intensity_FPS%d.mp4')
    run_ffmpeg_animation(out_dir + '/eq_current', 'eq_current_FPS%d.mp4')
    run_ffmpeg_animation(out_dir + '/mer_intensity', 'mer_intensity_FPS%d.mp4')
    run_ffmpeg_animation(out_dir + '/mer_current', 'mer_current_FPS%d.mp4')
    

if __name__ == '__main__':
    main()
