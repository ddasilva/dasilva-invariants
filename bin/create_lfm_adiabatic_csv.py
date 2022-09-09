#!/bin/env python
"""Calculate adiabatic parameters throught storm from LFM otuput."""

import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
import glob
import dask
from dask.distributed import Client, progress
from dask_jobqueue import PBSCluster
import numpy as np
import pandas as pd
from dasilva_invariants import invariants, meshes


def process_hdf_path(hdf_path, radius, pitch_angle):
    """Process one HDF file, corresponding to a single timestep of model
    output.

    Args
       hdf_path: string path to HDF file
       radius: starting radii to process (from dayside)
       pitch_angle: pitch angle to process
    Returns
      data for row, to be joined on filename
    """
    mesh = meshes.get_lfm_hdf4_data(hdf_path)
    output = {
        'filename': os.path.basename(hdf_path)
    }

    starting_point = (-radius, 0, 0)

    # Calculate K adiabtic invariant --------------------------------
    key = 'K_radius=%.2f:pitch_angle=%.2f' % (radius, pitch_angle)
    
    try:
        result = invariants.calculate_K(
            mesh, starting_point, pitch_angle=pitch_angle
        )
    except invariants.FieldLineTraceInsufficient:
        result = None
        
    if result is None:
        output[key] = np.nan
    else:
        output[key] = result.K
        
    # Calcualte Lstar adiabatic invariant ----------------------------
    key = 'LStar_radius=%.2f:pitch_angle=%.2f' % (radius, pitch_angle)
    
    try:
        result = invariants.calculate_LStar(
            mesh, starting_point, starting_pitch_angle=pitch_angle,
            num_local_times=15
        )
    except invariants.FieldLineTraceInsufficient:
        result = None
        output[key] = -1.0
    except invariants.DriftShellSearchDoesntConverge:
        result = None
        output[key] = -2.0        
        
    if result is not None:
        if result.drift_is_closed:            
            output[key] = result.LStar
        else:
            output[key] = -3.0

    return output


def main():
    """Main function of the program."""
    # Parser command line arguments ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--data_dir', default='/glade/scratch/danieldas/data')
    parser.add_argument('--radii', type=str, default='-4,-6,-8')
    parser.add_argument('--pitch-angle', type=str, default='30,60')
    parser.add_argument('-p', '--pbs-project-id', type=str)
    
    args = parser.parse_args()

    radii = [float(val) for val in args.radii.split(',')]
    pitch_angles = [float(val) for val in args.pitch_angle.split(',')]

    # Lookup list of model output (one file per timestep) -------------------
    hdf_dir = f'{args.data_dir}/{args.run_name}/*mhd*.hdf'
    hdf_paths = glob.glob(hdf_dir)
    hdf_paths.sort()

    # Create joblib tasks and process in parallel ---------------------------
    tasks = []

    for hdf_path in hdf_paths[::10]:
        for radius in radii:
            for pitch_angle in pitch_angles:                
                tasks.append(dask.delayed(process_hdf_path)(
                    hdf_path, radius, pitch_angle
                ))

    print(f'Total of {len(tasks)} tasks')
        
    if args.pbs_project_id:
        print('Setting up PBS cluster')
        cluster = PBSCluster(
            cores=50, processes=50, memory='100 GB', queue='regular',
            walltime='04:00:00',
            project=args.pbs_project_id)
        cluster.scale(jobs=args.n_jobs)
        client = Client(cluster)
        print('Dashboard', client.dashboard_link)
        tasks = [task.persist() for task in tasks]
        progress(tasks)
        outputs = [task.compute() for task in tasks]
    else:
        client = Client(n_workers=args.n_jobs)
        outputs = dask.compute(tasks)

    # Organize rows
    df_rows = {}
    
    for output in outputs:
        key = output['filename']
        if key not in df_rows:
            df_rows[key] = {}

        df_rows[key].update(output)

    # Write output -----------------------------------------------------------
    output_file = args.run_name + '_invariants.csv'
    df = pd.DataFrame(df_rows.values())
    df.to_csv(output_file, index=False)

    print(df.to_string(index=False))
    print()
    print(f'Wrote to {output_file}')


if __name__ == '__main__':
    main()
