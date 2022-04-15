#!/bin/env python
"""Calculate adiabatic parameters throught storm from LFM otuput."""

import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
import glob
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dasilva_invariants import invariants, meshes


def process_hdf_path(hdf_path, radii, mirror_lats):
    """Process one HDF file, corresponding to a single timestep of model
    output.

    Args
       hdf_path: string path to HDF file
       radii: list of float starting radii to process (on the nightside)
       mirror_lats: list of mirror latitudes, in degrees to process 

    Returns
      dictionary holding key/value pairs for this row of the CSV file
    """
    mesh = meshes.get_lfm_hdf4_data(hdf_path)
    output = {
        'filename': os.path.basename(hdf_path)
    }
    
    for radius in radii:
        for mirror_lat in mirror_lats:
            starting_point = (-radius, 0, 0)

            # Calculate K adiabtic invariant --------------------------------
            key = 'K_radius=%.2f:mirror_lat=%.2f' % (radius, mirror_lat)

            try:
                result = invariants.calculate_K(
                    mesh, starting_point, mirror_lat
                )
            except invariants.FieldLineTraceReturnedEmpty:
                result = None

            if result is None:
                output[key] = np.nan
            else:
                output[key] = result.K


            # Calcualte Lstar adiabatic invariant ----------------------------
            key = 'LStar_radius=%.2f:mirror_lat=%.2f' % (radius, mirror_lat)

            try:
                result = invariants.calculate_LStar(
                    mesh, starting_point, mirror_lat,
                    num_local_times=10, verbose=False, n_jobs=1,
                )
            except invariants.FieldLineTraceReturnedEmpty:
                result = None
            except invariants.DriftShellBisectionDoesntConverge:
                result = None
            
            if result is None:
                output[key] = np.nan
            else:
                output[key] = result.LStar
                
    return output


def main():
    """Main function of the program."""
    # Parser command line arguments ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--data_dir', default='/glade/scratch/danieldas/data')
    parser.add_argument('--radii', type=str, default='4,6,8')
    parser.add_argument('--mirror-lat', type=str, default='15,30')
        
    args = parser.parse_args()

    radii = [float(val) for val in args.radii.split(',')]
    mirror_lats = [float(val) for val in args.mirror_lat.split(',')]
    
    # Lookup list of model output (one file per timestep) -------------------
    hdf_dir = f'{args.data_dir}/{args.run_name}/*mhd*.hdf'
    hdf_paths = glob.glob(hdf_dir)
    hdf_paths.sort()
    
    # Create joblib tasks and process in parallel ---------------------------
    tasks = []
    
    for hdf_path in hdf_paths:
        tasks.append(delayed(process_hdf_path)(hdf_path, radii, mirror_lats))

    par = Parallel(n_jobs=args.n_jobs, backend='multiprocessing', verbose=50000)
    outputs = par(tasks[::10])

    # Write output -----------------------------------------------------------
    output_file = args.run_name + '_invariants.csv'
    df = pd.DataFrame(outputs)
    df.to_csv(output_file)

    print(f'Wrote to {output_file}')

    
if __name__ == '__main__':
    main()
