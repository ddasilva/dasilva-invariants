0#!/bin/env python
"""Process a traj file of particle trajectories for Mary Hudson's project."""

import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
from dataclasses import dataclass, asdict
import glob
import itertools

import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from dasilva_invariants import invariants, meshes


def main():
    """Main routine of the program."""
    # Parsse command line arguments ------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('traj_csv_path')
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--particle-num', type=int, required=True)
    parser.add_argument('--data_dir', default='/glade/scratch/danieldas/data')
    
    args = parser.parse_args()

    # Lookup list of model output (one file per timestep) --------------------
    hdf_dir = f'{args.data_dir}/{args.run_name}/*mhd*.hdf'
    hdf_paths = glob.glob(hdf_dir)
    hdf_paths.sort()

    # Create parallel processing pool ----------------------------------------
    par = Parallel(n_jobs=args.n_jobs, backend='multiprocessing', verbose=5000)

    # Loop through batches of and process in parallel ------------------------
    batch_size = args.n_jobs    
    fh = open(args.traj_csv_path)
    results = []

    processed_count = 0
    
    for lines in iter_batch(fh, batch_size):
        batch_tasks = []
        early_exit = False
        
        for line in lines:
            row = TrajFileRow.parse_line(line)

            if row.particle_num < args.particle_num:
                continue
            if row.particle_num > args.particle_num:
                early_exit = True
                continue

            hdf_path = hdf_paths[int(row.time)]
            row_task = delayed(process_row)(row, hdf_path)
            batch_tasks.append(row_task)

        if not batch_tasks:
            continue
        
        batch_results = par(batch_tasks)
        results.extend(batch_results)

        processed_count += len(batch_tasks)
        print(f'==> Running total: processed {processed_count}')

        if early_exit:
            break
        
    fh.close()

    # Write CSV output to terminal and disk ----------------------------------
    output_path = f'daSilva_P{args.particle_num}_'
    output_path += os.path.basename(args.traj_csv_path)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=0))
    df.to_csv(output_path, sep=' ', na_rep='NaN', header=False, index=False)

    print(f'Wrote to {output_path}')
    

def process_row(row, hdf_path):
    """Process one row of the Traj file. 

    This target is meant to be run in parallel.
    
    Args
      row: instance of TrajFileRow
      hdf_path: Path to accompanying LFM output file
    Returns
      output: dictionary containing the contents of the row but also 
       new keys for hdf_path and LStar    
    """
    mesh = meshes.get_lfm_hdf4_data(hdf_path)
    output = asdict(row)
    output['radial_dist_old'] = row.radial_dist
    output['lfm_file'] = os.path.basename(hdf_path)

    starting_point = (
        row.radial_dist * np.cos(np.deg2rad(row.az_loc)),  # X
        row.radial_dist * np.sin(np.deg2rad(row.az_loc)),  # Y
        0                                                  # Z
    )
    mirror_lat = 0
    
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
        output['radial_dist'] = np.nan
    else:
        output['radial_dist'] = result.LStar

    return output
            

@dataclass
class TrajFileRow:
    """Represents a row of the Traj file.

    Use the parse() class method to convert from a string line to this 
    object.
    """
    particle_num: int
    time: float                # seconds (counts from 0)
    radial_dist: float         # Re
    az_loc: float              # deg from noon (azimuthal location)
    energy: float              # MeV
    drift_vel: float           # Re/s
    initial_radial_loc: float  # Re
    initial_az_loc: float      # deg
    initial_energy: float      # MeV
    first_invariant: float     # MeV/nT
    
    @classmethod
    def parse_line(cls, line):
        """Convert a string line to a TrajFileRow object.

        Args
          line: string line from the file
        Returns
          instance of TrajFileRow
        """
        toks = line.strip().split()
        
        return cls(
            particle_num=int(toks[0]),
            time=float(toks[1]),
            radial_dist=float(toks[2]),
            az_loc=float(toks[3]),
            energy=float(toks[4]),
            drift_vel=float(toks[5]),
            initial_radial_loc=float(toks[6]),
            initial_az_loc=float(toks[7]),
            initial_energy=float(toks[8]),
            first_invariant=float(toks[9])
        )

    
def iter_batch(iterable, batch_size):
    """Iterate through batches of iterable.

    Taken from https://stackoverflow.com/a/62913856

    Args
      iterable: some iterable object
      batch_size: integer batch size (n)
    Yields
      batches of size <= n
    """
    iterator = iter(iterable)

    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch



if __name__ == '__main__':
    main()
