#!/usr/bin/env python
"""Calculate Phase Space Density as a function of L*, using RBSP data and
magnetic field models. 
"""
import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
from datetime import datetime
import glob
from typing import Any, Callable, Dict, List, Tuple

import dateutil.parser
from joblib import Parallel, delayed
import pandas as pd
import pyvista

from dasilva_invariants import meshes


def _get_tasks(data_dir: str, lfm_run_name: str, output_run_name: str,
               model_name: str, omni_data_path: str) -> List[Callable]:
    """Get all parallel tasks to make a Tsyganenko model file at each timestep.

    Args
      data_dir: Root directory to all runs
      lfm_run_name: Name of LFM run to base times from
      output_run_name: Name of run to output Tysganenko files to
      model_name: Name of Tsyganenko model, either 'T96' or 'TS05'
      omni_data_path: Path to OMNI parameters
    Returns
      list of parallel processing tasks (delayed function calls)
    """
    tasks = []
    lfm_times, lfm_paths = _get_lfm_files(data_dir, lfm_run_name)
    
    for lfm_time, lfm_path in zip(lfm_times, lfm_paths):
        # Create task using delayed interface
        tasks.append(delayed(_do_task)(
            model_name, data_dir, output_run_name, lfm_time, lfm_path,
            omni_data_path,
        ))

    return tasks



def _get_lfm_files(
    data_dir: str, lfm_run_name: str) -> Tuple[List[datetime], List[str]]:
    """Search list of lfm files, and parse associated times.

    Args
      data_dir: Root directory to all runs
      lfm_run_name: Name of run directory under data_dir
    Returns
      lfm_times: array of datetimes
      lfm_paths: path to HDF5 files
    """
    lfm_dir = f'{data_dir}/{lfm_run_name}/*mhd*.hdf'
    lfm_paths = glob.glob(lfm_dir)
    lfm_paths.sort()

    lfm_times = []
    
    for lfm_path in lfm_paths:
        file_name = os.path.basename(lfm_path)
        file_name, file_suff = lfm_path.split('.')
        date_str = file_name.split('_')[-1]
        date_str = date_str[::-1].replace('-', ':', 2)[::-1]

        lfm_time = dateutil.parser.isoparse(date_str).replace(tzinfo=None)
        lfm_times.append(lfm_time)
        
    return lfm_times, lfm_paths


def _do_task(
    model_name: str, data_dir: str, output_run_name: str, lfm_time: datetime,
    lfm_path: str, omni_data_path: str, 
) -> Dict[str, Any]:
    """Generate a Tsyganenko file and write to disk.

    Args
      model_name: Name of Tsyganenko model, either 'T96' or 'TS05'
      data_dir: Root directory to all runs
      output_run_name: Name of run to output Tysganenko files to
      lfm_time: Time associated with LFM file
      omni_data_path: Path to OMNI parameters
    Returns
      row for display in summary table
    """
    # Decide output file name
    output_path = os.path.join(
        data_dir, output_run_name, f'{model_name}_{lfm_time.isoformat()}.vtk'
    )

    # Generate Tsyganenko model
    tsy_mesh = meshes.get_tsyganenko_on_lfm_grid_with_auto_params(
        model_name, lfm_time, lfm_path, omni_data_path, tell_params=False
    )
    tsy_mesh.save(output_path)

    # Return row for summary table
    row = {}
    row['output_file'] = os.path.basename(output_path)
    row['time'] = lfm_time.isoformat()
    
    return row

    
def main() -> None:
    """Main function of the program."""
    # Parser command line arguments ------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('lfm_run_name', help='Name of LFM run to base times from')
    parser.add_argument('output_run_name', help='Name of run to output to')
    parser.add_argument('--model', dest='model_name', required=True,
                        choices=['T96', 'TS05'],
                        help='Name of Tsyganenko model')
    parser.add_argument('--n-jobs', required=True, type=int,
                        help='Number of parallel jobs')
    parser.add_argument('--data-dir', default='/glade/scratch/danieldas/data')
    parser.add_argument(
        '--omni-data-path',
        default='/glade/u/home/danieldas/scratch/data/WGhour-latest.d.zip'
    )

    args = parser.parse_args()

    # Create list of task
    # ------------------------------------------------------------------------
    tasks = _get_tasks(
        args.data_dir, args.lfm_run_name, args.output_run_name,
        args.model_name, args.omni_data_path
    )

    # Process tasks in parallel
    # ------------------------------------------------------------------------
    print(f'Total number of tasks: {len(tasks)}')

    pool = Parallel(n_jobs=args.n_jobs, backend='multiprocessing', verbose=5000)
    task_results = pool(tasks)

    # Collect results in dataframe and print to console and disk
    # ------------------------------------------------------------------------
    df = pd.DataFrame(task_results)

    print(df.to_string(index=0))
    print()
    

if __name__ == '__main__':
    main()
