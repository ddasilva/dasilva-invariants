#!/bin/env python

"""Run the L* vs number of local times experiment. This is meant to determine
the minimum number of local times needed in the drift shell calculation in 
order to generate a consistent L*. 

See also: LStar vs Number of Local Times Experiment.ipynb (plotting)
"""
import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
import dask
from dask.distributed import Client, progress
from dask_jobqueue import PBSCluster
from datetime import datetime
from typing import cast, Any, Callable, Dict, List, Tuple

import pandas as pd
import pyvista

from dasilva_invariants import meshes, invariants


def _get_tasks(
    mesh_name: str, mesh_file_name: str, pitch_angle: float,
    max_local_times: int, starting_distance: float
) -> List[Callable]:
    """Get all parallel tasks to experiment with a given mesh on disk for a
    given pitch angle and starting position at varyying number of local times

    Args
      mesh_name: unique identified name of the mesh
      mesh_file_name: path to the mesh on disk
      pitch_angle: pitch angle for the calculation
      max_local_times: upper limit on number of local times (inclusive)
      starting_distance: starting distance (may be negative for into tail)
    Returns
      list of parallel processing tasks (dasked delayed function calls)
    """
    tasks = []
    
    for _, num_local_times in enumerate(range(2, max_local_times + 1)):
        key = (mesh_name, num_local_times, pitch_angle)
        
        tasks.append(dask.delayed(_parallel_target)(
            key, mesh_file_name, (starting_distance, 0, 0),
            starting_pitch_angle=pitch_angle,
            num_local_times=num_local_times, verbose=0,
        ))

    return tasks


def _parallel_target(
    key: str, mesh_fname: str, *args, **kwargs
) -> Tuple[str, float]:
    """Parallel processing target. Returns the key and processed Lstar.

    Catches known exceptions. Returns

    -1 when drift shell is not closed
    -2 for when drift shell search doesn't convert
    -3 when the field line trace is insuficicent.

    Args
      key: key returned alone with result
      mesh_fname: path to loading mesh from disk (may be HDF4 for LFM or .vtk)
        for tsyganenko 
      *args, **kwargs: passed to calculate_LStar()    
    Returns
      key, lstar: tuple of the provided key and the resultant lstar found

    """
    if mesh_fname.endswith('.vtk'):
        mesh = pyvista.read(mesh_fname)
    else:
        mesh = meshes.get_lfm_hdf4_data(mesh_fname)

    try:
        result = invariants.calculate_LStar(mesh, *args, **kwargs)
    except invariants.DriftShellSearchDoesntConverge as e:
        return key, -1.0
    except invariants.FieldLineTraceInsufficient as e:
        return key, -2.0

    return key, result.LStar


def _generate_tsyganenko_fields(
    time: datetime, t96_file_name: str, ts05_file_name: str,
    lfm_file_name: str, params_path: str
) -> None:
    """Generate Tsyganenko fields (T95 and TS05) for a given time and save
    to disk.

    Args
      time: datetime instance (UTC, no time zone) for when to generate for    
      t96_file_name: path to save T96 field
      ts05_file_name: path to save TS05 field
      lfm_file_name: path to any LFM HDF4 (for obtaining grid)
      params_path: path to OMNI zip file to feed model
    """
    if not os.path.exists(t96_file_name):    
        t96_mesh = meshes.get_tsyganenko_on_lfm_grid_with_auto_params(
            'T96', time, lfm_file_name, params_path, tell_params=True,
        )
        t96_mesh.save(t96_file_name)
        
    if not os.path.exists(ts05_file_name):
        ts05_mesh = meshes.get_tsyganenko_on_lfm_grid_with_auto_params(
            'TS05', time, lfm_file_name, params_path, tell_params=True,
        )
        ts05_mesh.save(ts05_file_name)


def _get_run_configuration(
    args: argparse.Namespace
)-> Tuple[Dict[str, str], datetime, int, List[float]]:
    """"Get configuration for the run such as meshes, max number of local times,
    and pitch angles

    Args
      args: command line arguments
    Returns
      mesh_file_names: dictionaryy mapping mesh names to file paths
      tsyganenko_time: datetime instance (no timezone) for generating
        tysganenko models
      max_local_times: upper limit of number of local times in experiment
      pitch_angles: list of pitch angles
    """
    if args.file_set == 'quiet':
        mesh_file_names = {
            'LFM': os.path.join(
                args.data_dir,
                'LFM-20131002_RBSP/ElkStorm_mhd_2013-10-04T00-00-00Z.hdf'
            ),
            'LFM-RCM': os.path.join(
                args.data_dir,
                'LFMRCM-20131002_RBSP/ElkStorm-LR_mhd_2013-10-04T00-00-00Z.hdf'
            ),
            'T96': os.path.join(os.getcwd(), 'quiet_t96.vtk'),
            'TS05': os.path.join(os.getcwd(), 'quiet_ts05.vtk')
        }

        tsyganenko_time = datetime(2013, 10, 4, 0, 0)
        max_local_times = 15

    elif args.file_set == 'disturbed':
        mesh_file_names = {
            'LFM': os.path.join(
                args.data_dir,
                'LFM-20131002_RBSP/ElkStorm_mhd_2013-10-02T06-19-00Z.hdf'
            ),
            'LFM-RCM': os.path.join(
                args.data_dir,
                'LFMRCM-20131002_RBSP/ElkStorm-LR_mhd_2013-10-02T06-19-00Z.hdf'
            ),
            'T96': os.path.join(os.getcwd(), 'disturbed_t96.vtk'),
            'TS05': os.path.join(os.getcwd(), 'disturbed_ts05.vtk')
        }

        tsyganenko_time = datetime(2013, 10, 2, 6, 19)
        max_local_times = 40
    else:
        raise RuntimeError(f'Invalid file_set {args.file_set}')

    pitch_angles = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    
    return (mesh_file_names, tsyganenko_time, max_local_times, pitch_angles)


def main() -> None:
    """Main function of the program."""
    # Parser command line arguments ------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('file_set', choices=['quiet', 'disturbed'])
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('-d', '--starting-distance', type=float, default=-8.0)
    parser.add_argument('--data-dir', default='/glade/scratch/danieldas/data')
    parser.add_argument('-p', '--pbs-project-id', type=str)
    parser.add_argument(
        '--params-path',
        default='/glade/u/home/danieldas/scratch/data/WGhour-latest.d.zip'
    )

    args = parser.parse_args()
    
    # Get config for this file set -------------------------------------------
    mesh_file_names, tsyganenko_time, max_local_times, pitch_angles = \
        _get_run_configuration(args)

    # Generate tsyganenko fields and save to path specified ------------------
    _generate_tsyganenko_fields(
        tsyganenko_time, mesh_file_names['T96'], mesh_file_names['TS05'],
        mesh_file_names['LFM'], args.params_path
    )

    # Collect list of tasks to be processed in parallel ----------------------
    tasks = []
    
    for mesh_name, mesh_file_name in mesh_file_names.items():
        for pitch_angle in pitch_angles:
            cur_tasks = _get_tasks(
                mesh_name, mesh_file_name, pitch_angle, max_local_times,
                args.starting_distance
            )
            tasks.extend(cur_tasks)

    # Process tasks, and organize results -----------------------------------    
    print(f'Total number of tasks: {len(tasks)}')
    df_contents: Dict[int, Dict[str, float]] = {}   # keys are num local time
    
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
        task_results = [task.compute() for task in tasks]
    else:
        client = Client(n_workers=args.n_jobs)
        task_results = dask.compute(tasks)

    for (mesh_name, num_local_times, pitch_angle), lstar in task_results:
        if num_local_times not in df_contents:
            df_contents[num_local_times] = {}

        column_name = '%s_%d' % (mesh_name, pitch_angle)            
        df_contents[num_local_times][column_name] = lstar

    # Organize into dataframe -----------------------------------------------
    df_columns = ['num_local_times'] + sorted(df_contents[2])
    df_rows = []

    for num_local_times in sorted(df_contents):
        row: List[Any] = [num_local_times]
        
        for column_name in df_columns[1:]:
            lstar = df_contents[num_local_times][column_name]
            row.append(lstar)

        df_rows.append(row)

    df = pd.DataFrame(df_rows, columns=df_columns)

    print(df.to_string(index=False))

    output_file_name = (
        f'local_times_experiment_{args.file_set}_r{args.starting_distance}.csv'
    )
    
    if os.path.exists(output_file_name):
        df_exist = pd.read_csv(output_file_name)
        df_exist = cast(pd.DataFrame, df_exist)
        
        for col in df.columns:
            df_exist[col] = df[col]

        df_exist.to_csv(output_file_name, index=False)        
    else:
        df.to_csv(output_file_name, index=False)

    print(f'Wrote to {output_file_name}')


if __name__ == '__main__':
    main()
