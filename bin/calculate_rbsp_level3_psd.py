#!/usr/bin/env python
"""Calculate Phase Space Density as a function of L*, using RBSP data and
magnetic field models. 
"""
import sys, os
code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.append(code_above)

import argparse
import glob
from typing import Any, Callable, Dict, List, Tuple

import dask
from dask.distributed import Client, progress
from dask_jobqueue import PBSCluster
import dateutil.parser
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from dasilva_invariants.insitu import (
    InSituObservation,
    get_rbsp_electron_level3
)
from dasilva_invariants import invariants, meshes, psd


def _get_tasks(
    insitu_observations: List[InSituObservation],
    data_dir: str,
    run_name: str,
    particle: str,
    mu: float,
    K: float      
) -> List[Callable]:
    """Get all parallel tasks to process each timestep to find PSD vs L*
    curve.

    Args
      insitu_observation: Observational data loads from the module
        dasilva_invariants.insitu
      data_dir: Root directory to all runs
      run_name: Name of run directory under data_dir
      particle: Set the particle type, either 'electron' or 'proton'
      mu: Fixed first adiabatic invariant, units of MeV/G
      K: Fixed second adiabatic invariant, units of sqrt(G) Re
    Returns
      list of parallel processing tasks (dasked delayed function calls)
    """
    tasks = []
    mesh_times, mesh_paths = _get_mesh_files(data_dir, run_name)
    max_diff = 1.5 * (mesh_times[1] - mesh_times[0]).total_seconds()
    
    for insitu_observation in insitu_observations:  
        # Search for mesh that is close in time to observational data
        target_time = insitu_observation.time
        mesh_idx = np.argmin(np.abs(mesh_times - target_time))
        current_diff = (mesh_times[mesh_idx] - target_time).total_seconds()
        
        if current_diff > max_diff:
            continue

        # Create task using dask delayed interface
        tasks.append(dask.delayed(_do_task)(
            mu, K, insitu_observation, mesh_paths[mesh_idx], particle
        ))

    return tasks


def _do_task(
    mu: float, K: float, insitu_observation: InSituObservation, mesh_path: str,
    particle: str
) -> Dict[str, Any]:
    """Execute one parallel task

    Args
      mu: Fixed first adiabatic invariant, units of MeV/G
      K: Fixed second adiabatic invariant, units of sqrt(G) Re    
      insitu_observation: Observational data loads from the module
        dasilva_invariants.insitu
      mesh_path: Path to load mesh from disk
      particle: Set the particle type, either 'electron' or 'proton'
    Returns
      dictionary of row, which maps column name (string) to value
    """
    mesh = meshes.get_lfm_hdf4_data(mesh_path)
    
    row: Dict[str, Any] = {}
    row['time'] = insitu_observation.time.isoformat()
    row['magnetic_field_file'] = os.path.basename(mesh_path)
    row['sc_position_x_sm'] = insitu_observation.sc_position[0]
    row['sc_position_y_sm'] = insitu_observation.sc_position[1]
    row['sc_position_z_sm'] = insitu_observation.sc_position[2]
    
    try:
        result = psd.calculate_LStar_profile(
            fixed_mu=mu, fixed_K=K, insitu_observation=insitu_observation, mesh=mesh,
            particle=particle, calculate_lstar_kwargs={'num_local_times': 20}
        )
        row['psd'], row['LStar'] = result.phase_space_density, result.LStar
    except invariants.DriftShellSearchDoesntConverge as e:
        row['psd'], row['LStar'] = -1.0, -1.0
    except invariants.FieldLineTraceInsufficient as e:
        row['psd'], row['LStar'] = -2.0, -2.0

    return row


def _get_mesh_files(
    data_dir: str, run_name: str
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Search list of mesh files, and parse associated times.

    Args
      data_dir: Root directory to all runs
      run_name: Name of run directory under data_dir
    Returns
      mesh_times: array of datetimes
      mesh_paths: array of string file paths
    """
    hdf_dir = f'{data_dir}/{run_name}/*mhd*.hdf'
    hdf_paths = glob.glob(hdf_dir)
    hdf_paths.sort()

    mesh_times = np.zeros(len(hdf_paths), dtype=object)
    mesh_paths = np.zeros(len(hdf_paths), dtype=object)    
    
    for i, hdf_path in enumerate(hdf_paths):
        file_name = os.path.basename(hdf_path)
        file_name, file_suff = hdf_path.split('.')
        date_str = file_name.split('_')[-1]
        date_str = date_str[::-1].replace('-', ':', 2)[::-1]
        mesh_time = dateutil.parser.isoparse(date_str).replace(tzinfo=None)

        mesh_times[i] = mesh_time
        mesh_paths[i] = hdf_path

    return mesh_times, mesh_paths


def main() -> None:
    """Main function of the program."""
    # Parser command line arguments ------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('rbsp_level3_file', help='Path to RBSP L3 data file')
    parser.add_argument('run_name', help='Name of run')
    parser.add_argument('output_file', help='CSV file to write')
    parser.add_argument('n_jobs', type=int, help='Number of parallel jobs')
    parser.add_argument('--mu', required=True, help='Units: MeV/G')
    parser.add_argument('--K', required=True, help='Units: sqrt(G) Re')
    parser.add_argument('--particle', default='electron',
                        choices=['electron', 'proton'])
    parser.add_argument('--data-dir', default='/glade/scratch/danieldas/data')
    parser.add_argument('-p', '--pbs-project-id', type=str)

    args = parser.parse_args()

    # Load RBSP data
    # ------------------------------------------------------------------------
    if args.particle == 'electron':
        print('Loading RBSP electron data')
        insitu_observations = get_rbsp_electron_level3(args.rbsp_level3_file)
    else:
        raise NotImplementedError(
            'Only electron data implemented at this time'
        )

    # Create list of task
    # ------------------------------------------------------------------------
    tasks = _get_tasks(
        insitu_observations, args.data_dir, args.run_name, args.particle,
        args.mu, args.K
    )

    # Process tasks in parallel
    # ------------------------------------------------------------------------
    print(f'Total number of tasks: {len(tasks)}')
    df_contents: Dict[int, Dict[str, float]] = {}   # keys are num local time
    
    if args.pbs_project_id:
        print('Setting up PBS cluster')
        cluster = PBSCluster(
            cores=50, processes=50, memory='150 GB', queue='regular',
            walltime='04:00:00',
            project=args.pbs_project_id)
        cluster.scale(jobs=args.n_jobs)
        client = Client(cluster)
        print('Dashboard', client.dashboard_link)
        tasks = [task.persist() for task in tasks]  # type: ignore
        progress(tasks)
        task_results = [task.compute() for task in tasks]  # type: ignore
    else:
        client = Client(n_workers=args.n_jobs)
        task_results = dask.compute(tasks)

    # Collect results in dataframe and print to console and disk
    # ------------------------------------------------------------------------
    df = pd.DataFrame(task_results)
    df.to_csv(args.output_file, index=0)

    print(df.to_string(index=0))
    print()
    print(f'Wrote to {args.output_file}')
    

if __name__ == '__main__':
    main()
