0  #!/bin/env python
"""Process a traj file of particle trajectories for Mary Hudson's project."""

import sys, os

code_above = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
sys.path.append(code_above)

import argparse
from dataclasses import dataclass, asdict
import glob
import itertools

import fortranformat as ff
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from dasilva_invariants import invariants, meshes


def main():
    """Main routine of the program."""
    # Parsse command line arguments ------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("traj_csv_path")
    parser.add_argument("n_jobs", type=int)
    parser.add_argument("--particle-num-start", type=int, required=True)
    parser.add_argument("--particle-num-stop", type=int, required=True)
    parser.add_argument("--data_dir", default="/glade/scratch/danieldas/data")

    args = parser.parse_args()

    # Lookup list of model output (one file per timestep) --------------------
    hdf_dir = f"{args.data_dir}/{args.run_name}/*mhd*.hdf"
    hdf_paths = glob.glob(hdf_dir)
    hdf_paths.sort()

    # Create parallel processing pool ----------------------------------------
    par = Parallel(n_jobs=args.n_jobs, backend="multiprocessing", verbose=5000)

    # Loop through file of and process in parallel ------------------------
    fh = open(args.traj_csv_path)
    tasks = []

    for line in fh:
        row = TrajFileRow.parse_line(line)

        if row.particle_num < args.particle_num_start:
            continue
        if row.particle_num > args.particle_num_stop:
            break
        if int(row.time) % (60 * 5) != 0:
            continue

        hdf_path = hdf_paths[int(row.time)]
        row_task = delayed(process_row)(row, hdf_path)
        tasks.append(row_task)

    fh.close()
    results = par(tasks)

    # Write CSV output to terminal and disk ----------------------------------
    # output_path = f'daSilva_P{args.particle_num:06d}_'
    output_path = (
        f"daSilva_P{args.particle_num_start:06d}-" f"{args.particle_num_stop:06d}_"
    )
    output_path += os.path.basename(args.traj_csv_path)
    fh = open(output_path, "w")
    df = pd.DataFrame(results).fillna(0)

    print(df.to_string(index=0))

    writer = ff.FortranRecordWriter(
        "(i8,f10.2,f9.5,f9.1,f9.5,f12.7,f6.2,f8.2,f9.5,e14.6)"
    )

    for _, row in df.iterrows():
        line = writer.write(row.values.tolist())
        fh.write(line)
        fh.write("\n")

    fh.close()

    print(f"Wrote to {output_path}")


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
    # output['radial_dist_old'] = row.radial_dist
    # output['lfm_file'] = os.path.basename(hdf_path)

    starting_point = (
        row.radial_dist * np.cos(np.deg2rad(row.az_loc)),  # X
        row.radial_dist * np.sin(np.deg2rad(row.az_loc)),  # Y
        0,  # Z
    )
    mirror_lat = 0

    try:
        result = invariants.calculate_LStar(
            mesh,
            starting_point,
            mirror_lat,
            num_local_times=10,
            verbose=False,
            n_jobs=1,
        )
    except invariants.FieldLineTraceReturnedEmpty:
        result = None
    except invariants.DriftShellBisectionDoesntConverge:
        result = None

    if result is None:
        output["radial_dist"] = np.nan
    else:
        output["radial_dist"] = result.LStar

    return output


@dataclass
class TrajFileRow:
    """Represents a row of the Traj file.

    Use the parse() class method to convert from a string line to this
    object.
    """

    particle_num: int
    time: float  # seconds (counts from 0)
    radial_dist: float  # Re
    az_loc: float  # deg from noon (azimuthal location)
    energy: float  # MeV
    drift_vel: float  # Re/s
    initial_radial_loc: float  # Re
    initial_az_loc: float  # deg
    initial_energy: float  # MeV
    first_invariant: float  # MeV/nT

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
            first_invariant=float(toks[9]),
        )


if __name__ == "__main__":
    main()
