"""This module provides tools for loading In-Situ flux observations of the
from satellite missions. Data loaded through this module is then used with
the phase space density module, :py:mod:`~dasilva_invariants.psd`.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

from astropy import units
from astropy.constants import R_earth
import cdflib
import numpy as np
from numpy.typing import NDArray
import PyGeopack as gp

__all__ = ["InSituObservation", "get_rbsp_electron_level3"]


@dataclass
class InSituObservation:
    """Collection of variables describing a 3-dimensional in-situ flux measurement at
    a single timestep.

    Parameters
    ----------
    time : datetime
       time of observation, without timezone
    flux : NDArray[np.float64]
       unidirectional flux measurement, in units of units of 1/(cm^2 sec sr keV)
    energies : NDArray[np.float64]
       eneriges associated with flux measurement, in units of eV
    pitch_angles : NDArray[np.float64]
       pitch angles associated with flux measurement, in units of degrees
    sc_position : Tuple[float, float, float]
       Spacecraft position, in SM coordate system and units of Re
    """

    time: datetime
    flux: NDArray[np.float64]
    energies: NDArray[np.float64]
    pitch_angles: NDArray[np.float64]
    sc_position: Tuple[float, float, float]


def get_rbsp_electron_level3(hdf_path) -> List[InSituObservation]:
    """Loads InSituObservation instances from a Boyd et al Level 3 pitch angle
    resolved Radiation Belt Storm Probe (RBSP) dataset file. This data can be
    downloaded from `RBSP ECT Data Products <https://rbsp-ect.newmexicoconsortium.org/science/DataDirectories.php>`_ .

    Parameters
    ----------
    hdf_path : str
        Path to HDF5 RBSP Level 3 pitch angle resolved data file

    Returns
    -------
    insitu_datasets : List[:py:class:`~InSituObservation`]
        Observations loaded from disk and organized for further processing
    """
    # Load variables ---------------------------------------------------------
    cdf = cdflib.CDF(hdf_path)
    times = np.array(
        [
            datetime(1, 1, 1) + timedelta(days=-1, milliseconds=dt)
            for dt in cdf.varget("Epoch")
        ]
    )
    energies = cdf.varget("FEDU_Energy") * 1000  # keV -> eV
    flux = cdf.varget("FEDU")
    pitch_angles = cdf.varget("FEDU_Alpha")
    sc_positions_geo = (cdf.varget("Position") * units.km).to(R_earth).value

    # Convert spacecraft positions to sm coordinate system
    x_geo, y_geo, z_geo = sc_positions_geo.T
    dates = [int(time.strftime("%Y%m%d")) for time in times]
    uts = [int(time.strftime("%H")) + time.minute / 60 for time in times]

    x_sm, y_sm, z_sm = gp.Coords.ConvCoords(
        x_geo, y_geo, z_geo, dates, uts, CoordIn="GEO", CoordOut="SM"
    )

    sc_positions_sm = np.array([x_sm, y_sm, z_sm]).T

    # variables present in every object, will be copied for each timestep
    base_contents = {}
    base_contents["energies"] = energies
    base_contents["pitch_angles"] = pitch_angles

    # loop through timestep
    insitu_observations = []

    for i in range(times.size):
        contents = base_contents.copy()
        contents["time"] = times[i]
        contents["flux"] = flux[i]
        contents["sc_position"] = tuple(sc_positions_sm[i, :].tolist())

        insitu_observations.append(InSituObservation(**contents))

    # Return instance of InSituObservation
    return insitu_observations
