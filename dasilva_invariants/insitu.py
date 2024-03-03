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
from matplotlib.dates import date2num
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import PyGeopack as gp

from .utils import nanoTesla2Gauss


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
    Bobs: NDArray[np.float64]
       magnitude of measured magnetic field, in units of Gauss
    sc_position : Tuple[float, float, float]
       Spacecraft position, in SM coordate system and units of Re
    """

    time: datetime
    flux: NDArray[np.float64]
    energies: NDArray[np.float64]
    pitch_angles: NDArray[np.float64]
    Bmodel: NDArray[np.float64]
    Bobs: NDArray[np.float64]
    sc_position: Tuple[float, float, float]
    sc_position_geo: Tuple[float, float, float]    
    sc_position_gsm: Tuple[float, float, float]


def get_rbsp_electron_level3(ect_cdf_path, emphesis_cdf_path) -> List[InSituObservation]:
    """Loads InSituObservation instances from a Boyd et al Level 3 pitch angle
    resolved Radiation Belt Storm Probe (RBSP) dataset file. This data can be
    downloaded from `RBSP ECT Data Products <https://rbsp-ect.newmexicoconsortium.org/science/DataDirectories.php>`_ .

    Parameters
    ----------
    ect_cdf_path : str
        Path to CDF RBSP ECT Level 3 pitch angle resolved data file
    emphesis_cdf_path : str
        Path to CDF RBSP ECT Level 3 pitch angle resolved data file

    Returns
    -------
    insitu_datasets : List[:py:class:`~InSituObservation`]
        Observations loaded from disk and organized for further processing
    """
    # Load variables from ECT --------------------------------------------------
    ect_cdf = cdflib.CDF(ect_cdf_path)
    ect_times = np.array(
        [
            datetime(1, 1, 1) + timedelta(days=-366, milliseconds=dt)
            for dt in ect_cdf.varget("Epoch")
        ]
    )
    energies = ect_cdf.varget("FEDU_Energy") * 1000  # keV -> eV
    flux = ect_cdf.varget("FEDU")
    pitch_angles = ect_cdf.varget("FEDU_Alpha")
    sc_positions_geo = (ect_cdf.varget("Position") * units.km).to(R_earth).value
    Bmodel = nanoTesla2Gauss(ect_cdf.varget("B_Calc"))
    
    # Load variables from EMPHESIS ---------------------------------------------
    em_cdf = cdflib.CDF(emphesis_cdf_path)

    em_times_raw = cdflib.cdfepoch.to_datetime(em_cdf.varget('Epoch'))    
    em_times = np.array([pd.Timestamp(t).to_pydatetime() for t in em_times_raw])

    em_Bobs_nT = em_cdf.varget('Magnitude') # units of nT    
    em_Bobs = nanoTesla2Gauss(em_Bobs_nT)

    mask = em_Bobs > 0
    
    # Interpolate onto times of ECT
    Bobs = np.interp(        
        date2num(ect_times),
        date2num(em_times[mask]),
        em_Bobs[mask]
    )

    # Convert spacecraft positions to sm coordinate system
    x_geo, y_geo, z_geo = sc_positions_geo.T
    dates = [int(time.strftime("%Y%m%d")) for time in ect_times]
    uts = [int(time.strftime("%H")) + time.minute / 60 for time in ect_times]
    
    x_sm, y_sm, z_sm = gp.Coords.ConvCoords(
        x_geo, y_geo, z_geo, dates, uts, CoordIn="GEO", CoordOut="SM"
    )

    sc_positions_sm = np.array([x_sm, y_sm, z_sm]).T

    x_gsm, y_gsm, z_gsm = gp.Coords.ConvCoords(
        x_geo, y_geo, z_geo, dates, uts, CoordIn="GEO", CoordOut="GSM"
    )

    sc_positions_gsm = np.array([x_gsm, y_gsm, z_gsm]).T
    
    # variables present in every object, will be copied for each timestep
    base_contents = {}
    base_contents["energies"] = energies
    base_contents["pitch_angles"] = pitch_angles

    # loop through timestep
    insitu_observations = []

    for i in range(ect_times.size):
        contents = base_contents.copy()
        contents["time"] = ect_times[i]
        contents["flux"] = flux[i]
        contents["Bmodel"] = Bmodel[i]        
        contents["Bobs"] = Bobs[i]
        contents["sc_position"] = tuple(sc_positions_sm[i, :].tolist())
        contents["sc_position_geo"] = tuple(sc_positions_geo[i, :].tolist())
        contents["sc_position_gsm"] = tuple(sc_positions_gsm[i, :].tolist())
        
        insitu_observations.append(InSituObservation(**contents))

    # Return instance of list of InSituObservation
    return insitu_observations
