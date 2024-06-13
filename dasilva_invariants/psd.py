"""Calculation of f(L*) vs L* profiles at fixedb mu and K. This code implements the algorithm
outlined for this task in `Green and Kivelson, 2004 <https://doi.org/10.1029/2003JA01015>`_.
"""
from dataclasses import dataclass
import os
from typing import Any, Dict
import warnings


from astropy.constants import R_earth, m_e, m_p, c
from astropy import units
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.stats import linregress


from .insitu import InSituObservation
from .invariants import calculate_K, calculate_LStar, CalculateLStarResult
from .models import MagneticFieldModel

__all__ = ["CalculateLStarProfileResult", "calculate_LStar_profile"]


class UnableToCalculatePSD(Exception):
    """Generic exception for being unable to calcualte PSD"""


@dataclass
class CalculateLStarProfileResult:
    """Phase space density (PSD) observation, f(L*) and its associated L*. 

    Parameters
    -----------
    phase_space_density : float
       Phase space density at corresponding L*, in units of (c / (cm*MeV))**3
    LStar : float
       LStar corresponding to phase space density, unitless
    lstar_result : :py:class:`~dasilva_invariants.invariants.CalculateLStarResult`
       Result which fully specifies the drift shell
    fixed_mu : float
       Fixed first adiabatic invariant used in this calculation, units of
       MeV/G
    fixed_K : float
       Fixed second adiabatic invariant used in this calculation, units of
       sqrt(G) Re
    particle : {'electron', 'proton'}
       Type of particle used in this calculation
    """

    phase_space_density: float
    LStar: float
    lstar_result: CalculateLStarResult
    fixed_mu: float
    fixed_K: float    
    fixed_E: float
    pitch_angle: float
    B: float
    particle: str

    

def calculate_LStar_profile(
    fixed_mu: float,
    fixed_K: float,
    insitu_observation: InSituObservation,
    model: MagneticFieldModel,
    particle: str = "electron",
    cache_dir=None,
    calculate_lstar_kwargs: Dict[Any, Any] = {},
) -> CalculateLStarProfileResult:
    """Calculation of f(L*) vs L* profiles at fixed mu and K.

    Parameters
    ----------
    mu : float
        F ixed first adiabatic invariant, units of MeV/G
    K : float
        Fixed second adiabatic invariant, units of sqrt(G) Re
    insitu_observation : :py:class:`~InSituObservation`
        Observational data accompanying this measurement
    model : :py:class:~MagneticFieldModel`
        Grid and magnetic field, loaded using models module
    particle : {'electron', 'proton'}
        Set the particle type, either 'electron' or 'proton'
    calculate_lstart_kwargs : dict
        Dictionary of arguments to pass to
        calculate_LStar(). Use this to specify options such as mode
        or number of local times.

    Returns
    -------
    result : `~CalculateLStarProfileResult`
        Holds phase space density observation paired with L*.
    """
    assert particle in (
        "electron",
        "proton",
    ), f"calculate_LStar_profile(): Invalid particle {repr(particle)}"

    # Decide on B -------------------------------------------------------------
    #B = np.linalg.norm(model.interpolate(insitu_observation.sc_position)) * units.G
    #B = insitu_observation.Bmodel * units.G
    B = insitu_observation.Bobs * units.G
    
    # Extract variables from insitu_observation into local namespace with untis
    flux_units = 1 / (units.cm**2 * units.s * units.keV)  # also per ster

    flux = insitu_observation.flux * flux_units
    energies = insitu_observation.energies * units.eV
    pitch_angles = insitu_observation.pitch_angles
    sc_position = insitu_observation.sc_position * R_earth

    # Find the pitch angle associated with the K given at the given spacecraft
    # location.
    # -----------------------------------------------------------------------
    # Find K at each pitch angle
    Ks = np.zeros(pitch_angles.size, dtype=float)  # K at each pitch angle
    reuse_trace = None  # cache object for trace

    for i, pitch_angle in enumerate(pitch_angles):
        result = calculate_K(
            model,
            insitu_observation.sc_position,
            pitch_angle=pitch_angle,
            reuse_trace=reuse_trace,
            Blocal=B.value,
        )
        Ks[i] = result.K
        reuse_trace = result._trace

    # Interpolate monotonic subject of pitch angle vs K curve to find solution
    # of this code section.
    mask = (pitch_angles <= 90) & (Ks > 0)
    I = np.argsort(Ks[mask])

    if mask.sum() > 2:
        fit_x = np.log10(Ks[mask][I])
        fit_y = pitch_angles[mask][I]
        #fixed_pitch_angle = np.interp(np.log10(fixed_K), fit_x, fit_y)        
        fit = CubicSpline(fit_x, fit_y)
        #fit = make_smoothing_spline(fit_x, fit_y)
        fixed_pitch_angle = float(fit(np.log10(fixed_K)))
    else:
        raise UnableToCalculatePSD('Not enough points to interpolate K')

    if fixed_pitch_angle > 90 or fixed_pitch_angle < 0:
        raise UnableToCalculatePSD('Fixed Pitch Angle outside of bounds')        
    
    # Compute and interpolate flux at fixed K, for each energy.
    # -----------------------------------------------------------------------
    flux_step2 = np.zeros(energies.size, dtype=float)
    mass = {"electron": m_e, "proton": m_p}[particle]

    for i in range(energies.size):
        E = energies[i]
        mask = flux[:, i] > 0

        if mask.any():
            fit_x = pitch_angles[mask]
            fit_y = flux[:, i][mask].value
            flux_step2[i] = np.interp(
                fixed_pitch_angle,
                fit_x, fit_y
            )
        else:
            flux_step2[i] = np.nan

    flux_step2 *= flux[:, 0].unit 

    # Find fixed E associated with the first adiabatic invariant (fixed_mu)
    #
    # Solve the quadratic equation, taking real root. See Green 2004 (Journal of
    # Geophysical Research), Step 3.
    # ------------------------------------------------------------------------    
    fixed_mu_units = fixed_mu * units.MeV / units.G
    fixed_pitch_angle_rad = np.deg2rad(fixed_pitch_angle)

    a = 1 / c**2
    b = 2 * mass
    c_ = -2 * mass * B * fixed_mu_units / np.sin(fixed_pitch_angle_rad) ** 2

    fixed_E = (-b + np.sqrt(np.square(b) - 4 * a * c_)) / (2 * a)
    fixed_E = fixed_E.to(units.MeV)

    # Interpolate the flux_step2(E) structure at the fixed energy associated with
    # fixed_mu.
    # ------------------------------------------------------------------------
    mask = np.isfinite(flux_step2)
    with warnings.catch_warnings(action="ignore"):
        fit_x = np.log10(energies[mask].to(units.keV).value)
        fit_y = np.log10(flux_step2[mask].value)  # type: ignore
        
    fit_mask = np.isfinite(fit_x) & np.isfinite(fit_y)
    #fit = np.poly1d(np.polyfit(fit_x, fit_y, 1))
    #fit = CubicSpline(fit_x[fit_mask], fit_y[fit_mask])
    fit = make_smoothing_spline(fit_x[fit_mask], fit_y[fit_mask])
    fixed_E_unitless = fixed_E.to(units.keV).value

    # import pylab as plt
    # plt.plot(10**fit_x, 10**fit(fit_x), 'r-', label='Power Law Fit')
    # plt.plot(10**fit_x, 10**make_smoothing_spline(fit_x[fit_mask], fit_y[fit_mask])(fit_x), 'g-', label='Smoothing Spline Fit')
    # plt.plot(10**fit_x, 10**fit_y, 'k.', label='Measurements')
    # plt.axvline(fixed_E_unitless, color='k', label='Target E')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('E (keV)')
    # plt.ylabel('Flux')
    # plt.grid(color='#ccc', linestyle='dashed')
    # plt.legend()
    # plt.savefig('Power_Law.png', dpi=300)
    # plt.close()
    # raise RuntimeError()
    
    with warnings.catch_warnings(action="ignore"):
        try:
            flux_final = 10 ** float(fit(np.log10(fixed_E_unitless)))
        except:
            flux_final = np.nan
            
    flux_final *= flux_step2.unit  # type: ignore
    p_squared = (fixed_E**2 + 2 * mass * c**2 * fixed_E) / c**2  # relativistic     
    f_final = flux_final / p_squared
    
    # Convert f_final to proper phase space density units
    momentum_units = units.g * units.nm / units.s
    psd_units = 1 / (momentum_units * units.nm) ** 3
    psd_units = (c / (units.cm * units.MeV)) ** 3
    phase_space_density = f_final.to(psd_units).value

    # Calculate L* paired with this measurement
    # ------------------------------------------------------------------------
    cache_key = insitu_observation.time.isoformat()
    cache_fname = f'{cache_dir}/{cache_key}.txt'
    
    if cache_dir and os.path.exists(cache_fname):
        lstar_result = None
        LStar = float(open(cache_fname).read())
    else:
        lstar_result = calculate_LStar(
            model,
            insitu_observation.sc_position,
            starting_pitch_angle=fixed_pitch_angle,
            Blocal=B.value,
            **calculate_lstar_kwargs,
        )
        LStar = lstar_result.LStar

        if cache_dir:
            with open(cache_fname, 'w') as fh:
                fh.write(str(LStar))
        
    return CalculateLStarProfileResult(
        phase_space_density=phase_space_density,
        LStar=LStar,
        lstar_result=lstar_result,
        fixed_mu=fixed_mu,
        fixed_K=fixed_K,
        fixed_E=fixed_E_unitless,
        pitch_angle=fixed_pitch_angle,
        B=B.value,
        particle=particle,
    )
