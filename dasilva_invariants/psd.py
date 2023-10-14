"""Calculation of f(L*) vs L* profiles at fixedb mu and K. This code implements the algorithm
outlined for this task in `Green and Kivelson, 2004 <https://doi.org/10.1029/2003JA01015>`_.
"""
from dataclasses import dataclass
from typing import Any, Dict

from astropy.constants import R_earth, m_e, m_p, c
from astropy import units
import numpy as np
from scipy.stats import linregress

from .insitu import InSituObservation
from .invariants import calculate_K, calculate_LStar, CalculateLStarResult
from .meshes import MagneticFieldModel

__all__ = ["CalculateLStarProfileResult", "calculate_LStar_profile"]


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
    particle: str


def calculate_LStar_profile(
    fixed_mu: float,
    fixed_K: float,
    insitu_observation: InSituObservation,
    mesh: MagneticFieldModel,
    particle: str = "electron",
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
    mesh : :py:class:~MagneticFieldModel`
        Grid and magnetic field, loaded using meshes module
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
            mesh,
            insitu_observation.sc_position,
            pitch_angle=pitch_angle,
            reuse_trace=reuse_trace,
        )
        Ks[i] = result.K
        reuse_trace = result._trace

    # Interpolate monotonic subject of pitch angle vs K curve to find solution
    # of this code section.
    mask = pitch_angles <= 90
    I = np.argsort(Ks[mask])

    (fixed_pitch_angle,) = np.interp([fixed_K], Ks[mask][I], pitch_angles[mask][I])

    # Compute and interpolate phase space density at fixed K, for each energy.
    #
    # Uses equation f = j / p^2, where
    #   f is phase space density
    #   j is flux
    #   p is momentum
    #  -----------------------------------------------------------------------
    f_step2 = np.zeros(energies.size, dtype=float)
    mass = {"electron": m_e, "proton": m_p}[particle]

    for i in range(energies.size):
        E = energies[i]
        p_squared = (E**2 + 2 * mass * c**2 * E) / c**2  # relativistic
        mask = flux[:, i] > 0

        if mask.any():
            f_step2[i] = np.interp(
                fixed_pitch_angle,
                pitch_angles[mask],
                (flux[:, i][mask] / p_squared).value,
            )
        else:
            f_step2[i] = np.nan

    f_step2 *= flux[:, 0].unit / p_squared.unit

    # Find fixed E associated with the first adiabatic invariant (fixed_mu)
    #
    # Solve the quadratic equation, taking real root. See Green 2004 (Journal of
    # Geophysical Research), Step 3.
    # ------------------------------------------------------------------------
    B = np.linalg.norm(mesh.interpolate(insitu_observation.sc_position))
    B *= units.G
    fixed_mu_units = fixed_mu * units.MeV / units.G
    fixed_pitch_angle_rad = np.deg2rad(fixed_pitch_angle)

    a = 1 / c**2
    b = 2 * m_e
    c_ = -2 * m_e * B * fixed_mu_units / np.sin(fixed_pitch_angle_rad) ** 2

    fixed_E = (-b + np.sqrt(np.square(b) - 4 * a * c_)) / (2 * a)
    fixed_E = fixed_E.to(units.MeV)

    # Interpolate the f_step2(E) structure at the fixed energy associated with
    # fixed_mu. Fit f_step2(E) to power law dist, the find f_step2(fixed_E).
    # ------------------------------------------------------------------------
    mask = np.isfinite(f_step2)

    # A linear regression of two log-space variables is a power law relation
    # in linear space.
    fit_x = np.log10(energies[mask].to(units.keV).value)
    fit_y = np.log10(f_step2[mask].value)  # type: ignore
    fit = linregress(fit_x, fit_y)

    fixed_E_unitless = fixed_E.to(units.keV).value
    f_final = 10 ** (fit.slope * np.log10(fixed_E_unitless) + fit.intercept)
    f_final *= f_step2.unit  # type: ignore

    # Convert f_final to proper phase space density units
    momentum_units = units.g * units.nm / units.s
    psd_units = 1 / (momentum_units * units.nm) ** 3
    psd_units = (c / (units.cm * units.MeV)) ** 3
    phase_space_density = f_final.to(psd_units).value

    # Calculate L* paired with this measurement
    # ------------------------------------------------------------------------
    lstar_result = calculate_LStar(
        mesh,
        insitu_observation.sc_position,
        starting_pitch_angle=fixed_pitch_angle,
        **calculate_lstar_kwargs,
    )

    return CalculateLStarProfileResult(
        phase_space_density=phase_space_density,
        LStar=lstar_result.LStar,
        lstar_result=lstar_result,
        fixed_mu=fixed_mu,
        fixed_K=fixed_K,
        particle=particle,
    )
