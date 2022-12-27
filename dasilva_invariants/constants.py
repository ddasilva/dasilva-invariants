"""Constants for the package not found in astropy or scipy."""

from . import utils

# Earth dipole moment, in Gauss
(EARTH_DIPOLE_B0,) = utils.nanoTesla2Gauss([30e3])

# Inner boundary of LFM magnetic field model
LFM_INNER_BOUNDARY = 2.11
