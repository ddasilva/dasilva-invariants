"""Constants for the package not found in astropy or scipy."""

from . import utils

# Earth dipole moment, in Gauss
(EARTH_DIPOLE_B0,) = utils.nanoTesla2Gauss([30e3])
