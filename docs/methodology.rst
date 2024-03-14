Methodology and References
==========================

This technique is a tool for studying radiation belt physics. It computes adiabatic invariant quantities, known as K and L*. 

**What are the Adiabatic Invariants?**
Adiabatic invariants are quantities in classical mechanics that remain constant under slow, reversible changes in a system. They arise in systems where changes occur gradually, allowing the system to adjust without dissipating energy. The simplest example of an adiabatic invariant in the motion of charged particles in a magnetic field, known as the magnetic moment. Another example is the adiabatic invariant associated with the action variable in the motion of a pendulum or a simple harmonic oscillator. Adiabatic invariants play a crucial role in understanding the behavior of physical systems in various fields, including plasma physics, quantum mechanics, and celestial mechanics.

In radiation belt physics, adiabatic invariants refer to quantities that remain approximately constant as charged particles move in the Earth's magnetic field. These invariants help describe the dynamics of charged particles trapped in the Earth's magnetic field, forming the Van Allen radiation belts. Understanding these adiabatic invariants is crucial for studying the acceleration, transport, and loss processes of energetic particles in the radiation belts, which have implications for space weather and spacecraft operations.

**How are the Adiabatic Invariants Calculated?**
This package uses the "Roederer Method" for calculating L*, and integration over the bounce path to calculate K. 

The strongest way to learn the methodology of this library is to read the publication, `da Silva et al., 2024: Numerical Calculations of Adiabatic Invariants from MHD-Driven Magnetic Fields <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Numerical+Calculations+of+Adiabatic+Invariants+from+MHD-Driven+Magnetic+Fields&btnG=>`_.

**References**
    * `Roederer, Juan G., and Hui Zhang. Dynamics of magnetically trapped particles. Springer-Verlag Berlin An, 2016.  <https://link.springer.com/book/10.1007/978-3-642-41530-2>`_
    * `Schulz, M. "Canonical coordinates for radiation-belt modeling." GEOPHYSICAL MONOGRAPH-AMERICAN GEOPHYSICAL UNION 97 (1996): 153-160. <https://doi.org/10.1029/GM097p0153>`_
    * `Green, Janet C., and M. G. Kivelson. "Relativistic electrons in the outer radiation belt: Differentiating between acceleration mechanisms." Journal of Geophysical Research: Space Physics 109.A3 (2004). <https://doi.org/10.1029/2003JA010153>`_
