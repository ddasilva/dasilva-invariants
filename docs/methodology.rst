Methodology
===========

This technique is a tool for studying radiation belt physics, and is really only useful in that context. The most formal way to learn the methodology of this library is to read `da Silva et al., 2024: Numerical Calculations of Adiabatic Invariants from MHD-Driven Magnetic Fields <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Numerical+Calculations+of+Adiabatic+Invariants+from+MHD-Driven+Magnetic+Fields&btnG=>`_. 

The adiabatic invariants parameterization of phase space density (PSD) is a tool to study trapped particle motion in magnetospheres. In the adiabatic invariant parameterization of PSD, a trapped particle’s velocity is characterized by three quantities invariant over its three periodic motions (gyrations, bounce, azimuthal drift).  The invariant coordinate (μ, K, L*) is a function of the particle’s velocity, the particle’s position in space/time, and the field along the particle’s full orbit.  Provided the field along the orbit does not change too fast, the invariants are also conserved during large-scale reconfigurations of the magnetosphere. This makes them an effective underlying "state variable" of the dynamical system, obtained through a combination of observation and modeling.

References
----------
    * `Roederer, Juan G., and Hui Zhang. Dynamics of magnetically trapped particles. Springer-Verlag Berlin An, 2016.  <https://link.springer.com/book/10.1007/978-3-642-41530-2>`_
    * `Schulz, M. "Canonical coordinates for radiation-belt modeling." GEOPHYSICAL MONOGRAPH-AMERICAN GEOPHYSICAL UNION 97 (1996): 153-160. <https://doi.org/10.1029/GM097p0153>`_
    * `Green, Janet C., and M. G. Kivelson. "Relativistic electrons in the outer radiation belt: Differentiating between acceleration mechanisms." Journal of Geophysical Research: Space Physics 109.A3 (2004). <https://doi.org/10.1029/2003JA010153>`_
