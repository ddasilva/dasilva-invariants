da Silva Invariants
===================
   
This package provides tools for radiation belt physicists to calculate the adiabiatic invariant K and L* from from gridded models of Earth's magnetic field. For more information on the technique, see `Methodology <methodology.html>`_.

This package supports the `T96 <https://geo.phys.spbu.ru/~tsyganenko/empirical-models/magnetic_field/t96/>`_ and `TS05 <https://geo.phys.spbu.ru/~tsyganenko/empirical-models/magnetic_field/ts05/>`_ empirical Tsyganenko magnetic field models, as well as the `LFM <https://doi.org/10.1016/j.jastp.2004.03.020>`_ and `SWMF <https://clasp.engin.umich.edu/research/theory-computational-methods/space-weather-modeling-framework/>`_ MHD simulation code.

Installing the Dependencies
----------------------
To use this module, install the conda environment file (or copy it into your own), which will also install the module. The ability to compile fortran files is required.

.. code::

   $ conda env create -f environment.yml
   $ conda activate dasilva-invariants

At a Glance
-----------

Calculating L* from TS05
+++++++++++++++++++++++
Below is code which calculates L* using the magnetic fields obtain from TS05 and placed on a regular grid, for a particle observated with a pitch angle of 60° observed at (-6.6, 0, 0) R :sub:`E` (SM coordinate system).

.. code-block:: python

    from dasilva_invariants import models, invariants
    from datetime import datetime
    import numpy as np

    # Get TS05 model input parameters
    time = datetime(2015, 10, 2)
    url = "http://mag.gmu.edu/ftp/QinDenton/5min/merged/latest/WGparameters5min-latest.d.zip",
    params = models.get_tsyganenko_params(time, url)
    
    # Evaluate TS05 model on regular grid 
    axis = np.arange(-10, 10, 0.15)
    x, y, z = np.meshgrid(axis, axis, axis)
    model = models.get_tsyganenko(
        "TS05", params, time,
        x_re_sm_grid=x,
        y_re_sm_grid=y,
        z_re_sm_grid=z,
        inner_boundary=1
    )

    # Calculate L* 
    result = invariants.calculate_LStar(
        model,
        starting_point=(-6.6, 0, 0),
        starting_pitch_angle=60
    )

    print(f"L* = {result.LStar}")



Calculating K from SWMF 
+++++++++++++++++++++++
This code calculates the second adiabatic invariant K for a particle bouncing through (-6.6, 0, 0) R :sub:`E` (SM coordinate system) and mirroring at at 50° magnetic latitude, using magnetic fields from SWMF simulation output in CDF format (as obtained from the CCMC).

.. code-block:: python

    from dasilva_invariants import models, invariants

    model = models.get_model("SWMF_CDF", "3d__var_1_e20151221-001700-014.out.cdf")

    result = invariants.calculate_K(
        model,
        starting_point=(-6.6, 0, 0),
        mirror_latitude=50
    )

    print(f"K = {result.K}")


User Documentation
------------------
.. toctree::
  :maxdepth: 1
             
  dasilva_invariants.rst

