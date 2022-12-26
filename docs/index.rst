da Silva Invariants
===================
   
.. note:: This package is under development and is not yet publically available.

This package provides tools for radiation belt physicists to peform adiabiatic invariant analysis from magnetic field models of the Earth's magnetosphere. These parameters characterize three periodic motions associated with particles trapped in magnetospheres, and are used to study the dynamical system. For more information on the technique, see `Introduction to Method <intro.html>`_.

This package supports the `T96 <https://geo.phys.spbu.ru/~tsyganenko/empirical-models/magnetic_field/t96/>`_ and `TS05 <https://geo.phys.spbu.ru/~tsyganenko/empirical-models/magnetic_field/ts05/>`_ empirical Tsyganenko magnetic field models, as well as the `LFM <https://doi.org/10.1016/j.jastp.2004.03.020>`_ MHD simulation code. Support for additional models is planned in the future.

Installing the package
----------------------
To install this package, you can use

.. code::

   pip install dasilva-invariants

Calculating L* from TS05
------------------------
Below is code which calculates L* using the magnetic fields obtain from TS05 and placed on the LFM grid, for a particle observated with a pitch angle of 60° observed at (-6.6, 0, 0) R :sub:`E` (SM coordinate system). The LFM grid is specified by an LFM HDF4 file. This code is assumed to be run after the `PyGeopack environment is configured <configure_pygeopack.html>`_.

.. code-block:: python

    from dasilva_invariants import meshes, invariants
    from datetime import datetime

    ts05_mesh = meshes.get_tsyganenko_on_lfm_grid_with_auto_params(
        "TS05",
        time=datetime(2015, 10, 2),
        lfm_hdf4_path="LFM_mhd_2013-10-02T00-00-00Z.hdf",
        param_path="http://virbo.org/ftp/QinDenton/hour/merged/latest/WGhour-latest.d.zip",
        tell_params=True,
    )

    try:
        result = invariants.calculate_LStar(
            ts05_mesh,
            starting_point=(-6.6, 0, 0),
            starting_pitch_angle=60
        )
    except invariants.DriftShellSearchDoesntConverge as e:
        print("Unable to calculate drift shell; may not exist.")
	raise SystemExit(1)

    print(f"L* = {result.LStar}")


Calculating K from LFM
----------------------
This code calculates the second adiabatic invariant K for a particle bouncing through (-6.6, 0, 0) R :sub:`E` (SM coordinate system) and mirroring at at 50° magnetic latitude, using magnetic fields from the LFM simulation code. 

.. code-block:: python

    from dasilva_invariants import meshes, invariants

    lfm_mesh = meshes.get_lfm_hdf4_data("LFM_mhd_2013-10-02T00-00-00Z.hdf")

    result = invariants.calculate_K(
        lfm_mesh,
        starting_point=(-6.6, 0, 0),
        mirror_latitude=50
    )

    print(f"K = {result.K}")


User Documentation
------------------
.. toctree::
  :maxdepth: 1
             
  dasilva_invariants.rst

