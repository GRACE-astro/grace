.. _gracepy-physics-utils:

Physics Utilities
==================

GRACEpy provides several modules for common physics and analysis tasks: physical constants, unit system conversions, gravitational wave post-processing, Kerr-Schild coordinate transformations, and equation of state table handling.


Physical Constants
********************

The ``analysis.constants`` module defines physical constants in SI and CGS units, particle masses, and conversion factors commonly needed in numerical relativity and astrophysics.

.. code-block:: python

    import analysis.constants as pc

    pc.G_si        # gravitational constant [m^3/(kg s^2)]
    pc.c_si        # speed of light [m/s]
    pc.Msun_si     # solar mass [kg]
    pc.Kb_si       # Boltzmann constant [J/K]
    pc.Mparsec_si  # megaparsec [m]

CGS variants are available with the ``_cgs`` suffix (e.g. ``pc.c_cgs``, ``pc.G_cgs``, ``pc.Msun_cgs``).

**Particle masses** are given in MeV (``me_MeV``, ``mp_MeV``, ``mn_MeV``) and converted to SI and CGS (``me_si``, ``me_cgs``, etc.).

**Energy conversions**: ``erg_to_J``, ``eV_to_J``, ``MeV_to_J``, ``eV_to_kg``, ``MeV_to_kg``, ``eV_to_erg``, ``MeV_to_erg``, ``eV_to_g``, ``MeV_to_g``.

**Code unit conversions** (assuming :math:`G = c = M_\odot = 1`):

.. code-block:: python

    pc.CU_to_m       # code units -> meters
    pc.CU_to_s       # code units -> seconds
    pc.CU_to_ms      # code units -> milliseconds
    pc.CU_to_cm      # code units -> centimeters
    pc.CU_to_J       # code units -> Joules
    pc.CU_to_erg     # code units -> ergs
    pc.CU_to_Gauss   # code units -> Gauss
    pc.CU_to_Tesla   # code units -> Tesla


Unit Systems
**************

The ``eos.units_system`` module defines a ``unit_system`` class that represents a coherent set of physical units and automatically derives compound units (velocity, pressure, energy density, etc.) from the base dimensions.

.. code-block:: python

    from eos.units_system import (
        unit_system,
        SI_UNIT_SYSTEM,
        CGS_UNIT_SYSTEM,
        GEOM_UNIT_SYSTEM,
        COMPOSE_UNIT_SYSTEM,
    )

Each ``unit_system`` is constructed from four base dimensions — mass, length, time, and magnetic field strength — expressed in SI. The following derived units are computed automatically:

- ``velocity``, ``acceleration``, ``force``
- ``surface``, ``volume``
- ``pressure``, ``dens`` (mass density), ``energy``, ``edens`` (energy density)

Dividing two unit systems produces a conversion factor object:

.. code-block:: python

    uconv = CGS_UNIT_SYSTEM / GEOM_UNIT_SYSTEM
    pressure_cgs = pressure_geom * uconv.pressure

Pre-defined unit systems:

- ``SI_UNIT_SYSTEM`` — base SI (kg, m, s)
- ``CGS_UNIT_SYSTEM`` — CGS (g, cm, s)
- ``GEOM_UNIT_SYSTEM`` — geometric units with :math:`G = c = M_\odot = 1`
- ``COMPOSE_UNIT_SYSTEM`` — natural units used by the `CompOSE <https://compose.obspm.fr/>`_ equation of state database (MeV, fm)


Gravitational Wave Analysis
******************************

The ``analysis.gw_utils`` module provides routines for post-processing gravitational wave data extracted from GRACE simulations.

Fixed-frequency integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Double-integrate :math:`r\Psi_4` to obtain the strain :math:`h` using the fixed-frequency integration (FFI) method:

.. code-block:: python

    from analysis.gw_utils import fixed_frequency_integration

    h = fixed_frequency_integration(t, psi4, f0, N=2, window="tukey", wpars=[0.2])

Arguments:

- ``t`` — uniform time array
- ``psi4`` — complex :math:`r\Psi_4(t)` timeseries
- ``f0`` — cutoff frequency (frequencies below ``f0`` are suppressed)
- ``N`` — number of integrations (default 2: :math:`\Psi_4 \to h`)
- ``window`` — windowing function applied before the FFT (``"tukey"``, ``"blackman"``, or ``None``)

Phase, frequency, and retarded time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from analysis.gw_utils import get_phase, get_inst_frequency, retarded_time

    phi = get_phase(h)              # unwrapped phase of a complex waveform
    f   = get_inst_frequency(t, h)  # instantaneous GW frequency
    t_ret = retarded_time(t, r, M)  # retarded time using the tortoise coordinate

Waveform alignment
~~~~~~~~~~~~~~~~~~~~

Align two waveforms using the procedure of `Boyle et al. (2009) <https://doi.org/10.1103/physrevd.78.104020>`_:

.. code-block:: python

    from analysis.gw_utils import align_waveforms

    psi2_aligned, phi2_aligned, dt, dphi = align_waveforms(t, psi1, psi2, t1, t2)

This minimizes the integrated squared phase difference in the window ``[t1, t2]`` to find the optimal time and phase shifts.

Nakano extrapolation
~~~~~~~~~~~~~~~~~~~~~~

Extrapolate :math:`r\Psi_4` to infinite extraction radius using the method of Nakano et al.:

.. code-block:: python

    from analysis.gw_utils import nakano_extrap

    rpsi_inf = nakano_extrap(t, rpsilm, Madm, r, l, f0)


Kerr-Schild Transformations
******************************

The ``analysis.kerr_schild`` module provides coordinate transformations and metric quantities for the Kerr spacetime in Cartesian Kerr-Schild (CKS) coordinates.

.. code-block:: python

    import analysis.kerr_schild as ks

    # Coordinate transformation: CKS -> Boyer-Lindquist
    r_bl, theta_bl, phi_bl = ks.cks_to_bl(xyz, a)

    # Metric quantities at a given point
    alpha = ks.get_alpha(xyz, a)     # lapse function
    beta  = ks.get_beta(xyz, a)      # shift vector (3 components)
    gamma = ks.get_gamma(xyz, a)     # spatial metric (6 symmetric components)
    gi    = ks.get_gammainv(xyz, a)  # inverse spatial metric
    sg    = ks.get_sqrtg(xyz, a)     # sqrt(det(gamma))

Here ``xyz`` is an array of Cartesian coordinates ``[x, y, z]`` and ``a`` is the dimensionless Kerr spin parameter.


Equation of State Tables
***************************

The ``eos.eos_table`` module provides classes for reading, converting, and exporting nuclear equation of state (EOS) tables.

Reading a table
~~~~~~~~~~~~~~~~~~

Two table formats are supported: `CompOSE <https://compose.obspm.fr/>`_ (HDF5) and `stellarcollapse.org <https://stellarcollapse.org/>`_ (HDF5).

.. code-block:: python

    from eos.eos_table import compose_eos_table, scollapse_eos_table

    # CompOSE format
    eos = compose_eos_table("eos_compose.h5")

    # stellarcollapse format
    eos = scollapse_eos_table("eos_stellarcollapse.h5")

Both readers convert the table data into GRACE's internal geometric unit system (:math:`G = c = M_\odot = 1`).

The table is stored on a regular grid in :math:`(\log\rho, \log T, Y_e)` and exposes the following thermodynamic quantities: ``logpress``, ``logeps`` (specific internal energy), ``entropy``, ``cs2`` (sound speed squared), ``mu_e``, ``mu_p``, ``mu_n`` (chemical potentials), and composition fractions ``Xn``, ``Xp``, ``Xa``.

Interpolation
~~~~~~~~~~~~~~~

Pressure (or other quantities) can be interpolated on the 3D grid:

.. code-block:: python

    p = eos.p_of_rho_T_ye(rho, T, ye)

Exporting cold (isothermal) slices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common workflow is to extract a cold slice at beta equilibrium for use in initial data solvers:

.. code-block:: python

    eos.export_cold_table(
        "cold_eos.dat",
        tab_format="GRACE",        # or "LORENE"
        temperature=1.0e-15,       # MeV
        resample=500,              # resample to 500 points (optional)
        attach_polytrope=True,     # extend with a Gamma=4/3 polytrope at low density
        remove_radiation=True,     # subtract photon pressure contribution
        rho_junction=1e-11,        # density at which to attach the polytrope
    )

The ``tab_format`` argument selects the output format: ``"GRACE"`` produces an ASCII table in GRACE's native format, while ``"LORENE"`` writes a table compatible with the LORENE initial data library.

LORENE tables
~~~~~~~~~~~~~~~

The ``lorene_table`` class reads EOS tables in the LORENE ASCII format and provides interpolation routines:

.. code-block:: python

    from eos.eos_table import lorene_table

    lt = lorene_table("eos_lorene.dat")

    # Interpolation by baryon number density
    p = lt.p__n(n_b)     # pressure
    e = lt.e__n(n_b)     # energy density
    h = lt.h__n(n_b)     # specific enthalpy
