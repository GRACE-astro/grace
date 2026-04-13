.. _gracepy-data-analysis:

Data Analysis
==============

This section covers the tools for reading and analyzing GRACE simulation output:
volume and plane data (via VTK/XMF), scalar timeseries, gravitational wave modes, and
the high-level simulation interface that ties everything together.


XMF Descriptors
****************

XMF descriptors are light-weight aggregators of metadata regarding the structure of HDF5 output from GRACE. They are needed to process the data both in ParaView and in Python.

After installing the GRACEpy library you can run the ``create_descriptor`` command (see :doc:`cli_tools`). Its syntax is

.. code-block:: bash

    create_descriptor <path_to_data> <descriptor_name> [--mode <mode>] [--filter <filter>]

The optional ``mode`` argument can be ``auto`` ``temporal`` ``spatial`` ``spherical`` (default ``auto``). This specifies what kind of data is being processed,
with ``auto`` trying to detect it from the file names, ``temporal`` indicating that the descriptor should be a temporal collection of grids (i.e. a timeseries of volume files or surface files from a single surface),
``spatial`` indicating that the files are a spatial collection (e.g. multiple surfaces) and ``spherical`` indicating that the data represents spherical surfaces.

The ``filter`` argument allows you to pass a regular expression used to filter out filenames.

When processing the data in Python you should almost always use a temporal collection, since this allows you to process time-series of 2D and 3D data. For example, if a directory contains xy, xz and yz data and you want
to process the xy plane you would do

.. code-block:: bash

    create_descriptor ./data_dir desc_xy --mode temporal --filter "*xy*"


VTK Reader
***********

The ``grace_xmf_reader`` class (in ``grace_tools.vtk_reader_utils``) ingests XMF descriptors and provides a convenient interface to extract volume and plane data.

.. code-block:: python

    import grace_tools.vtk_reader_utils as gtv
    reader = gtv.grace_xmf_reader("descriptor.xmf")

This loads only the metadata and is lightweight even for very large datasets. The reader exposes the available output times and variables:

.. code-block:: python

    reader.available_times()
    # [0.0, 1.0, 2.0]
    reader.available_variables()
    # ["rho", "vel"]

Extracting data
~~~~~~~~~~~~~~~~

The main purpose of the reader is to output the data in a format that can be plotted or used in post-processing:

.. code-block:: python

    xyz, rho = reader.get_var("rho", time=1.0)
    # xyz: np.array [ncells, 3] — cell-center coordinates
    # rho: np.array [ncells]    — data values

In VTK terms, the data is represented as an unstructured grid, so it needs to be plotted using triangulations in ``matplotlib``:

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    fig, ax = plt.subplots()
    triangulation = tri.Triangulation(xyz[:, 0], xyz[:, 1])
    ax.tricontourf(triangulation, rho, cmap="inferno", levels=500)


1D and 2D slicing
~~~~~~~~~~~~~~~~~~~

The reader can also sample data along a 1D line or slice 3D data along a plane:

.. code-block:: python

    # 1D line sample between two points
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])
    xyz1D, rho1D = reader.get_var_1D_slice("rho", time, p1, p2, npoints)

    # 2D plane slice
    xyz2D, rho2D = reader.get_var_2D_slice("rho", time, plane_normal, plane_origin)

The 1D slice returns a ``[npoints, 3]`` array of coordinates and the corresponding interpolated values. The line is constructed between ``p1`` and ``p2`` using ``npoints`` uniformly spaced points.

.. note::

    For best results, choose points and spacings that are well adapted to the numerical grid.


Scalar Data
*************

The ``grace_scalars_reader`` class (in ``grace_tools.scalar_reader_utils``) reads all ``.dat`` files from one or more output directories and categorizes them automatically.

.. code-block:: python

    import grace_tools.scalar_reader_utils as gts
    scalars = gts.grace_scalars_reader("/path/to/output_scalar")

Scalar files are classified into the following categories:

**Reductions** — variables ending with ``_max``, ``_min``, ``_norm2``, or ``_integral``:

.. code-block:: python

    # Access a timeseries by variable name
    ts = scalars.maximum["rho"]
    ts.time       # np.array of coordinate times
    ts.data       # np.array of data values (or dict for multi-column files)
    ts.iteration  # np.array of iteration numbers

    # List all available variables for a given reduction
    scalars.maximum.available_vars()

**EM energy** — the ``E_em.dat`` diagnostic:

.. code-block:: python

    scalars.em_energy.time
    scalars.em_energy.columns   # column names from the file header

**Mass flux** — files matching ``Mdot_{type}_{detector}.dat``, organized by detector:

.. code-block:: python

    scalars.mass_flux["GW_1"]["rest_mass"]   # timeseries for rest-mass flux at GW_1

**Compact object locations** — files matching ``co_{name}_loc.dat``:

.. code-block:: python

    scalars.co_locations["BH_0"].time
    scalars.co_locations["BH_0"].data  # multi-column position data

Restart merging
~~~~~~~~~~~~~~~~

When multiple directories are passed (e.g. from successive restarts), scalar files with matching names are automatically merged in time order:

.. code-block:: python

    scalars = gts.grace_scalars_reader([
        "/sim/restart_0000/output_scalar",
        "/sim/restart_0001/output_scalar",
    ])


Gravitational Wave Data
*************************

The ``grace_gw_data`` class (in ``grace_tools.gw_reader_utils``) reads rPsi4 (or Psi4) files, groups them by detector and spherical harmonic mode ``(l, m)``, and stores the result as complex timeseries.

.. code-block:: python

    import grace_tools.gw_reader_utils as gtg
    gw = gtg.grace_gw_data("/path/to/output_scalar")

    gw.available_detectors()
    # ['GW_1', 'GW_2']

    mode22 = gw["GW_1"][2, 2]   # grace_gw_mode object
    mode22.time                  # coordinate time array
    mode22.data                  # complex rPsi4_{22}(t)

Each ``grace_gw_mode`` stores the spherical harmonic indices ``l`` and ``m``, the iteration and time arrays, and the complex ``data`` array (real and imaginary parts are paired automatically from separate files).

Initial data metadata
~~~~~~~~~~~~~~~~~~~~~~~

The ``grace_gw_data`` object can optionally carry the total ADM mass and initial orbital frequency of the system:

.. code-block:: python

    gw = gtg.grace_gw_data("/path/to/output_scalar", Madm=2.75, omega0=0.00894)
    gw.Madm    # 2.75
    gw.omega0  # 0.00894

When using ``grace_simulation``, these are **auto-detected from the FUKA info file** if the initial data type in the parameter file is ``fuka``. They can also be set explicitly via the ``grace_simulation`` constructor, which takes precedence over auto-detection:

.. code-block:: python

    sim = grace_simulation("/path/to/sim", Madm=2.75, omega0=0.00894)

For non-FUKA simulations, both default to ``None``.

Computing strain
~~~~~~~~~~~~~~~~~~

The ``compute_strain`` method integrates :math:`r\Psi_4` to obtain :math:`r \cdot h` for a given detector and mode, using the stored ``omega0`` as the default cutoff frequency:

.. code-block:: python

    t, rh = gw.compute_strain("GW_1", (2, 2))

    # or with an explicit cutoff frequency:
    t, rh = gw.compute_strain("GW_1", (2, 2), f0=0.001)

.. note::

    Since GRACE outputs :math:`r\Psi_4`, all integrated quantities carry a factor of :math:`r` (the extraction radius in code units, :math:`M_\odot`). In particular, ``compute_strain`` returns :math:`r \cdot h`, not :math:`h`. To obtain the physical strain at a distance :math:`D`, divide by :math:`D` (in the same units).

Additional keyword arguments (``N``, ``window``, ``wpars``) are passed through to ``fixed_frequency_integration`` from ``analysis.gw_utils``.

Radiated quantities
~~~~~~~~~~~~~~~~~~~~~

Three methods compute radiated energy, angular momentum, and linear momentum by summing over all available modes at a detector. All formulas work directly with :math:`r\Psi_4` and its time integrals — the :math:`r^2` factors cancel in the bilinear products, so no explicit extraction radius is needed.

**Radiated energy:**

.. code-block:: python

    t, dEdt, E = gw.radiated_energy("GW_1")

Returns the energy flux :math:`dE/dt = \frac{1}{16\pi}\sum_{l,m}|r\dot{h}_{lm}|^2` and the cumulative radiated energy :math:`E(t)`.

**Radiated angular momentum** (z-component):

.. code-block:: python

    t, dJzdt, Jz = gw.radiated_angular_momentum("GW_1")

Returns :math:`dJ_z/dt` and cumulative :math:`J_z(t)`. The sign convention is that :math:`dJ_z/dt < 0` for a system losing angular momentum.

**Radiated linear momentum** (gravitational wave recoil):

.. code-block:: python

    t, dPdt, P = gw.radiated_linear_momentum("GW_1")
    # dPdt and P have shape (3, npoints) for x, y, z components

The linear momentum computation uses coupling coefficients from the spin-weight :math:`s = -2` spherical harmonic recurrence relations to couple adjacent :math:`(l,m)` modes. The recoil (kick) velocity of the remnant is :math:`v_{\mathrm{kick}} = -P / M_{\mathrm{remnant}}`.


Detectors
***********

The ``grace_detector`` and ``grace_detector_set`` classes (in ``grace_tools.detector_utils``) provide a unified per-detector view that combines metadata from the parameter file with data from the scalar and GW readers.

.. code-block:: python

    from grace_tools.detector_utils import grace_detector_set

    # Usually built automatically by grace_simulation, but can be
    # constructed from a parsed parameter file:
    dset = grace_detector_set.from_parfile_config(config)

Each ``grace_detector`` carries:

- ``name``, ``radius``, ``center`` — from the parameter file
- ``resolution``, ``sampling_policy`` — detector discretization parameters
- ``gw`` — reference to a ``grace_gw_detector`` (GW modes at this detector)
- ``mass_flux`` — reference to a ``grace_timeseries_array`` (mass flux data)

GW modes can be accessed directly from the detector:

.. code-block:: python

    det = sim.detectors["GW_1"]
    det.radius              # extraction radius
    det[2, 2]               # shortcut for det.gw[2, 2]
    det.mass_flux["rest_mass"]


The Simulation Interface
**************************

The ``grace_simulation`` class (in ``grace_tools.simutils``) is the main entry point for post-processing. It parses the parameter file, auto-discovers output directories (including across restarts), builds XMF descriptors, and exposes all readers as attributes.

.. code-block:: python

    from grace_tools.simutils import grace_simulation
    sim = grace_simulation("/path/to/simulation")

Constructor arguments:

- ``simdir`` — path to the simulation directory
- ``parfile`` (optional) — explicit path to the YAML parameter file. If omitted, the class searches ``config/parfile/`` (simpilot layout) or the simulation root for a ``.yaml`` file.
- ``ppdir`` (optional) — directory for post-processing output (descriptors, plots). Defaults to ``./plots``.
- ``Madm`` (optional) — total ADM mass of the system. Auto-detected from FUKA initial data if not provided.
- ``omega0`` (optional) — initial orbital frequency. Auto-detected from FUKA initial data if not provided.

Available attributes:

.. code-block:: python

    sim.name       # simulation name from the parameter file
    sim.xyz        # grace_xmf_reader for volume output
    sim.xy         # grace_xmf_reader for the xy plane
    sim.xz         # grace_xmf_reader for the xz plane
    sim.yz         # grace_xmf_reader for the yz plane
    sim.scalars    # grace_scalars_reader
    sim.gw         # grace_gw_data
    sim.detectors  # grace_detector_set

The class supports both **simpilot-managed** directories (with ``config/parfile/`` and ``restart_NNNN/`` subdirectories) and **flat layouts** where all output lives under a single directory.

HDF5 export and import
~~~~~~~~~~~~~~~~~~~~~~~~

All scalar and GW data can be exported to a single HDF5 file for archival or sharing:

.. code-block:: python

    sim.export_scalars()
    # writes <ppdir>/<simname>_scalars.h5

    # or with an explicit path:
    sim.export_scalars("/path/to/output.h5")

The HDF5 file stores the simulation name, detector metadata (radius, center, resolution), ADM mass and orbital frequency (if available), and all scalar/GW data organized as:

- ``reductions/{max,min,norm2,integral}/{varname}`` — scalar reductions
- ``em_energy`` — electromagnetic energy diagnostic
- ``mass_flux/{detector}/{type}`` — per-detector mass flux
- ``co_locations/{name}`` — compact object positions
- ``gw/{detector}/l{l}_m{m}`` — complex GW mode data
- ``detectors/{name}`` — detector attributes (radius, center, etc.)

Reconstructing from HDF5
~~~~~~~~~~~~~~~~~~~~~~~~~~

A simulation object can be reconstructed from an exported HDF5 file using the ``from_hdf5`` classmethod. This enables a workflow where data is exported on an HPC cluster and analyzed locally:

.. code-block:: python

    # On the cluster
    sim = grace_simulation("/scratch/bns_run")
    sim.export_scalars()

    # On your laptop (after downloading the .h5 file)
    sim = grace_simulation.from_hdf5("bns_run_scalars.h5")

    sim.scalars    # fully reconstructed
    sim.gw         # fully reconstructed, with Madm and omega0
    sim.detectors  # fully reconstructed with metadata

    # Volume/plane readers are not available (None)
    sim.xyz        # None
    sim.xy         # None

    # GW analysis works out of the box
    t, h = sim.gw.compute_strain("GW_1", (2, 2))
