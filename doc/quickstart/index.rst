.. _grace-quickstart:

Quickstart
============

This page walks you from a fresh clone to your first GRACE simulation. The path
shown here is the **simplest viable build** — a CPU-OpenMP backend with
default options — intended to get you running a smoke-test in under fifteen
minutes. For the full reference of build flags see :doc:`../code_building_guide/index`,
and for runtime parameters see :doc:`../userguide/index`.


Prerequisites
*************

GRACE has two dependency modes — pick the one that matches your environment.

System dependencies (always required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These come from your OS or HPC modules.

- **CMake** ≥ 3.22
- **C++20** compiler (GCC ≥ 12, Clang ≥ 16, or NVHPC ≥ 24.5 for CUDA builds)
- **MPI** implementation (OpenMPI, MPICH, Cray-MPICH, ...)
- **HDF5** with parallel support
- **libxml2**, **zlib**

The remaining five GRACE dependencies — **Kokkos**, **p4est**, **Catch2**,
**spdlog**, **yaml-cpp** — can be supplied two ways:

Mode A: bundled (recommended for laptop / first build / CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository with ``--recursive`` and configure with
``-DGRACE_USE_BUNDLED_DEPS=ON``.  GRACE will build all five from the in-tree
submodules under ``extern/`` using exactly the same compile flags as the
main build.  Self-contained, reproducible, zero version-mismatch issues.

.. code-block:: bash

    git clone --recursive https://github.com/GRACE-astro/grace.git grace-src
    cd grace-src
    cmake -B build -S . \
          -DGRACE_USE_BUNDLED_DEPS=ON \
          -DGRACE_ENABLE_OMP=ON \
          -DGRACE_METRIC_EVOL=COWLING
    cmake --build build -j

Mode B: system installs (alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leave ``GRACE_USE_BUNDLED_DEPS`` off (the default).  GRACE locates each
dependency through a ``<DEP>_ROOT`` environment variable, typically populated
by ``module load`` on cluster systems:

.. code-block:: bash

    export KOKKOS_ROOT=/path/to/kokkos/install
    export P4EST_ROOT=/path/to/p4est/install
    export SPDLOG_ROOT=/path/to/spdlog/install
    export YAML_ROOT=/path/to/yaml-cpp/install
    export CATCH2_ROOT=/path/to/catch2/install

The ``env/`` directory in the GRACE repository contains example environment
files for several systems; copying one as a starting point is the easiest
path.

.. warning::

   If you supply Kokkos from a system-wide installation, you must ensure
   it was built with **relocatable device code (RDC)** enabled for your
   GPU backend
   (``-DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON`` or the matching
   ``HIP`` / ``SYCL`` flag).  GRACE has device functions defined in ``.cpp``
   files that are called from kernels in other translation units, which
   only links when the active backend's RDC is on.  Many pre-built Kokkos
   modules on HPC systems ship with RDC off — check, or build your own.
   The bundled-deps build (Mode A) handles this automatically.


Build
*****

For the rest of this tutorial we follow **Mode A** (bundled dependencies)
— the shortest path to a working build.  After running the ``cmake -B
build ...`` command from the Mode A block above:

.. code-block:: bash

    cmake --build build -j

The configure step prints a **configuration summary** at the end listing
the resolved value of every flag plus the active backend.  The same
summary is also written to ``build/grace_config_summary.txt`` — archive
it alongside any simulation output for reproducibility.

If the build succeeds, the GRACE executable lives at ``build/grace``.

.. tip::

   Building for GPU? Swap the OpenMP flag for the matching backend:
   ``-DGRACE_ENABLE_CUDA=ON``, ``-DGRACE_ENABLE_HIP=ON``, etc.  In
   bundled-deps mode (Mode A) you also need to pass the corresponding
   Kokkos arch flag (e.g. ``-DKokkos_ARCH_HOPPER90=ON``); see the
   :doc:`building guide <../code_building_guide/index>` for the full
   architecture-flag list.


Run your first simulation
*************************

GRACE ships several ready-to-run parameter files under ``examples/``,
organized by problem class (``z4c/``, ``grmhd/``, ``cowling_grmhd/``,
``symmetry_audit/``).  A good first run is the Balsara MHD shocktube on
a fixed flat background.  Two sized variants are provided:

- ``shocktubes/balsara1_cpu_tutorial.yaml`` — small grid (10 cells per
  block, refinement level 0, narrow slab), runs in **about 17 s on 4
  CPU cores**; recommended for the first tutorial pass.
- ``shocktubes/balsara1.yaml`` — canonical resolution (``32³`` per block,
  level 1, 1.3 M cells); GPU-friendly, takes a few minutes on a CPU
  laptop.

For the tutorial run, use the small one:

.. code-block:: bash

    mkdir run-shocktube && cd run-shocktube
    cp ../examples/cowling_grmhd/shocktubes/balsara1_cpu_tutorial.yaml .

    # OpenMP backend: one MPI rank, threads driven by OMP_NUM_THREADS
    export OMP_NUM_THREADS=4
    mpirun -n 1 ../build/grace --grace-parfile ./balsara1_cpu_tutorial.yaml

If you need to make it faster still (or slower for better-resolved output),
the relevant knobs on a Balsara-style shocktube are:

- Lower ``npoints_block_x = npoints_block_y = npoints_block_z`` together
  (GRACE requires cubic blocks; e.g. set all three to ``8``).
- Narrow the domain extents **proportionally** along all axes so the
  tree size shrinks without spawning extra trees along the long axis;
  reduce ``final_time`` correspondingly.
- And / or drop ``initial_refinement_level``.

You should see GRACE print a banner, the configuration summary, then a stream
of evolution-step log lines. When the run finishes you'll have populated
output directories (defaults defined in ``parameters/IO.yaml``):
``./output_scalar`` for scalar diagnostics, ``./output_volume`` for 3D
volume data, ``./output_surface`` for 2D plane and sphere data, and
``./logs`` for runtime logs.


Inspect the output
******************

Scalar diagnostic files (``.dat``) are plain text and can be plotted directly.
For this shocktube setup, ``scalar_output_minmax: ["rho","press","eps","Bdiv"]``
produces one ``<var>_<reduction>.dat`` file per quantity per reduction:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   it, t, val = np.loadtxt("output_scalar/rho_maximum.dat",
                            skiprows=1, unpack=True)
   plt.plot(t, val)
   plt.xlabel("t")
   plt.ylabel(r"$\rho_{\max}$")
   plt.savefig("rho_max.png")

Volume and plane data are written as HDF5 with companion XDMF descriptors
that ParaView (and GRACEpy's reader) can open directly.  The descriptors
are NOT written by GRACE; they are generated from the HDF5 outputs by a
small companion utility shipped with GRACEpy.

**TL;DR — generate XDMF descriptors after the run**:

.. code-block:: bash

   # Install GRACEpy first (see :doc:`../python_interface/index`).
   # The `pip install` step exposes the `create_descriptor` console script.

   # This tutorial only writes plane data; one descriptor per orientation:
   create_descriptor --mode temporal --filter "*xy*" \
       ./output_surface ./output_surface/desc_xy

The ``--filter`` glob picks the HDF5 files that go into the descriptor
(here, only the ``xy``-plane snapshots), and ``--mode temporal`` writes
a time-series collection.  Once ``desc_xy.xmf`` exists, open it in ParaView
or load it with GRACEpy's reader as shown below.  See
:doc:`../python_interface/index` for the full GRACEpy install + usage
reference, including the ``yz`` / ``xz`` / volume / sphere variants for
runs that write those.

To plot a 1D slice of a plane field (e.g. :math:`\rho` along ``x`` on
the equatorial plane), GRACEpy's XDMF reader gives you the cell-centered
data + coordinates of the whole plane; masking by the smallest
:math:`|y|`, :math:`|z|` picks out the central row:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import grace_tools.vtk_reader_utils as gtv

   reader = gtv.grace_xmf_reader("./output_surface/desc_xy.xmf")
   t = reader.available_times()
   xyz, rho = reader.get_var("rho", t[-1])

   # pick the central cell-row in (y, z) on the xy plane
   y0 = np.amin(np.abs(xyz[:, 1]))
   z0 = np.amin(np.abs(xyz[:, 2]))
   tol = 1e-10
   mask = (np.abs(xyz[:, 1] - y0) < tol) & (np.abs(xyz[:, 2] - z0) < tol)

   plt.scatter(xyz[mask, 0], rho[mask], marker=".")
   plt.xlabel(r"$x$")
   plt.ylabel(r"$\rho$")
   plt.savefig("balsara1_rho.png")

The same pattern works for ``"press"``, ``"Bvec"``, etc. — see
:doc:`../userguide/index` for the variable conventions.


What to read next
*****************

- :doc:`../code_building_guide/index` — full reference of every CMake flag,
  including GPU backends, Z4c evolution, optional physics modules, initial-data
  libraries, and bit-exact-conservation switches.
- :doc:`../userguide/index` — runtime parameters, output formats, AMR setup,
  variable conventions, reflection-symmetry treatment.
- :doc:`../python_interface/index` — the GRACEpy companion package for
  initial-data preparation, codegen, postprocessing, and the ``simpilot`` CLI.
- :doc:`../gallery/index` — example simulations with parameter files and
  representative output.


Common pitfalls
***************

- **MPI rank binding.** On NUMA / multi-socket hosts GRACE benefits from
  explicit CPU binding; without it (e.g. ``MALLOC_ARENA_MAX=1`` plus missing
  pinning) performance can collapse by an order of magnitude. Most ``mpirun``
  / ``srun`` invocations need a ``--cpu-bind`` flag or equivalent.
- **Kokkos backend mismatch.** GRACE will refuse to build if the Kokkos
  installation it links to wasn't built with the device backend you selected
  via ``GRACE_ENABLE_*``. Check the Kokkos install's ``KokkosConfig.cmake``
  if you see a backend-related error.
- **Parameter typos.** YAML config errors are reported at startup with the
  exact offending key path. The schemas under ``parameters/*.yaml`` are the
  authoritative reference for every supported key.
