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
=====================================

These come from your OS or HPC modules.

- **CMake** ≥ 3.22
- **C++20** compiler (GCC ≥ 12, Clang ≥ 16, or NVHPC ≥ 24.5 for CUDA builds)
- **MPI** implementation (OpenMPI, MPICH, Cray-MPICH, ...)
- **HDF5** with parallel support
- **libxml2**, **zlib**

The remaining five GRACE dependencies — **Kokkos**, **p4est**, **Catch2**,
**spdlog**, **yaml-cpp** — can be supplied two ways:

Mode A: bundled (recommended for laptop / first build / CI)
============================================================

Clone the repository with ``--recursive`` and configure with
``-DGRACE_USE_BUNDLED_DEPS=ON``.  GRACE will build all five from the in-tree
submodules under ``extern/`` using exactly the same compile flags as the
main build.  Self-contained, reproducible, zero version-mismatch issues.

.. code-block:: bash

    git clone --recurse-submodules --shallow-submodules \
              https://github.com/Carlo-Musolino/grace.git grace-src
    cd grace-src
    cmake -B build -S . \
          -DGRACE_USE_BUNDLED_DEPS=ON \
          -DGRACE_ENABLE_OMP=ON \
          -DGRACE_METRIC_EVOL=COWLING
    cmake --build build -j

The ``--shallow-submodules`` flag fetches Kokkos, p4est, and the other
in-tree dependencies at depth 1, which on a slow link is the difference
between a few-minute clone and a multi-hour one.  Use plain ``--recursive``
instead if you need full submodule history (e.g. for development against
a specific upstream commit).  If you forgot to clone with submodules, run
``git submodule update --init --recursive --depth 1`` afterwards.

Mode B: system installs (recommended for HPC clusters)
======================================================

Leave ``GRACE_USE_BUNDLED_DEPS`` off (the default).  GRACE locates each
dependency through a ``<DEP>_ROOT`` environment variable, typically populated
by ``module load`` on cluster systems:

.. code-block:: bash

    export KOKKOS_ROOT=/path/to/kokkos/install
    export P4EST_ROOT=/path/to/p4est/install
    export SPDLOG_ROOT=/path/to/spdlog/install
    export YAML_ROOT=/path/to/yaml-cpp/install
    export CATCH2_ROOT=/path/to/catch2/install

This mode is preferred on clusters because the locally-installed Kokkos is
typically tuned with the right CUDA/HIP arch flags + register-pressure
heuristics, which the bundled build cannot match.  The ``env/`` directory
in the GRACE repository contains example environment files for several
systems; copying one as a starting point is the easiest path.


Clone and build
***************

.. code-block:: bash

    git clone --recursive https://github.com/Carlo-Musolino/grace.git grace-src
    cd grace-src

    # source your environment file (sets KOKKOS_ROOT, P4EST_ROOT, etc.)
    source env/my-laptop.env

    # configure: CPU OpenMP backend, Cowling metric (fixed background)
    cmake -B build -S . \
          -DCMAKE_BUILD_TYPE=Release \
          -DGRACE_ENABLE_OMP=ON \
          -DGRACE_METRIC_EVOL=COWLING

    # build
    cmake --build build -j

The configure step prints a **configuration summary** at the end listing the
resolved value of every flag plus the active backend. The same summary is
written to ``build/grace_config_summary.txt`` — archive it alongside any
simulation output for reproducibility.

If the build succeeds, the GRACE executable lives at ``build/grace``.

.. tip::

   Building for GPU? Swap the OpenMP flag for the matching backend:
   ``-DGRACE_ENABLE_CUDA=ON``, ``-DGRACE_ENABLE_HIP=ON``, etc. Make sure your
   Kokkos installation was built with the same device backend.


Run your first simulation
*************************

GRACE ships several ready-to-run parameter files under ``examples/``,
organized by problem class (``z4c/``, ``grmhd/``, ``cowling_grmhd/``,
``symmetry_audit/``).  The simplest is a 1D relativistic shocktube on a
Cowling background — runs to completion in seconds on a laptop CPU:

.. code-block:: bash

    mkdir run-shocktube && cd run-shocktube
    cp ../examples/cowling_grmhd/shocktubes/balsara1.yaml .

    mpirun -n 2 ../build/grace --grace-parfile ./balsara1.yaml

You should see GRACE print a banner, the configuration summary, then a stream
of evolution-step log lines. When the run finishes you'll have a populated
output directory (default: ``./output_scalar`` for scalar diagnostics and
``./output_3d``, ``./output_xy``, ... for volume and plane data).


Inspect the output
******************

Scalar diagnostic files (``.dat``) are plain text and can be plotted directly:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   it, t, val = np.loadtxt("output_scalar/dens_integral.dat",
                            skiprows=1, unpack=True)
   plt.plot(t, val)
   plt.xlabel("t")
   plt.ylabel("∫ dens dV")
   plt.savefig("dens_integral.png")

Volume and plane data are written as HDF5 with companion XDMF descriptors
that ParaView can open directly. To generate the descriptors, use the helper
shipped with GRACEpy (see :doc:`../python_interface/index`).


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
