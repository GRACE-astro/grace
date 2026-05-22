.. _grace-building:

Building The Code
=====================

GRACE's build system is based on `CMake <https://cmake.org/>`__. This guide will provide a detailed explanation of all the configure flags that
can be passed to the build system to customize the code's behaviour, and assumes that you already installed all the code's dependencies.
If you haven't done that yet, please consult the :doc:`introduction <../introduction/index>`.
If you're unfamiliar with CMake, please consult their extensive documentation to learn more about this tool. For our purposes,
CMake is a build system that we use to generate Makefiles that can then be used to generate the code. GRACE's build process
is effectively divided into two parts: the configure step and the build step. The configure step is where all the build-time flags
controlling the compilers to be used, the flags to be passed to the compiler, as well as all the GRACE-specific flags that set the
compile-time environment are passed. The build step is simply a way to automatically call make on all the generated Makefiles.
We will now describe all the options that grace accepts as inputs during its configure step and how these influence the resulting
executable, as well as some relevant CMake specific flags that are especially relevant to GRACE.

GRACE flags come in two flavours:

- **Boolean options** (``ON`` / ``OFF``), declared in CMake with ``option(...)``. These enable or disable a feature.
- **String-valued selectors** (e.g. ``WENOZ``, ``HLL``, ``Z4``), declared in CMake with ``set(... CACHE STRING ...)``. These pick one flavour out of a small fixed set.

The way to specify either kind to CMake is the same:

.. code-block:: bash

    $ cmake -B build -S ./ -D<FLAG_NAME>=<FLAG_VALUE>

Environment File
************************************

The recommended way of building GRACE is to first create an environment file. Some examples can be found in the GRACEpy repository as well as the main GRACE repository under `env`.

In short, these files should load necessary modules and define environment variables pointing to the installation path of GRACE dependencies.

The CMake build system looks for dependencies in the location pointed to by the environment variable ``<DEP>_ROOT``, so for instance your environment file should contain ``KOKKOS_ROOT=/my/kokkos/installation``.

After creating and sourcing your environment file you're ready for the configure step.

Configure Options
************************************

What follows is a description of all the configure-time flags that can be passed to GRACE's CMake build system, divided into categories.

At the end of the configure step GRACE prints a configuration summary
("GRACE Configuration Summary") that lists the resolved value of every
flag below. The same summary is also written to
``${CMAKE_BINARY_DIR}/grace_config_summary.txt`` so you can archive it
alongside benchmark output or simulation runs.


Backend selection
***********************************

GRACE needs to be made aware of the backend which it is being used on. The Kokkos installation that is linked at compile time must also have this backend enabled, and must be built against the matching GPU architecture (e.g. ``-DKokkos_ARCH_HOPPER90=ON`` for CUDA on H100, ``-DKokkos_ARCH_AMD_GFX942=ON`` for HIP on MI300). The full list of architecture flags lives in the `Kokkos documentation <https://kokkos.org/kokkos-core-wiki/keywords.html#architecture-keywords>`__; in bundled-deps mode (``-DGRACE_USE_BUNDLED_DEPS=ON``) the same ``-DKokkos_ARCH_*`` flags pass straight through to the in-tree Kokkos build.

These options are mutually exclusive at runtime, and only one should be enabled at a time. OpenMP is always used for host-only operations.

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type
     - Description
   * - GRACE_ENABLE_CUDA
     - Boolean
     - Enable CUDA backend (NVIDIA GPUs).
   * - GRACE_ENABLE_HIP
     - Boolean
     - Enable HIP backend (AMD GPUs and APUs).
   * - GRACE_ENABLE_SYCL
     - Boolean
     - Enable SYCL backend (Intel GPUs; experimental).
   * - GRACE_ENABLE_OMP
     - Boolean
     - Enable host-only OpenMP parallelism.
   * - GRACE_ENABLE_SERIAL
     - Boolean
     - Enable serial execution of parallel loops (default fallback if no other backend is selected).


Physics-scheme selectors
***********************************

These are *string-valued* selectors. Each picks one flavour of a compile-time scheme.

.. list-table::
   :header-rows: 1
   :widths: 25 30 65

   * - CMake Parameter
     - Allowed Values
     - Description
   * - GRACE_METRIC_EVOL
     - ``COWLING`` | ``Z4`` (default ``COWLING``)
     - Metric-evolution scheme. ``COWLING`` freezes the background metric (no spacetime evolution); ``Z4`` enables Z4c evolution.
   * - GRACE_RIEMANN_SOLVER
     - ``HLL`` | ``ADV`` | ``LLF`` (default ``HLL``)
     - GRMHD Riemann solver. ``HLL`` is the 2-wave HLLE solver. ``LLF`` is local Lax-Friedrichs (Rusanov). ``ADV`` is an advanced-solver framework (HLLD for MHD with HLLE fallback; an HLLC pure-hydro branch is planned) — reserved for a follow-up release: selecting ``ADV`` compiles, but the kernel aborts at the first flux evaluation.
   * - GRACE_EMF_SCHEME
     - ``GS`` | ``UCT`` (default ``GS``)
     - CT EMF reconstruction scheme. ``GS`` is the Gardiner-Stone EMF reconstructed in-kernel from face fluxes during the directional flux sweep. ``UCT`` is upwind constrained transport with edge EMFs computed in a separate pass.
   * - GRACE_RECONSTRUCTION
     - ``WENOZ`` | ``MC2`` | ``MINMOD`` | ``DONOR_CELL`` (default ``WENOZ``)
     - Spatial reconstruction class for face-centred primitives. ``WENOZ`` is 5th-order WENO-Z (production default). ``MC2`` and ``MINMOD`` are 2nd-order slope-limited. ``DONOR_CELL`` is 1st-order piecewise-constant (intended for symmetry-preservation diagnostics, not production).
   * - GRACE_RECON_THERMO
     - ``TEMP`` | ``PRESS`` (default ``TEMP``)
     - Thermodynamic primitive reconstructed at faces. ``TEMP`` reconstructs temperature (GRACE convention, preserves cold-K exactly along polytropic equilibria). ``PRESS`` reconstructs pressure (continuous across contact discontinuities).
   * - GRACE_Z4C_DER_ORDER
     - ``2`` | ``4`` | ``6`` (default ``6``)
     - Truncation order of every Z4c RHS finite-difference operator (centered 1st-derivative, biased L1/R1, centered 2nd-derivative diagonal and mixed, upwind 1st, Kreiss-Oliger). Operators are auto-generated by the codegen pipeline with symmetry-equivariant grouping baked into the emitted C99 expressions. Extending beyond order 6 requires re-running the codegen with a larger ``ORDERS`` tuple.
   * - GRACE_MATTER_METRIC_DER_ORDER
     - ``2`` | ``4`` | ``6`` (defaults to ``GRACE_Z4C_DER_ORDER``)
     - FD order used by matter subsystems (GRMHD, M1) when sampling spatial derivatives of the metric for geometric source terms. Defaults to ``GRACE_Z4C_DER_ORDER`` so matter and metric live on the same dispersion curve at all wavenumbers. Mismatching the two introduces a coherent ringing artefact at NS surfaces in BNS configurations (the May-2026 wake-ripple); override only deliberately for diagnostic A/B comparisons. A configure-time warning is emitted if the two differ.


Numerical schemes (boolean)
***********************************

These are boolean compile-time switches that toggle a numerical scheme on or off.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - CMake Parameter
     - Description
   * - GRACE_ENABLE_FOFC
     - First-Order Flux Correction (default ``ON``). After each high-order flux sweep, GRACE flags cells whose tentative update would have required c2p flooring and recomputes the offending faces with donor-cell + LLF. Disable for symmetry-preservation diagnostics or when bisecting a flux-related bug.
   * - GRACE_SAME_LEVEL_FLUX_AVERAGE
     - Same-level face-flux averaging (default ``OFF``). After each substep's reflux at fine-coarse interfaces, also exchange and average face fluxes at same-level interior faces so both sides apply the same bit pattern. Combined with the existing fine-coarse reflux and deterministic MPI sums, this gives the bit-exact mass-conservation property ``total_mass(t) + Σ dM_outflux(t) ≡ M(0)``. Eliminates the linear-in-time FMA-roundoff drift that would otherwise accumulate. Costs are similar in scale to the existing reflux machinery — an extra send/recv per pair of same-level interior faces. Opt in for runs where bit-exact conservation matters; off for production performance.
   * - GRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS
     - Canonical-ordered MPI reductions (default ``OFF``). Replaces the implementation-defined tree reduction in ``mpi_allreduce`` (for ``SUM``, ``MAX``, ``MIN``) with an ``Allgather`` + local sum in ascending rank order. The result is bit-identical across all ranks AND reproducible across MPI partitionings (the same simulation at 1, 2, 8, ... ranks gives exactly the same diagnostic sums). Required for bit-exact mass conservation across partitions. Cost is one ``Allgather`` plus ``O(nproc)`` memory per intercepted reduction — negligible for the small-count diagnostic reductions it targets, but profile before enabling if a large-count ``Allreduce`` sits in a hot path.


Optional physics modules
***********************************

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type
     - Description
   * - GRACE_ENABLE_M1
     - Boolean
     - Enable M1 radiative transport (developer-only, work in progress).
   * - GRACE_ENABLE_PARTICLES
     - Boolean
     - Build the GRACE particle subsystem (tracers; future MC / PIC).
   * - GRACE_FREEZE_HYDRO
     - Boolean
     - Freeze hydrodynamics evolution (debug / metric-only runs).


Initial-data libraries
***********************************

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type
     - Description
   * - GRACE_ENABLE_LORENE
     - Boolean
     - Enable LORENE support.
   * - GRACE_ENABLE_FUKA
     - Boolean
     - Enable FUKA / Kadath support. Requires ``HOME_KADATH`` to point at a built Kadath tree.
   * - GRACE_ENABLE_TWO_PUNCTURES
     - Boolean
     - Enable TwoPunctures support.


Dependency-resolution mode
***********************************

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - CMake Parameter
     - Description
   * - GRACE_USE_BUNDLED_DEPS
     - Build **Kokkos**, **p4est**, **Catch2**, **spdlog**, **yaml-cpp** from in-tree git submodules under ``extern/`` (default ``OFF``).  When ON, the deps are pulled into the build as ``add_subdirectory`` targets and compiled with the same flags as GRACE itself — reproducible, no version-mismatch issues, and ``git clone --recursive`` plus ``cmake && make`` is enough to get a working build.  When OFF, GRACE searches for system installs via ``find_package`` (and the ``<DEP>_ROOT`` env-var convention), which is the recommended path on HPC clusters with arch-tuned ``module load`` paths for Kokkos.  See the :doc:`quickstart guide <../quickstart/index>` for the two-mode walkthrough.


Performance tuning
***********************************

GRACE selects a per-architecture performance-tuning header that defines kernel launch-bounds / tile macros. Selection happens automatically from ``Kokkos_ARCH_*``; pass ``GRACE_PERF_TUNING`` to override.

.. list-table::
   :header-rows: 1
   :widths: 25 35 65

   * - CMake Parameter
     - Allowed Values
     - Description
   * - GRACE_PERF_TUNING
     - empty | ``Mi300A`` | ``Mi250X`` | ``A100`` | ``H100`` | ``generic``
     - Manually select a tuning header. Leave empty to auto-detect from ``Kokkos_ARCH_*``.


Miscellaneous Options
***********************************

.. list-table::
   :header-rows: 1
   :widths: 25 25 75

   * - CMake Parameter
     - Type
     - Description
   * - GRACE_ENABLE_TESTING
     - Boolean
     - Build unit tests (default ``ON``).
   * - GRACE_ENABLE_BENCHMARKS
     - Boolean
     - Build the benchmark / profiling drivers.
   * - GRACE_BUILD_DOCS
     - Boolean
     - Build this documentation.
   * - GRACE_BUILD_DOCS_ONLY
     - Boolean
     - Only build this documentation (skip the main build).
   * - GRACE_NSPACEDIM
     - Integer
     - Number of spatial dimensions (only ``3`` is supported right now).
   * - GRACE_ENABLE_VTK
     - Boolean
     - Enable VTK output (legacy; HDF5 is the supported output format).
   * - GRACE_ENABLE_PROFILING
     - Boolean
     - Enable profiling hooks (HIP / ROCm only at the moment).
   * - GRACE_CARTESIAN_COORDINATES
     - Boolean
     - Use Cartesian coordinates (default).
   * - GRACE_SPHERICAL_COORDINATES
     - Boolean
     - Use spherical coordinates.
