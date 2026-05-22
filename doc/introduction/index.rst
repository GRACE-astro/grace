.. _grace-intro:

Introduction
============

GRACE (General Relativistic Astrophysics Code for Exascale) is a
Finite-Volume framework for solving hyperbolic partial differential
equations on adaptively-refined grids.  Its primary target is
general-relativistic magnetohydrodynamics (GRMHD) on dynamical curved
spacetimes — with binary neutron-star mergers as the headline application
— but the substrate (block-structured AMR via `p4est`_, performance-
portable kernels via `Kokkos`_) is general enough to host other
hyperbolic systems.

GRACE is designed to run efficiently on modern GPU architectures (CUDA,
HIP) and on multi-core CPUs (OpenMP), from a laptop to exascale clusters.


Where to start
**************

- The :doc:`../quickstart/index` walks you from a fresh clone to a first
  simulation in under fifteen minutes.
- The :doc:`../code_building_guide/index` is the build-time reference:
  every CMake flag plus per-library dependency setup.
- The :doc:`../userguide/index` is the runtime reference: parameter
  schemas, variable conventions, grid model, output formats.


Licensing
*********

GRACE is distributed under the `GNU General Public License v3.0
<https://www.gnu.org/licenses/gpl-3.0.html>`_ (or, at your option, any
later version).  The full license text ships in ``LICENSE.md``;
third-party components (Kokkos, p4est, Catch2, spdlog, yaml-cpp) carry
their own upstream licenses, summarised in ``THIRD_PARTY_LICENSES.md``.
Authors and contributors are listed in ``AUTHORS``; see
:doc:`../citing/index` for the citation policy.

.. _Kokkos: https://github.com/kokkos/kokkos
.. _p4est: https://www.p4est.org/
