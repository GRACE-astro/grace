.. _grace-devguide:

GRACE Developer Guide
=====================

This guide is for developers who need to understand or extend GRACE
beyond what is covered in the :doc:`../userguide/index`.  It documents
the design principles the code was built on, the lower-level
infrastructure those principles motivated, and the gotchas that have
accumulated as the code has matured.  Familiarity with modern C++
(C++20), template metaprogramming, and the basics of Kokkos and MPI
is assumed; expertise is not.

The guide is organised as a series of self-contained chapters, each
focused on one subsystem.  Read it in order on first contact;
otherwise treat the index as a reference.


Design principles
-----------------

GRACE's architecture rests on a small number of explicit choices
that recur across subsystems.  Knowing them up front makes individual
modules considerably easier to read, because most of what looks like
ceremony is one of these principles being enforced.


Performance portability over premature abstraction
**************************************************

Every compute kernel in GRACE is written as a Kokkos parallel
construct operating on Kokkos ``View``\ s.  This is a deliberate
constraint: it forces the same source to run on CPU (OpenMP), CUDA,
and HIP without per-backend specialisations, and it keeps the kernel
boundary syntactically explicit, which helps profiling and debugging.
The cost is that GRACE leans on Kokkos idioms (lambdas,
``MDRangePolicy``, ``TeamPolicy``, scratch memory) rather than
wrapping them in a "physics-friendly" abstraction layer that would
have to be re-tuned each time the underlying portability layer
changes.

Abstraction in GRACE is added when it earns its place by removing
duplicated logic, not in anticipation of future requirements.


Single source of truth
**********************

Configuration, physics, and symbolic algebra each flow from a single
authoritative source:

- **Runtime configuration**: every parameter is declared in a YAML
  schema (see :ref:`grace-config-parser`).  The schema provides
  defaults, ranges, and types at startup; user YAML overrides only
  what it explicitly sets.  No parameter is looked up from more than
  one place.
- **Build-time codegen**: physics RHS expressions (Z4c, GRMHD, M1
  closures) are emitted by SymPy notebooks into header files of the
  form ``*_subexpressions.hh``.  These headers are version-controlled
  but never edited by hand; the notebook is the source.
- **Equation of state**: the EOS instance is the only source of
  thermodynamics.  No subsystem maintains a parallel fit.


Composable physics modules
**************************

Physics is organised as a set of "evolution systems" — GRMHD, Z4c,
M1 radiation transport, particles — each a class that declares its
evolved variables, its RHS, and any auxiliary quantities it owns.
The top-level driver composes the modules at runtime: a TOV run uses
GRMHD + Z4c, a BNS run adds magnetic constrained transport and
optionally M1, a pure spacetime test uses Z4c alone.  Modules do not
directly reference each other; they communicate through the shared
variable list and through explicit coupling points (e.g. the M1 →
GRMHD matter-source feedback).


Determinism and symmetry as first-class concerns
************************************************

GRACE goes to considerable lengths to keep simulations
bit-reproducible under MPI repartition and discrete symmetry (e.g.
mirror reflection) when the input data and operators are themselves
equivariant.  This imposes constraints across the code:

- Reductions and accumulations whose result depends on rank-ordering
  are avoided in the evolution path or replaced with
  ``MPI_Allreduce``-based associative variants.
- Inner sums in stencils, interpolators, and source terms are
  written with explicit pair-symmetric brackets where required.
- Floating-point-noise-sensitive paths (the GRMHD wavespeed
  computation, the C2P inverter, the Z4c constraint propagators) are
  audited against discrete symmetries.

The result is a code in which divergence from symmetry can almost
always be traced to a discrete physical event (refinement,
atmosphere reset, EOS branch) rather than to a numerical artefact.


Explicit lifetimes for global resources
***************************************

Several pieces of state in GRACE are unavoidably global: the MPI
communicator, the Kokkos runtime, the parameter store, the p4est
forest, the EOS instance, the variable list, the coordinate system.
Treating these as ad-hoc singletons would create initialisation- and
destruction-order bugs that are notoriously hard to track down,
especially across MPI finalisation and Kokkos shutdown.

Instead GRACE uses a single, policy-based singleton template — the
``singleton_holder`` — that gives each globally-unique object an
explicit *longevity* in a controlled destruction order and routes
its creation through a swappable allocation policy.  This is the
subject of the next chapter.


Contents
--------

.. toctree::
   :maxdepth: 1

   singleton_holder
   config_parser
   p4est_wrapper
   variable_layout
   ghost_zones
   ghost_exchange_pipeline
   refluxing
   codegen_pipeline
   io_and_diagnostics
   amr_regrid
   checkpointing
   unit_testing
