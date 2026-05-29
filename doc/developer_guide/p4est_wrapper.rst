.. _grace-p4est-wrapper:

The p4est wrapper: forest, tree, quadrant, connectivity
=======================================================

GRACE's AMR substrate is the `p4est <https://p4est.github.io/>`_
library [Burstedde2011]_, a parallel forest of octrees implementation
written in C.  p4est handles the topology of the refinement hierarchy
— which leaves exist on which rank, who their neighbours are, how to
refine or coarsen, how to repartition — and exposes a low-level C API
to do all of it.  GRACE wraps the slice of that API it actually uses
in a thin C++ layer that gives the rest of the code something it can
reason about by reference rather than by pointer.

This chapter introduces p4est's vocabulary, describes the GRACE
wrapper classes around it, and lists the subset of the p4est API
that the wrapper exposes.  It does **not** describe how GRACE stores
field variables on top of the topology — the
``(i, j, k, var, q)`` Kokkos ``View`` layout, the
cells-per-quadrant convention, physical coordinates — that is the
subject of the next chapter.  Here we only describe the topological
skeleton.


p4est vocabulary
----------------

Four words do most of the work.  Reading any AMR code with p4est in
it is a great deal easier once they are pinned down:

**Connectivity** (``p4est_connectivity_t``)
   The coarsest-level topology: a graph of *root cubes*, each
   identified by an integer tree index, glued together along faces,
   edges, and corners.  The connectivity carries the vertex
   coordinates of each root cube in *physical* space, and the
   periodicity flags.  It is static — it never changes after the
   forest is built.

**Forest** (``p4est_t``)
   The full refined mesh.  A forest consists of one *tree* per root
   cube in the connectivity, plus a partition of the leaf quadrants
   across MPI ranks.  Refinement, coarsening, repartition, and ghost
   construction all act on the forest.

**Tree** (``p4est_tree_t``)
   The refinement structure rooted at a single root cube.  Each tree
   is an octree (3D) or quadtree (2D), with the root cube at level 0
   and each internal node refined into 8 (or 4) children at the next
   level.  The leaves of a tree are the *quadrants* that actually
   carry data.

**Quadrant** (``p4est_quadrant_t``)
   A leaf of a tree, identified by its tree, its level, and its
   integer logical coordinates within the tree.  This is the unit at
   which GRACE allocates field variables: each leaf quadrant owns a
   block of cells (see the variable-layout chapter for the details).

There is a small but important detail in the vocabulary that
recurrent confusion turns on: **a quadrant is not a cell**.  It is a
block of cells (with ghostzones!).  Anyone arriving from an AMR code where the AMR unit
*is* the cell will need to keep this distinction in mind.  Most of
the GRACE wrapper, and almost all of the user-facing kernels, work
at the quadrant level.


The GRACE wrapper classes
-------------------------

The wrapper layer lives in ``include/grace/amr/`` and follows two
consistent naming conventions:

- **Owning singletons** end in ``_impl_t``: they hold a p4est handle,
  manage its lifetime, and are accessed through the
  :ref:`grace-singleton-holder`.
- **Non-owning views** end in ``_t``: they wrap a pointer to a p4est
  struct without taking ownership, and exist for the duration of an
  enclosing operation.

The classes are:

``connectivity_impl_t``  (``include/grace/amr/connectivity.hh``)
   Singleton wrapper around ``p4est_connectivity_t*``.  Owns the
   topology — vertex coordinates, tree-to-tree face graph, periodicity
   flags — and exposes methods such as ``tree_coordinate_extents()``
   to recover a tree's physical bounding box, ``tree_to_tree()`` to
   walk the face-neighbour graph, and ``vertex_coordinates()`` for the
   raw vertex positions.  The instance is built once during startup and lives 
   for the duration of the run.

``forest_impl_t``  (``include/grace/amr/forest.hh``)
   Singleton wrapper around ``p4est_t*``.  Owns the forest itself and
   exposes the trees on this rank: ``first_local_tree()``,
   ``last_local_tree()``, ``tree(which_tree)``,
   ``local_num_quadrants()``, and the rank-to-quadrant offset map
   used to translate between global and local quadrant indices.  The
   forest can be rebuilt during regrid (see the AMR chapter for
   that pipeline); its singleton identity is preserved.

``tree_t``  (``include/grace/amr/tree.hh``)
   Non-owning view on a ``p4est_tree_t*``.  Yields the quadrants in
   the tree via ``quadrants()``, individual quadrants by index via
   ``quadrant(iquad)``, and the tree-local-to-global quadrant offset
   via ``quadrants_offset()``.

``quadrant_t``  (``include/grace/amr/quadrant.hh``)
   Non-owning view on a ``p4est_quadrant_t*``.  Exposes the
   essentials needed everywhere downstream:

   - ``level()`` — the refinement level (0 = root).
   - ``qcoords()`` — integer logical coordinates within the tree.
     With the default argument these are expressed at the current
     level (so they range from 0 to ``2^level - 1``); pass ``false``
     to get them at p4est's internal ``P4EST_MAXLEVEL`` resolution.
   - ``spacing()`` — the *logical* edge length of the quadrant
     (``1.0 / 2^level``), measured in units of the parent tree's
     logical extent.  To convert to physical length, multiply by the
     tree's physical extent obtained from the connectivity.
   - ``tree_index()`` — which tree this quadrant belongs to.
   - ``linearid(level)`` — the Morton index at a given level, useful
     as a stable identifier across repartitioning.

The non-owning view classes are cheap to construct, but they go
stale the moment p4est mutates the underlying structures (a regrid,
a partition, a refine/coarsen pass).  They are short-lived by design
— construct, use, discard.


Connectivities GRACE supports
-----------------------------

p4est can describe arbitrary topologies, but GRACE uses a small set
of them.  The shipping connectivities are built in
``include/grace/amr/connectivities_impl.hh`` and
``src/amr/connectivity.cpp`` via two p4est calls:

- **Brick** (``p4est_connectivity_new_brick``).  An ``nx × ny × nz``
  block of unit-cube trees stacked face-to-face, with optional
  periodicity along each axis.  This is the standard layout for BNS,
  TOV, single black-hole, and shocktube runs.  Tree ``itree`` lies
  at ``(itree % nx, (itree / nx) % ny, itree / (nx * ny))`` in the
  brick, and its physical extents come from the connectivity vertex
  table.  The brick is what almost every GRACE simulation uses.

- **Custom** (``p4est_connectivity_new_copy``).  For setups that do
  not fit a brick — e.g. non-rectangular domains — GRACE assembles
  the vertex and tree-to-vertex tables directly and hands them to
  the p4est copy constructor.  These are rare in production.

Other p4est-supplied connectivities (sphere of cubes, cubed sphere,
double-shell, etc.) are not currently wired up.  Several downstream
subsystems — ghost descriptors, constrained-transport refluxing,
prolongation kernels — assume brick-like adjacency in places, so
adding a non-brick connectivity requires auditing those code paths
first.


The p4est API subset GRACE uses
-------------------------------

p4est is a substantial library.  GRACE uses a deliberately narrow
slice of it, partly to keep the wrapper small, partly to avoid
exposing call patterns that would conflict with the
deterministic-evolution guarantees discussed in the design
principles.  The slice covers four areas.

**Connectivity construction**
   ``p4est_connectivity_new_brick``,
   ``p4est_connectivity_new_copy``,
   ``p4est_connectivity_is_valid``,
   ``p4est_connectivity_save``.

**Forest construction and lifecycle**
   ``p4est_new`` / ``p4est_new_ext`` for initial forest creation,
   ``p4est_destroy`` at finalisation, ``p4est_reset_data`` if user
   data on quadrants needs reinitialising.

**Traversal and inspection**
   The ``first_local_tree`` / ``last_local_tree`` /
   ``local_num_quadrants`` fields on ``p4est_t``, the
   ``quadrants`` array on ``p4est_tree_t``, and the
   ``x`` / ``y`` / ``z`` / ``level`` / ``p.which_tree`` fields on
   ``p4est_quadrant_t``.  Plus
   ``p4est_quadrant_linear_id`` for stable Morton ids.

**Refinement, coarsening, balance, partition, ghost**
   ``p4est_refine_ext``, ``p4est_coarsen_ext``,
   ``p4est_balance_ext``, ``p4est_partition``,
   ``p4est_ghost_new``, ``p4est_ghost_destroy``.
   These are exercised only by the regrid pipeline (see the AMR
   chapter), never directly by user-level evolution code.

**Iteration and search**
   ``p4est_iterate`` is used during ghost-descriptor construction to
   walk face / edge / corner neighbour relations between local
   quadrants and the ghost layer.  ``p4est_search`` is used by the
   particle subsystem (to locate the owning quadrant of a tracer)
   and by some output paths that need to map a physical point to a
   quadrant.  Neither call ever appears in the per-step evolution
   path; both are restricted to setup and to non-hot infrastructure.

**Forest serialisation**
   ``p4est_save_ext`` and ``p4est_load_ext`` are used by the
   checkpoint handler to persist and restore the forest topology
   (tree structure, leaf-quadrant pattern, FMR-floor user data).
   This is the *only* part of GRACE's checkpoint that goes through
   the p4est API; the field data is written separately through a
   custom HDF5 path with the co-tracker pattern (see the
   checkpoint chapter for the details).  The split matters because
   the GRACE checkpoint is **rank-agnostic**: a run saved on
   *N* ranks can be restored on *M* ranks unchanged, because both
   ``p4est_load_ext`` and the field-data reader treat the partition
   as an output of the load rather than an input.  Production runs
   exercise this regularly when re-balancing across queue
   allocations.

Notably **not** used: p4est's own MPI helpers (GRACE drives MPI
directly through its own wrappers — see :ref:`grace-singleton-holder`
for the runtime singleton), p4est's connectivity-from-file readers,
p4est's geometry callback API, and many smaller utilities in the
library that GRACE simply has no use for.  Treat the list above as
the *complete* p4est surface area GRACE depends on; if you catch
yourself reaching for a different p4est function, check first
whether the wrapper already exposes the same information another
way.


What GRACE stores in the quadrant's user data
----------------------------------------------

Each ``p4est_quadrant_t`` carries a small *user-data* slot reserved
for the application.  The slot is presented by p4est as a union of
``user_int``, ``user_long``, and ``user_data`` — but they are the
**same memory**, interpreted differently.  GRACE uses the
``user_data`` member of the union; the ``user_int`` and
``user_long`` members must therefore be treated as off-limits.
Writing to either one of them corrupts the ``user_data`` GRACE
relies on.

Concretely, GRACE stores two small pieces of information per
quadrant:

- **A regrid flag** indicating whether the quadrant should be
  refined, coarsened, or left alone at the next regrid call.  It is
  written by the refinement-criteria pass and consumed by
  ``p4est_refine_ext`` / ``p4est_coarsen_ext`` through their
  callback interface.
- **The FMR minimum allowed level**, which prevents a quadrant
  inside a user-declared fixed-mesh-refinement region from
  coarsening below the level the user requested for that region.

That is the full extent of the per-quadrant user-data payload.
Everything else — field variables, ghost layouts, descriptors,
EMF / flux registers — lives in Kokkos ``View``\ s indexed by a
flat quadrant index ``q``, not in the p4est struct.

**This memory must not be touched outside the regrid pipeline.**
Writing to ``user_data``, ``user_int``, or ``user_long`` from
evolution-side code will corrupt the regrid flags and the FMR
floors, with consequences ranging from silent loss of refinement
to corrupted forest invariants on the next regrid call.  The few
operations that legitimately modify these values all live under
``include/grace/amr/regrid/`` and ``src/amr/regrid/`` and go through
named helpers; if you find yourself wanting to set a quadrant flag
from somewhere else, the right move is to add a helper alongside
those, not to reach into ``user_data`` directly.


Where the wrapper draws its lines
---------------------------------

A few principles are enforced consistently in the wrapper and worth
naming, because they recur whenever a new contributor wants to "just
do X with p4est":

- **Mutation of the forest is centralised**.  Refine, coarsen,
  balance, and partition calls live only in the regrid pipeline.
  Per-step evolution code is read-only against the forest.  This is
  what makes per-step bit-determinism tractable: the topology can
  change only at well-defined points in the timestep.
- **Field data is owned by Kokkos, not by p4est**.  GRACE does not
  use p4est's per-quadrant ``user_data`` slot for evolved variables;
  storing field arrays there would defeat performance portability.
  The mapping from quadrant to storage is a flat integer index ``q``
  documented in the variable layout chapter.  The ``user_data`` slot
  itself is reserved by GRACE for regrid flags and FMR floors only
  (see above); the ``user_int`` and ``user_long`` aliases of the
  same memory are off-limits.
- **Connectivity is a single global object**.  All p4est calls that
  take a ``p4est_connectivity_t *`` use the connectivity singleton.
  There is no support for, or test coverage of, multiple forests
  over different connectivities in the same run.


Pitfalls
--------

- **A quadrant is not a cell**.  See the introduction.  This is the
  single most common source of off-by-orders-of-magnitude bugs in
  AMR code.
- **Logical vs physical extents**.  ``quadrant_t::spacing()`` and
  ``qcoords()`` are in *logical* units (the parent tree spans the
  unit cube).  Physical sizes require multiplying by the tree's
  physical extent, queried from the connectivity through
  ``tree_coordinate_extents(itree)``.  See the variable layout
  chapter for the helpers that combine these.
- **Non-cubic bricks**.  When the brick has unequal physical extents
  across axes (e.g. a thin slab), each tree is still a unit cube in
  logical space, but its physical extents differ per axis.  Any
  code that assumes isotropic ``dx`` will break silently; always
  compute the physical spacing per axis through the connectivity.
- **Non-power-of-2** ``npoints_block``.  GRACE issues a warning at
  connectivity construction when ``npoints_block`` is not a power of
  two; the discrete cell-centre coordinates then lose the floating-
  point symmetry properties that several diagnostics rely on.
  Production runs should use power-of-2 block sizes; if you need an
  odd block for a specific test, the warning is informational, not
  blocking.
- **Stale non-owning views after regrid**.  Any ``tree_t`` or
  ``quadrant_t`` held across a regrid call points into invalid
  memory.  Always re-acquire views after operations that could
  rebuild the forest.
- **Mutating quadrants directly**.  Setting ``x`` / ``y`` / ``z`` /
  ``level`` on a ``p4est_quadrant_t`` outside the regrid pipeline
  will corrupt the forest invariants p4est relies on for
  neighbour-finding and Morton ordering.  Refinement must go
  through ``p4est_refine_ext`` and friends; coarsening through
  ``p4est_coarsen_ext``.
- **The ``user_*`` union aliases**.  ``p.user_data``,
  ``p.user_int``, and ``p.user_long`` are three names for the same
  memory.  GRACE writes through ``user_data`` only (regrid flags +
  FMR floors).  Writing through ``user_int`` or ``user_long``
  anywhere in the code will silently overwrite what the regrid
  pipeline put there, with effects ranging from missed refinement
  to broken forest invariants.


.. [Burstedde2011] C. Burstedde, L. C. Wilcox, and O. Ghattas,
   *p4est: Scalable Algorithms for Parallel Adaptive Mesh
   Refinement on Forests of Octrees*, SIAM Journal on Scientific
   Computing 33 (3), 1103–1133 (2011).
