.. _grace-ghost-exchange-pipeline:

The ghost-exchange pipeline
===========================

The previous chapter described the *static* side of ghost exchange:
descriptors, storage layout, BC kinds.  This chapter describes the
*operational* side: how that descriptor graph is compiled into a
task graph at regrid time, how that task graph is replayed every
substep, how the kernels actually execute, and how the subsystem
composes with constrained-transport refluxing.

The discussion follows the lifecycle in order:

- a thin driver (``apply_boundary_conditions``) that runs the
  pre-compiled task graph;
- the build pipeline that creates the task graph after every
  regrid (six phases per staggering);
- the task graph itself — the kinds of tasks, the four GPU streams,
  the dependency structure;
- the executor that drives the graph (ready queue, polling, atomic
  dependency release);
- the pack / unpack and prolongation / restriction kernels;
- why a *second* physical-BC pass exists (``deferred_phys_bc``);
- determinism guarantees and the few sharp edges.

CT refluxing — the related machinery that maintains ``∇·B = 0``
at coarse–fine interfaces — is structurally similar in flavour
(descriptors built at regrid, buffer construction patterns
analogous to those described here) but is a separate subsystem:
it maintains its own descriptors, is *not* task-based — it runs
imperatively as part of the evolution step — and does not share
the executor or buffer storage with ghost exchange.  It is
documented in its own chapter.


The driver
----------

The per-substep entry point is

.. code-block:: cpp

   namespace grace::amr {
     void apply_boundary_conditions() ;
     void apply_boundary_conditions(
       grace::var_array_t& vars,
       grace::staggered_variable_arrays_t& stag_vars,
       grace::var_array_t& vars_p,
       grace::staggered_variable_arrays_t& stag_vars_p,
       double dt, double dtfact
     ) ;
   }

(``src/amr/boundary_conditions.cpp``).  Both overloads do the same
thing: acquire the singleton ghost manager, package the four
relevant variable Views and the substep ``dt`` / ``dtfact`` into a
``view_alias_t`` bundle, run the pre-compiled task graph through
the executor, fence, and barrier:

.. code-block:: cpp

   auto& ghost         = grace::amr_ghosts::get() ;
   auto& halo_executor = ghost.get_task_executor() ;
   halo_executor.run(view_alias_t{&vars, &vars_p, &stag_vars, &stag_vars_p, dt, dtfact}) ;
   halo_executor.reset() ;
   Kokkos::fence() ;
   parallel::mpi_barrier() ;

The driver does **no** descriptor walking, no decision-making, no
dispatch.  All of that work has already been compiled into the
graph at regrid time.  This is the central performance principle
of the subsystem: per-substep cost is bounded by graph execution,
not by graph construction.


The build pipeline (per regrid)
-------------------------------

After every regrid, ``amr_ghosts_impl_t::update()`` (in
``src/amr/amr_ghosts.cpp``) rebuilds the task graph.  The build is
split per staggering: ``build_task_list<STAG_CENTER>`` for the
cell-centred fields, then ``build_task_list_face_stag`` three more
times for ``STAG_FACEX``, ``STAG_FACEY``, ``STAG_FACEZ``.  Each
call follows the same six-phase template:

**Phase 1 — MPI task creation.**  Pre-create one ``mpi_task_t``
per non-zero send size and one per non-zero receive size, for
every remote rank.  Each MPI task encapsulates an ``MPI_Isend`` or
``MPI_Irecv`` and an ``MPI_Request`` to be polled.  The tasks are
added to ``task_list`` with the right communicator tag
(``GRACE_HALO_EXCHANGE_TAG_CC`` for cell-centred, distinct tags
per face-staggered direction).  Send / receive task ids are
recorded by rank for later dependency wiring.

**Phase 2 — Stream allocation.**  Four GPU streams are pulled from
``device_stream_pool``: ``copy_stream``, ``pup_stream``,
``phys_bc_stream``, ``interp_stream``.  Pack / unpack, copies,
physical BCs, and prolongation each run on their own stream, so
they can overlap on backends that support concurrent kernels.

**Phase 3 — Restriction tasks** (``insert_restriction_tasks``).
For every quadrant whose ghost layer requires restricted data
from finer neighbours, a restriction kernel is enqueued on
``interp_stream``.  This produces the coarse buffer that
prolongation later reads.  ``restrict_tid`` is set to the task id
of the last restriction task so downstream copies and packs can
depend on it.

**Phase 4 — Copy tasks** (``insert_copy_tasks``).  Same-level,
same-rank copies fill ghost zones directly from neighbour
interiors.  Three kernel variants run here: plain
quadrant-to-quadrant copies, coarse-buffer-peer copies (when the
sibling lives in the coarse buffer), and copies from / to the
coarse buffer.  All run on ``copy_stream`` with dependencies on
``restrict_tid`` where coarse data is involved.

**Phase 5 — Pack / unpack tasks** (``insert_pup_tasks``).  Pack
kernels read interior cells of local quadrants and write into the
send buffer; their task ids are wired as dependencies of the
corresponding MPI send tasks.  Unpack kernels do the reverse:
they take MPI receive task ids as dependencies, read from the
receive buffer, and write into ghost zones.  Multiple pack /
unpack variants exist:

- plain ``pack_kernels`` / ``unpack_kernels`` for same-level
  exchanges;
- ``cbuf_p2p_pack_kernels`` / ``cbuf_p2p_unpack_kernels`` for
  coarse-buffer-to-peer routes;
- ``pack_finer_kernels`` for the finer-side packing in
  restriction-bound routes;
- ``pack_to_cbuf_kernels`` / ``unpack_to_cbuf_kernels`` /
  ``unpack_from_cbuf_kernels`` for the coarse-buffer plumbing.

All pack / unpack tasks run on ``pup_stream``.

**Phase 6 — Physical BCs and prolongation.**  Three sub-phases:

1. ``insert_phys_bc_tasks`` schedules the first BC pass on
   ``phys_bc_stream`` and returns the subset of BC kernels that
   must run *again* after prolongation (the "deferred" set).
2. ``insert_prolongation_tasks`` schedules the prolongation
   kernels on ``interp_stream``.  These read from the coarse
   buffer (filled by restriction in phase 3 and by MPI / cbuf
   plumbing in phase 5) and write into fine ghost zones at
   coarse–fine interfaces.
3. ``insert_deferred_phys_bc_tasks`` schedules the second BC pass
   on ``phys_bc_stream``, which re-fills the ghost cells where a
   physical boundary intersects an AMR coarse–fine interface (see
   "Why a deferred BC pass exists" below).

This template runs once per staggering.  STAG_CENTER uses
plain Lagrange prolongation; the three face staggerings use
div-preserving prolongation.  The four resulting task graphs are
concatenated into a single ``task_list`` and exposed through a
single executor.


The task graph
--------------

A *task* is a unit of work plus a dependency record.  The base
type ``task_t`` (``include/grace/utils/task_queue.hh``) carries:

- a ``kind ∈ {GPU_KERNEL, MPI_TRANSFER, CPU_EXEC}``;
- a list of *dependencies* — task ids that must complete before
  this task may run;
- a list of *dependents* — task ids whose pending count this task
  decrements when it completes;
- a ``task_id`` used to address it from the executor.

Three concrete derived types specialise the execution path:

- ``gpu_task_t`` wraps a Kokkos kernel launched on a specific
  device stream.  Completion is detected by polling a CUDA / SYCL
  / HIP event with ``dev_event.query()``.
- ``mpi_task_t`` wraps an ``MPI_Isend`` or ``MPI_Irecv`` plus its
  ``MPI_Request``.  Completion is detected by polling with
  ``MPI_Test``.
- ``cpu_task_t`` wraps a host-side callable.  Completion is
  synchronous with ``run()``.

The dependency edges are wired explicitly during build.  Pulling
out the structurally important patterns:

**MPI tasks are one per rank pair, per staggering.**  For each
remote peer rank ``r`` with non-zero traffic to this rank, the
build emits exactly one ``MPI_Isend`` task and one ``MPI_Irecv``
task per staggering.  Multiple pack kernels feed into the single
send task as dependents; the single recv task fans out into
multiple unpack kernels.  The executor polls each MPI task
independently — the graph never waits on "all sends done" or
"all receives done" before progressing.  As soon as a particular
``(local, r)`` pair completes its receive, the unpack kernels for
that pair are released, and any prolongation kernels whose
stencils only touch ghosts from that pair can proceed.  This is
the key concurrency property of the design: a slow rank pair
delays only its own downstream tasks, not the whole exchange.

**Restriction feeds the coarse buffer; copies feed ghosts directly.**
Same-level copies have no upstream dependencies (apart from the
coarse-buffer-touching ones, which depend on restriction).  MPI
sends depend on their pack predecessors; MPI receives are pure
sources.

**Prolongation depends on every ghost slot its stencil reads.**
This is the dependency rule that most often surprises people.  A
prolongation kernel filling a fine face ghost zone is not a
trivial point-wise interpolation: it has a stencil that reaches
into the adjacent edges (and, for high-order variants, the
adjacent corners) of the coarse buffer.  The dependency list for
that prolongation task therefore includes:

- the restriction kernel for this quadrant (which produces the
  coarse-buffer face data);
- the copy / unpack kernels for the adjacent coarse-buffer edge
  and corner slots that the stencil reads;
- the first physical-BC pass, if the prolongation stencil reaches
  into a region whose values were set by the BC.

The stencil width is the binding constraint: cell-centred hydro
prolongation uses a relatively narrow stencil, but the Z4c
prolongation uses the widest stencil in the code and so produces
the densest dependency fan-in.  Build helpers
(``insert_prolongation_tasks``) walk the descriptor list and
record every such edge; if you add a new variable with a
different stencil width, the helper must be told, or its
prolongation will fire too early on stale data.

**Phys-BC and ghost copies are stream-isolated** from pack /
unpack and prolongation, so multi-stream-capable backends can
overlap them.  The four streams (``copy_stream``, ``pup_stream``,
``phys_bc_stream``, ``interp_stream``) carry independent kernel
queues; cross-stream ordering is enforced only by explicit
dependency edges, never by stream membership.

There is **no implicit synchronisation** in the graph: any
ordering not expressed by a dependency edge is up to the
executor to schedule freely.  Adding new tasks always requires
both producing their dependency list and registering them as
dependents of their producers.


The executor
------------

``executor::run()`` is the single loop that drains the graph:

.. code-block:: cpp

   struct executor {
       std::deque<task_id_t>           ready ;        // FIFO of dispatchable tasks
       std::vector<task_id_t>          gpu_pending ;  // launched, awaiting event
       std::vector<task_id_t>          mpi_pending ;  // posted, awaiting MPI_Test
       std::vector<runtime_task_view>  rt ;           // task + atomic pending count
       void run(view_alias_t alias) ;
       void complete_and_release(task_id_t id) ;
       void reset() ;
   } ;

The loop:

1. **Dispatch ready tasks.**  Pop the front of ``ready`` and
   ``run()`` it.  GPU and MPI tasks immediately go into the
   corresponding pending list; CPU tasks complete synchronously
   and call ``complete_and_release`` themselves.
2. **Poll GPU tasks.**  Walk ``gpu_pending``; for each, query the
   device event.  On completion, ``complete_and_release(id)``.
3. **Poll MPI tasks.**  Walk ``mpi_pending``; for each, call
   ``MPI_Test(&mpi_req, …)``.  On completion,
   ``complete_and_release(id)``.
4. Loop until ``ready``, ``gpu_pending``, and ``mpi_pending`` are
   all empty.

``complete_and_release`` is the dependency mechanism: for each
dependent of the completing task, atomically decrement its
``pending`` count; if that reaches zero, push the dependent into
``ready``.  The use of an atomic pending counter means dependents
can be released safely from both the polling thread and from
synchronous CPU completion, even though the executor itself is
single-threaded.

The FIFO ordering of ``ready`` is the determinism anchor for the
graph: given the same task list and the same arrival order in
``ready``, the same execution interleaving results.  Combined with
sorted communicator-key assignment in
``amr_ghosts_construct_buffers.cpp`` (which fixes the MPI buffer
slot of every message before the run starts), the per-substep
floating-point output is bit-identical across MPI rank counts —
the property exercised by ``test_checkpoint_roundtrip`` and the
``checkpoint_1_rank`` cross-rank restart test.


Pack / unpack kernels
---------------------

The pack and unpack kernels are the only kernels that bridge
between the per-quadrant Kokkos View and the flat MPI buffer.
They live in
``include/grace/amr/ghostzone_kernels/pack_unpack_kernels.hh``
and follow a small template:

.. code-block:: cpp

   template <element_kind_t elem_kind, typename view_t>
   struct pack_op {
       view_t src ;                   // cc View or face-staggered View
       ghost_array_t dst ;            // send buffer
       index_transformer_t transf ;   // staggering-aware index mapping
       /* ... per-quadrant loop ... */
   } ;

The kernel iterates a ``MDRangePolicy`` over the descriptor list
for its element kind (face / edge / corner) and its staggering.
At each ``(q, ielem, ivar, k, j, i)`` point the kernel:

1. Calls ``in_range(transf, …)`` to filter out cells outside the
   actual valid range of this staggering — recall that the buffer
   envelope is sized to the max envelope, so the kernel touches
   only the cells in this staggering's true footprint
   (see :ref:`grace-ghost-zones`).
2. Computes the source ``(i_src, j_src, k_src)`` from the
   ``(i, j, k)`` buffer index via ``index_transformer_t``, which
   adds ``sx, sy, sz`` offsets for staggered fields.
3. Reads ``src(i_src, j_src, k_src, ivar, q)`` and writes
   ``dst.at_faces<elem_kind>(…)`` (or ``at_edges`` / ``at_corners``).

The unpack kernel is the symmetric inverse, with a wider
``in_range`` that includes the edge / corner boundary cells.  A
separate ``view_is_cbuf`` flag switches packing / unpacking to
``at_cbuf()`` accessors when the kernel is feeding the coarse
buffer instead of the standard send buffer.

For face-staggered fields, the half-cell backward shift documented
in :ref:`grace-variable-layout` is what the ``index_transformer_t``
encodes.  Lower-side face packing offsets the source index by
``sx`` to skip the *shared interface face* — that face belongs to
the lower-side quadrant in GRACE's staggering convention and
must not be packed twice across an MPI exchange.


Prolongation and restriction
----------------------------

Prolongation runs after MPI exchange and fills fine ghost zones
from a coarse buffer.  Restriction runs before MPI pack and
produces the coarse buffer from finer-side data.  Both have two
flavours.

**Cell-centred prolongation / restriction**
(``include/grace/amr/ghostzone_kernels/prolongation_kernels.hh``
and ``restrict_kernels.hh``) use plain Lagrange interpolation
with build-time-selectable order (2nd or 4th).  Restriction uses
volume-conservative averaging.  The slope-limited 2nd-order
variant is used for hydro conservatives at coarse–fine interfaces
to preserve positivity across the interpolant.

**Div-preserving prolongation / restriction** is selected for the
face-staggered ``B`` to keep ``∇·B = 0`` to round-off.  The
implementation follows the Tóth–Roe construction and runs in two
phases:

- Phase 1 reads the tangentially-known coarse face values and
  computes the new fine face values at half-cell positions.
- Phase 2 closes the divergence-free constraint by setting the
  remaining fine faces from the constraint itself.

The boundary face fill in phase 1 uses a check
``i_f == ngz - 2`` that requires ``ngz`` to be even (see the
chapter-6 pitfall).  GRACE always builds with ``ngz`` equal to 2
or 4, so the check is satisfied in practice.

Both prolongation and restriction read / write the coarse buffer
(``_coarse_buffers`` for cell-centred,
``_stag_coarse_buffers`` for face-staggered) rather than directly
touching neighbour quadrant Views.  The buffer interposition is
what lets the same prolongation kernel handle both local and
remote sources of coarse data — by the time prolongation runs,
the coarse buffer has been populated either by a same-rank pack
or by an MPI unpack.


Why a deferred BC pass exists
-----------------------------

The deferred BC pass exists because of one specific geometric
case: a coarse–fine AMR interface that also touches a *reflection*
boundary.  The mechanism is subtle and worth laying out step by
step.

**The setup.**  Consider a quadrant whose ghost layer needs to be
filled by prolongation from a coarser neighbour, and whose
prolongation footprint also intersects a physical boundary that
is a reflection plane.  The prolongation kernel reads from the
coarse buffer, not from the coarse neighbour quadrant directly;
the coarse buffer is a downsampled image that must include the
boundary cells the prolongation stencil reaches into.

**The dependency chain.**  For the prolongation to produce the
correct fine-side values, the *coarse buffer* must already
contain the reflection-BC-filled values at the relevant boundary
cells.  So the first phys-BC pass must do two things, in order:

1. Fill the BC into the real ghost-layer View on the *coarse*
   side (the usual job of any phys-BC pass).
2. Mirror that BC into the *coarse buffer* so prolongation can
   pull from it.

The marker ``tag_bcs_in_cbuf<stag>`` decides which BC slots fall
into this overlap region and therefore must be replicated into
the coarse buffer.

**After prolongation.**  Once prolongation has run, the fine
ghost-layer cells that overlap the boundary now hold values that
are a function of the prolonged coarse data, not of the original
boundary configuration.  Any *physical* BC that depends on those
fine values — and at a reflection boundary that BC value *is*
that fine value, mirrored across the plane — must be re-applied
now that the underlying state has been updated.  This is the
deferred BC pass.

**Cascading.**  BCs are not independent: an edge BC reads the
values produced by adjacent face BCs, a corner BC reads the
values produced by adjacent edge BCs.  If a face BC must be
deferred (because its prolongation footprint intersects a
reflection plane), then any edge BC adjacent to that face that
reads the face's values must also be deferred.  The same
cascade extends to corners.  The build pipeline computes this
closure when assembling ``deferred_phys_bc_kernels`` and
schedules every BC kernel in the closure to run again after
prolongation.

**Why only reflection.**  The deferred mechanism only works for
BCs whose value at a fine cell can be derived locally from other
cells in the same staggering at the same level.  Reflection
satisfies this: the mirrored value across a reflection plane
exists as a function of the interior of the post-prolongation
state.  Sommerfeld does **not**: the outgoing-wave condition
involves finite-difference derivatives in physical space, which
have no sensible representation in a downsampled coarse buffer,
and which cannot be reconstructed from the prolonged fine ghost
values without knowing the pre-prolongation history.  In
practice this is fine, because GRACE never places an AMR
coarse–fine interface against a Sommerfeld boundary — the AMR
levels that touch the outer Sommerfeld surface are uniformly
coarse, so no prolongation footprint can intersect it.  Outflow
and Lagrange-extrap BCs are similarly never paired with
deferred handling because production setups put them at the
outer domain boundary, beyond the AMR-active region.


Pitfalls
--------

- **Adding a task without updating dependents / dependencies.**
  A new task added to ``task_list`` after the graph has been
  built (or with malformed edges) sits in ``ready`` forever or
  fires before its prerequisites complete.  Always go through
  ``insert_*_tasks`` helpers, which handle the wiring.
- **Stream assumptions.**  Tasks on the same stream are serialised
  by the device; tasks on different streams may run concurrently.
  Code that assumes a write happens-before a read on the same
  ghost layer must either share a stream or wire an explicit
  dependency edge.
- **Polling order does not equal completion order.**  Tasks
  finish in whatever order the GPU and MPI runtimes decide.  Any
  logic that depends on "the previous task's result is visible"
  must say so through a dependency edge, not through ordering in
  the descriptor list.
- **MPI tag collisions.**  The cell-centred and per-staggering
  exchanges use distinct ``GRACE_HALO_EXCHANGE_TAG_*`` values.
  Adding a new staggering or a new exchange flavour without
  allocating a new tag will silently cross-talk with an
  existing one — the receive will land in the wrong buffer.
- **Deferred BC tagging coverage.**  ``tag_bcs_in_cbuf`` decides
  which ghost slots need re-firing after prolongation.  A bug
  here is invisible at convergence in smooth regions but shows
  up as O(1) BC violations at coarse-fine corners on boundary
  faces.  When debugging boundary-corner issues, this tagger is
  the first suspect.
- **GPU shared-face atomic ordering.**  Two adjacent quadrants
  share a face's worth of staggered B; unpack kernels on those
  quadrants may overlap in time on the same stream.  The unpacks
  do not contend because each writes its own ghost layer, but
  any kernel that *combines* ghost layers across quadrants (an
  edge or corner correction, a cross-quadrant reduction) must
  use atomics or explicit inter-quadrant synchronisation.
- **The graph is invalidated by regrid.**  Holding a
  ``task_id_t`` across regrid is meaningless.  Anything that
  drives the executor from outside (instrumentation, optional
  passes) must subscribe through ``update()`` and rebuild
  bookkeeping on every regrid.
