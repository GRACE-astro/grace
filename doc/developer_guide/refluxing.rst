.. _grace-refluxing:

Refluxing at coarse–fine interfaces
===================================

Finite-volume evolution on an adaptively refined mesh is only
conservative if the numerical flux through a coarse–fine interface
is the same number when seen from the coarse side and when seen
from the fine side.  An equivalent statement: the area-weighted
sum of the four fine fluxes on a hanging face must equal the
single coarse flux on the parent face.  GRACE's Riemann solver
does not enforce this — the two sides compute their fluxes
independently, from independently reconstructed primitives, and
will disagree at the truncation-error level — so a follow-up
correction is required.  That correction is *refluxing*.

The same problem occurs for the staggered magnetic field: if the
edge-centred EMFs on the two sides of a coarse–fine interface
disagree, the constrained-transport update will accumulate
``∇·B ≠ 0`` at the interface.  GRACE's EMF reflux fixes this in
the same spirit, by replacing the coarse-side EMFs with the
properly-restricted fine-side values.

This chapter describes both: how face-flux reflux and EMF reflux
are organised, where they run in the evolution loop, what their
descriptor / buffer structure looks like, why they are *not*
implemented through the ghost-exchange executor, and the
floating-point determinism constraints that the implementation
satisfies.


Two flavours, one shape
-----------------------

GRACE has two distinct refluxing passes that share a structural
template but operate on different data:

**Face-flux reflux** corrects the conservative-variable fluxes on
the *cell faces* of a coarse–fine interface.  It is what keeps
total mass, momentum, energy, and the other GRMHD conservatives
exactly conserved across AMR boundaries.  It runs for every
quadrant pair where a coarse face hangs on a set of fine faces.

**EMF reflux** corrects the electromotive forces on the *cell
edges* of a coarse–fine interface.  It is what keeps the
discrete ``∇·B = 0`` constraint exactly satisfied across AMR
boundaries under constrained transport.  It has a richer set of channels
because EMFs live on edges (which can be hanging *or* shared
between coarse-fine interfaces in more configurations than
faces).

Both flavours follow the same shape:

1. **Fill phase** — pack the relevant quantities (fluxes or EMFs)
   into per-rank send buffers, post non-blocking MPI sends and
   receives.  The async handles live in a *transfer context* that
   the correction step will later consume.
2. **Apply phase** — wait on the context, then run device kernels
   that *replace* (not add) the coarse-side values in place with
   the area-restricted fine-side values.

Both flavours hold their own descriptors and their own send /
receive buffer Views; they do not share storage or descriptor
structures with the ghost-exchange subsystem.


Where they run in the evolution loop
------------------------------------

Refluxing is *not* task-based.  It is a sequence of plain
function calls inside the RK-substep body in
``src/evolution/evolve.cpp``:

.. code-block:: cpp

   compute_fluxes(...) ;       // per-direction Riemann solves
   compute_emfs(...) ;         // CT EMFs

   #ifdef GRACE_ENABLE_FOFC
   flag_fofc_cells(...) ;
   apply_fofc_correction(...) ;
   #endif

   auto flux_context = reflux_fill_flux_buffers() ;   // async send/recv
   auto emf_context  = reflux_fill_emf_buffers()  ;   // async send/recv

   reflux_correct_fluxes(flux_context) ;              // replaces flux array in place
   boundary_outflow_t::get().accumulate(dt, dtfact) ; // diagnostic uses corrected flux

   add_fluxes_and_source_terms(...) ;                 // consumes the corrected fluxes

   reflux_correct_emfs(emf_context) ;                 // replaces EMF array in place
   update_CT(...) ;                                   // consumes the corrected EMFs

   update_fd(...) ;

Three ordering facts to internalise:

- **Reflux runs after FOFC** so the corrections operate on the
  final flux / EMF state the substep will actually use, not on
  an intermediate state that FOFC would have overwritten anyway.
- **Flux reflux runs before** ``add_fluxes_and_source_terms``,
  because that kernel reads the flux array; if the array were
  not yet corrected, the divergence update would propagate the
  wrong mass into the coarse cells at the interface.
- **EMF reflux runs before** ``update_CT``, for the analogous
  reason on the staggered B field.

The ``parallel::grace_transfer_context_t`` returned by the fill
functions encapsulates the in-flight MPI requests for that pass.
Returning it as an explicit handle gives the call site the
opportunity to interleave other work between the fill and the
correction.  In the current evolution loop the interleaved work
is small (the diagnostic accumulate, and for EMF the entire
``add_fluxes_and_source_terms`` kernel), but the structure leaves
room to expand.


The five EMF channels
---------------------

EMF reflux is structurally more elaborate than face-flux reflux
because EMFs live on edges, and edges at coarse–fine interfaces
come in more flavours than faces.  Five channels participate,
each carrying a distinct quantity across a distinct descriptor
list:

1. **Hanging-face EMFs** (``_reflux_face_descs``).  The fine-side
   EMFs on the four fine faces that hang on a coarse face,
   restricted onto the coarse face.  Consumed by
   ``reflux_correct_emfs``'s face-application kernel, which
   *skips* the four edge positions at ``{ngz, nx/2+ngz}`` because
   those are handled by channels 4 and 5 below.
2. **Coarse-face EMFs** (``_reflux_coarse_face_descs``).
   Same-level coarse-side EMFs on a face whose neighbour is at the
   same level.
3. **Hanging-edge EMFs** (``_reflux_edge_descs``).  EMFs on edges
   that hang at a coarse–fine interface.  Two sub-types coexist,
   distinguished by how the p4est iterator created them:

   - *Face-registered*: created during the face traversal in
     ``register_refluxing_edges`` in ``iterate_faces.cpp``.
     Carry ``off_i`` / ``off_j`` offsets multiplied by ``nx/2``
     in the kernel to shift into the correct half of the fine
     quadrant. These correspond to edges of the fine quadrants
     that are not topolgical edges in the p4est sense since they 
     fall in the middle of a face on the coarse side.
   - *Edge-registered*: created during the edge traversal in
     ``iterate_edges.cpp``.  Have ``n_sides ∈ {2, 4}`` for
     boundary / interior edges (see below).

4. **Coarse-edge EMFs** (``_reflux_coarse_edge_descs``).
   Same-level edge corrections that average EMFs from up to four
   sides.

Channels 3 and 4 carry edge values that may be touched by
multiple descriptors from neighbouring faces and edges
simultaneously.  This is the source of two of the subtler
determinism constraints documented below.

Face-flux reflux is simpler: there is only one channel
(``_reflux_face_descs`` again, but consumed by the flux pipeline
instead of the EMF pipeline), and the per-descriptor target slot
is uniquely keyed by ``(qid, idir, side, i, j, ivar)`` so no
multi-writer collisions arise.


Descriptors and lifecycle
-------------------------

Reflux descriptors are stored in vectors owned by the AMR-ghosts
singleton but distinct from the ghost-exchange descriptors:

- ``_reflux_face_descs``      — face-level descriptors
- ``_reflux_coarse_face_descs`` — coarse-face descriptors
- ``_reflux_edge_descs``      — hanging-edge descriptors
- ``_reflux_coarse_edge_descs`` — coarse-edge descriptors

Each descriptor records the *quadrant ids and ranks of every side
of the coarse–fine relation*, plus the buffer slot it occupies
in the send / receive buffers (see the underlying types
``hanging_face_reflux_desc_t``, ``full_face_reflux_desc_t``,
``hanging_edge_reflux_side_t``, ``hanging_remote_reflux_desc_t``
in ``include/grace/amr/amr_ghosts.hh``).

Lifecycle:

1. **Created** during the p4est iterator callbacks
   (``grace_iterate_faces``, ``grace_iterate_edges``), which push
   descriptors into the four ``_reflux_*_descs`` vectors via the
   ``iter_data`` payload.
2. **Cleared** at the start of ``amr_ghosts_impl_t::update()``,
   which is called after every regrid.
3. **Buffer ids assigned** by ``build_reflux_buffers()``, which
   sorts descriptors by a deterministic communicator key
   ``comm_key_t = (rank, quad_id, elem_id, kind, is_cbuf_p2p)``,
   deduplicates, and assigns each descriptor a ``buf_id``
   pointing into the Kokkos send / receive arrays.
4. **Consumed** by ``reflux_fill_*`` and ``reflux_correct_*`` on
   every RK substep until the next regrid.

The sort-and-dedupe by ``comm_key_t`` is what makes the
per-descriptor MPI buffer slot deterministic — it is fixed
purely by the topology of the forest and the rank assignment,
not by descriptor-iteration order.  This is the foundation that
the bit-reproducibility guarantees rest on.


Buffers and the transfer context
--------------------------------

Each reflux channel has its own pair of send / receive buffers,
accessed through methods on the ghost-layer singleton:

.. code-block:: cpp

   ghost_layer.get_reflux_emf_send_buffer()             // hanging-face EMFs
   ghost_layer.get_reflux_emf_recv_buffer()
   ghost_layer.get_reflux_emf_coarse_send_buffer()      // coarse-face EMFs
   ghost_layer.get_reflux_emf_coarse_recv_buffer()
   ghost_layer.get_reflux_emf_edge_send_buffer()        // hanging-edge EMFs
   ghost_layer.get_reflux_emf_edge_recv_buffer()
   ghost_layer.get_reflux_emf_coarse_edge_send_buffer() // coarse-edge EMFs
   ghost_layer.get_reflux_emf_coarse_edge_recv_buffer()

Plus the flux pair for the face-flux pass.

The transfer context returned by ``reflux_fill_*`` is a
``parallel::grace_transfer_context_t`` — a lightweight handle
that owns the in-flight ``MPI_Request`` handles and an
opportunity to attach completion side-effects.  The fill function
posts all sends and receives, captures the requests in the
context, and returns.  The correction function uses the context
to wait on completion before its kernel reads from the receive
buffer.  No tasks, no executor: just two phases per call,
explicitly ordered.

Why a context handle instead of an internal sync?  Because
``reflux_fill_*`` returns *before* the MPI traffic has completed,
giving the call site the chance to do useful unrelated work
(in the current loop: the FOFC tail, the boundary-outflow
diagnostic, the ``add_fluxes_and_source_terms`` kernel) while
the network does its job.  By the time the matching
``reflux_correct_*`` call is reached, the receives have usually
already completed and the wait is free.


The correct-in-place pattern
----------------------------

This is the central design constraint that the whole reflux
subsystem turns on.  Both ``reflux_correct_fluxes`` and
``reflux_correct_emfs`` correct the *flux / EMF array* in place:

.. code-block:: cpp

   fluxes(ijk_fs, ivar, idir, qid_c) = F_fine_avg ;
   emf   (ijk_es, ivar, iedg, qid_c) = E_fine_avg ;

rather than patching ``new_state`` after the divergence update:

.. code-block:: cpp

   // OLD, BUGGY:
   new_state(ijk_cc, ivar, qid_c) += sign * dt * idx * (F_fine_avg - F_coarse) ;

The reason is correctness, not aesthetics.  At edge and corner
cells of a coarse–fine interface, the coarse cell is touched by
*multiple* reflux descriptors — for instance, the ``(+x, +y)``
edge cell of a coarse quadrant is touched by both the ``+x``-face
and the ``+y``-face descriptors.  Under the old ``+=`` pattern,
the kernel ``MDRangePolicy`` could schedule two descriptors'
writes to the same cell with no atomicity guarantee, producing
*both* a write race (two threads writing concurrently to the
same address) *and* a floating-point non-associativity hazard
(the sum result depends on which descriptor finishes first).
Bit-exactness across MPI rank counts was lost: the same physical
configuration partitioned over 1 rank vs 2 ranks produced
different round-off at edge / corner cells.

The fix is the in-place replacement above.  Each descriptor
writes to a *unique* ``(qid_c, idir, side, i, j, ivar)`` flux
slot, so multi-descriptor writes to the same slot do not occur.
Mathematically the two formulations are equivalent — replacing
the high-order coarse flux ``F_hi`` with the restricted
``F_fine_avg`` and then evaluating the divergence telescopes to
exactly the same ``Δstate`` — but only the in-place form is
race-free and bit-invariant under MPI repartition.

The same pattern is the right shape for any future
AMR-interface correction.  When you find yourself reaching for
a ``+=`` on a state variable from inside a descriptor kernel,
ask first whether multiple descriptors can touch the same target
cell.  If they can, the correction must instead go through a
buffer slot keyed uniquely per descriptor (a flux, an EMF, or a
new dedicated buffer) and the kernel that consumes that buffer
must run separately.


Deterministic edge accumulation
-------------------------------

EMF reflux at edge positions has a residual determinism problem
that the in-place pattern does not solve on its own.  Channels 3
and 4 (hanging-edge and coarse-edge EMFs) accumulate
contributions from *multiple* fine-side EMFs that, in general,
arrive in MPI-completion order — not in a deterministic order.
The same EMF would then be computed as different floating-point
numbers under MPI repartition.

The fix is canonical ordering of contributions by edge id.  In
``reflux_emf_compute_edge`` (hanging-edge) and
``reflux_coarse_emf_compute_coarse_edge`` (coarse-edge), the
relevant kernels collect all contributing EMFs into a small
local array, insertion-sort by ``edge_id``, and then accumulate
in the sorted order.  Insertion sort is the right choice on
device because the contributing count is tiny (at most a
handful) and the sort cost is dominated by load latency.

This restores rank-invariance: the accumulation order is fixed
by the edge topology, not by the order in which receives
complete.

A subtle complement: contributions that cross MPI ranks are
combined with ``MPI_Allreduce`` *before* the per-cell sort
where appropriate, so the rank dimension of the determinism
problem is also eliminated.  Together with the in-place
correction pattern, these two mechanisms make the entire reflux
result bit-identical across rank counts.


Boundary edges with reflection symmetry
---------------------------------------

In an earlier version of the code, every edge with ``sides < 4``
(at the p4est iterator boundary level) was skipped for reflux.
This was correct for the ``sides == 1`` case (a domain corner —
only one quadrant participates and there is nothing to correct)
but wrong for ``sides == 2`` (a domain face boundary that is also
a level interface).

Under reflection symmetry, the two missing quadrants on a
``sides == 2`` edge are mirror images of the two present ones,
and the EMF contribution they would make is identical (up to
parity).  The correct behaviour is to register the physical-BC
descriptor for that edge and then to build a reflux descriptor
with ``n_sides = 2`` and continue.  All downstream consumers
already loop over ``n_sides`` from the descriptor rather than a
hardcoded ``4``, so no changes were needed below the
registration layer.

The same does *not* apply at outflow / Sommerfeld boundaries —
there is no symmetry there that would let a reduced-count edge
be corrected meaningfully — but in practice GRACE never places
an AMR coarse–fine interface against those boundaries (the
outermost levels are uniformly coarse), so the question is moot.


Pitfalls
--------

- **Calling order in** ``evolve.cpp`` **is load-bearing.**
  ``reflux_fill_*`` must run after the substep's
  ``compute_fluxes`` / ``compute_emfs`` (and after FOFC, if
  enabled) but before the matching consumer kernel
  (``add_fluxes_and_source_terms`` for fluxes, ``update_CT`` for
  EMFs).  Inserting any kernel that reads the flux or EMF array
  between fill and correct is a silent correctness bug.
- **Diagnostic taps on conserved quantities must read post-reflux
  state.**  An RHS evaluation or conservation diagnostic between
  ``reflux_correct_fluxes`` and ``add_fluxes_and_source_terms``
  sees the corrected fluxes and is consistent.  One placed
  *before* ``reflux_correct_fluxes`` sees the uncorrected
  high-order coarse flux and reports a C-F mismatch that has
  no physical meaning.  The mass-conservation accumulator in
  ``boundary_outflow_t`` is positioned correctly for this reason;
  new diagnostics should follow the same placement.
- **No ``+=`` writes inside descriptor kernels.**  This is the
  single most important architectural rule of the subsystem.
  If a new correction needs to combine multiple contributions
  to the same target cell, the combination must go through a
  buffer slot keyed uniquely per descriptor, and the
  accumulation must run as a separate kernel with explicit
  ordering — never as concurrent ``+=`` updates.
- **Edge-correction kernels must canonicalise contribution
  order.**  Any new edge-level correction must insertion-sort
  its contributions by a topology-derived key (edge id, comm
  key, etc.) before accumulating.  Without this, MPI rank
  repartition produces non-bit-identical results at edge cells.
- **The descriptor vectors are invalidated by regrid.**  Like
  the ghost-exchange descriptors, the reflux descriptors are
  rebuilt by ``amr_ghosts_impl_t::update()`` after every regrid.
  Holding pointers, indices, or ``buf_id`` values across a
  regrid is meaningless.
- **The transfer context is single-use.**  Each call to
  ``reflux_fill_*`` produces a fresh context; the corresponding
  ``reflux_correct_*`` consumes it.  Re-using a context after
  the wait has completed, or attempting to fill a second pair
  of buffers from the same context, will produce MPI errors or
  silently corrupted buffers.
- **Channels 1 (hanging-face EMF) and 4 (hanging-edge EMF) must
  agree at edge positions.**  Channel 1's kernel skips the edge
  positions at ``{ngz, nx/2+ngz}`` because channel 4 owns them.
  If you ever add a new EMF-style channel, decide which
  positions it owns and make the existing channels skip those
  positions to avoid double-writes.
