.. _grace-checkpointing:

Checkpointing
=============

GRACE checkpoints persist the *minimum* state needed to resume a
simulation bit-identically from a saved iteration.  They are
shorter than the rest of the developer guide deserves: most of
the moving parts (the forest, the variable Views, the singletons)
have been documented in earlier chapters.  This chapter says what
the checkpoint *does* and *does not* contain, why the omissions
are safe, what the misc-data slot is for, and what to watch out
for when extending the checkpoint format.

The checkpoint handler is a singleton
(``include/grace/system/checkpoint_handler.hh``) exposing two
operations:

.. code-block:: cpp

   auto& cp = grace::checkpoint_handler::get() ;
   cp.save_checkpoint()       ;   // current state → file
   cp.load_checkpoint(iter)   ;   // file → singletons, iter=-1 → latest


What gets checkpointed
----------------------

In order of importance:

**The p4est forest topology.** Saved via ``p4est_save_ext`` /
loaded via ``p4est_load_ext`` (see :ref:`grace-p4est-wrapper`).
This is the only piece of the checkpoint that goes through the
p4est API; everything else is GRACE's own HDF5 writer.

**The evolved state.** Every variable registered as evolved 
(cell-centred state, staggered face B).
Stored as one HDF5 dataset per registered variable, sized by
the global quadrant count and ordered by global quadrant id.

**Ghost zones.** The ``ngz``-deep ghost layer wrapping every
quadrant is checkpointed alongside the interior cells.  This
is *not* an optimisation choice — it is required for correctness
when Sommerfeld outer boundaries are in play.  The Sommerfeld BC
implements an outgoing-wave condition via finite-difference
derivatives that read the *previous time level* in the ghost
layer; resuming from a saved state without those ghost cells
would produce wrong derivatives on the first post-load substep,
biasing the boundary signal for several light-crossing times.
The state-coverage test ``test_checkpoint_roundtrip`` explicitly
exercises the ghost layer.

**Co-tracker state.** The ``co_tracker`` singleton
(``include/grace/IO/diagnostics/co_tracker.hh``) carries the
miscellaneous per-run state that doesn't fit anywhere else:
compact-object positions, their characteristic radii, the
"have these objects merged yet?" boolean, and similar per-run
metadata.  

The file format is open-ended — anything that needs to
survive a restart but isn't part of the evolved variable list
ends up in the checkpoints.  See the next section for the pattern.

**Iteration counter and physical time.**  Saved as HDF5
attributes on the root group.


What is deliberately *not* checkpointed
---------------------------------------

**Auxiliary variables.**  Anything registered as auxiliary 
(primitives from c2p, c2p error bitmasks, cell-centred B reconstruction,
the various diagnostic fields).  These are deterministic
functions of the evolved state and are recomputed by
``compute_auxiliary_quantities()`` immediately after the load.
Checkpointing them would inflate file size by a large factor
without changing the resumed behaviour. (One small caveat: during 
evolution constraint violations are computed using the cached values
of the Ricci tensor and matter source terms from the last RK sub-step,
which makes them dramatically cheaper. At restart, everything is re-computed,
so the constraint violations at the same iterations might be slightly 
inconsistent).

**MPI topology.**  The number of ranks, the rank ownership of
quadrants, and any other partition-dependent metadata are
explicitly *not* part of the file.  Quadrant data is laid out in
global quadrant order — a single contiguous run across the
whole forest — and the loading rank reads only the slice
corresponding to its share of the post-load partition.  This is
what makes restarts across different rank counts cleanly
supported: a run saved on *N* ranks can be restored on *M*
ranks unchanged, because each loading rank computes its own
offset into the file from the fresh partition.  Verified by
``test_checkpoint_roundtrip`` at multiple rank counts and by the
``checkpoint_1_rank`` cross-rank restart test.

**Ghost-exchange descriptors, reflux descriptors, IO
descriptors.**  These are all derivable from the post-load
forest plus the variable registry; they are rebuilt by the
standard post-regrid bookkeeping in the load path.

**The forest revision number, the task ids in the executor, any
per-substep scratch.**  Transient state.  The load path
constructs a fresh task graph and a fresh executor from the
loaded forest.


The co-tracker misc-state pattern
---------------------------------

The co-tracker (compact object tracker) is the canonical pattern for anything that needs
to persist across a checkpoint but is not a field variable.  Its
shape:

- A small singleton holding a flat collection of typed slots
  (positions, radii, scalars, booleans, sometimes small arrays).
- A registration step at startup that declares which slots exist
  and what they mean.
- ``save()`` / ``load()`` methods that serialise the slot
  collection to / from the checkpoint HDF5 alongside the field
  data.

When adding new run-level metadata that needs to survive a
restart — a new diagnostic accumulator, a new AH tracker, a
counter that must not reset on resume — the right move is to
register a custom checkpoint load store that can follow the design
pattern of the co-tracker.

On-disk layout
--------------

A checkpoint is a single HDF5 file per saved iteration.  The
layout is flat:

- root attributes: ``iteration``, ``time``, format version, ...
- one dataset per evolved variable, shape
  ``(global_num_quadrants, ny_tot, nx_tot, nz_tot, n_components)``
  for cell-centred variables, with analogous shapes for the
  staggered face variants;
- a p4est-native subtree produced by ``p4est_save_ext`` holding
  the forest topology;
- a co-tracker subtree holding the misc-state slots.

The quadrant ordering inside each dataset is the *global*
quadrant index defined by the p4est partition at *save* time.
At load time this is irrelevant — every loading rank knows its
own slice of the global ordering from the freshly-partitioned
forest and reads the corresponding hyperslab independently
through collective MPI-IO.  No rank communicates the offsets to
any other rank; they all compute the same thing locally from the
forest.

Checkpoints are written and read with HDF5 collective MPI-IO
with compression disabled.  The combination is what makes the
round-trip bit-exact for ``double`` precision; any compression
codec or non-collective mode would compromise bit-exactness even
without rank reshuffling.


Pitfalls
--------

- **Do not apply boundary conditions post-load.**  The first
  evolve substep does this itself, with the correct ``dt`` and
  ``dtfact``.  An eager call to ``apply_boundary_conditions()``
  immediately after a load (with ``dt = 0``) silently perturbs
  the loaded state at the ulp level at reflection-symmetry
  corners where multiple BC kinds compose, breaking
  restart-bit-exactness for reflection-enabled runs.  This was a
  real bug at ``grace_initialize.cpp:270``, resolved by removing
  the unconditional BC call from the load path; see the
  resolved-bug memory ``checkpoint_load_apply_bc_bug`` for the
  full diagnosis.
- **Adding a new variable changes the checkpoint contents.**  A
  variable added to the evolved set will be present in
  checkpoints written after the change and *absent* from
  checkpoints written before it.  The load path must tolerate
  missing datasets (skip and zero-initialise, or refuse with a
  clean error), and the format-version attribute on the root
  group must be bumped so the policy is unambiguous.  Older
  checkpoints become unreadable across breaking schema changes
  unless an explicit migration step is provided.
- **Auxiliary recomputation is not free.**  Loading a checkpoint
  is followed by a full c2p sweep over every interior cell
  before the first substep runs.  On large grids this is
  several seconds of work; budget for it when scripting
  back-to-back restarts (e.g. parameter sweeps).
- **Cross-rank restart is bit-exact for the state, not for
  diagnostics.**  Per-rank reductions (the boundary-outflow
  accumulator's atomic-add, any other rank-dependent sum) can
  differ at ``ulp`` scale when the same run resumes on a
  different rank count.  This is expected: the *evolution* is
  rank-invariant, the *diagnostic taps* are not, by design.
  Conservation tests that compare across rank counts have to
  use the rank-invariant accumulators (``MPI_Allreduce``-summed,
  in-place flux corrections) documented in
  :ref:`grace-refluxing`.
- **Single-file checkpoints scale with output frequency.**  At
  HR with 16M cells and a full Z4c + GRMHD evolved set, a
  single checkpoint is tens of GB.  Schedule
  ``max_n_checkpoints`` and ``checkpoint_interval`` so the
  on-disk footprint stays bounded; the default cleanup logic
  rotates the oldest checkpoint when the limit is reached.
