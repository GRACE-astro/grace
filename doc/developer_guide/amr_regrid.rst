.. _grace-amr-regrid:

AMR regrid
==========

Everything previous chapters described — the p4est forest
(:ref:`grace-p4est-wrapper`), the per-quadrant variable storage
(:ref:`grace-variable-layout`), the descriptors and pipelines for
ghost exchange (:ref:`grace-ghost-zones`,
:ref:`grace-ghost-exchange-pipeline`), the reflux machinery
(:ref:`grace-refluxing`) — is *static* between regrids.  This
chapter documents what happens at the regrid call itself: how
quadrants are flagged, refined, coarsened, balanced, repartitioned,
and how the field data and the staggered magnetic field are
migrated onto the new mesh without losing conservation or breaking
``∇·B = 0``.

A regrid is the most architecturally dense operation in GRACE.
It touches every singleton, replaces every descriptor list,
re-allocates every variable View, executes a custom
GPU + MPI task DAG, and ends with a fresh forest and a fresh
ghost-exchange task graph ready for the next substep.  This
chapter follows the operation end-to-end.


When regrid runs
----------------

The entry point is the free function

.. code-block:: cpp

   namespace grace::amr {
     bool regrid() ;   // returns true if the grid actually changed
   }

defined in ``src/amr/regrid.cpp``.  It is called from two places:

- **Initialisation.**  ``grace_initialize`` may call ``regrid()``
  multiple times during initial-data setup; the details are in
  the "Initial-time regrid" section below.
- **Evolution loop.**  The main loop calls ``regrid()`` every
  ``amr.regrid_every`` substeps once the simulation is past any
  user-configured warm-up phase.  The return value tells the
  loop whether downstream rebuild work (descriptors, ghost
  buffers, output writers, AH finder) needs to fire.

Refining at user-specified Fixed-Mesh-Refinement regions, and
keeping the level structure 2:1-balanced across the whole
forest, are both responsibilities of the regrid pass itself —
the caller does not need to coordinate either.


Initial-time regrid
*******************

At startup, the AMR criterion has not yet seen the initial data,
so the initial forest has only the structure declared by the
user — typically a coarse base grid plus any *Fixed-Mesh-Refinement*
(FMR) boxes.  Reaching the mesh resolution that production needs
requires the criterion to fire over the initial-data fields and
add the dynamically-tracked levels.

GRACE exposes two regrid channels at initialisation, controlled
by ``amr.regrid_at_preinitial`` and ``amr.regrid_at_postinitial``:

- **Pre-initial regrid.**  When the refinement criterion does
  *not* depend on the initial-data fields — typically a pure
  position-based criterion such as "refine in a moving box of
  radius R around a coordinate centre" — ``regrid_at_preinitial``
  fires the criterion *before* the initial-data routine is
  called, and recursively, calling ``regrid()`` until the
  forest stabilises.  No initial-data evaluation happens between
  the levels, which makes this path cheap: only the mesh moves,
  the field-data migration in each pass is trivial (the state is
  zero / floor everywhere), and the criterion converges in a
  small number of passes equal to the deepest level it wants to
  add.
- **Post-initial regrid.**  When the criterion *does* depend on
  the initial-data fields — refining where ``ρ`` exceeds a
  threshold, where the lapse is small, where matter is above the
  atmosphere floor by some factor —
  ``regrid_at_postinitial`` runs after the initial-data routine
  has populated the state on the current forest.  Each regrid
  pass at this stage costs a full state migration *and* a
  re-evaluation of the initial-data routine on the newly created
  quadrants, because the criterion needs current state on the
  finer mesh to decide whether to refine further.  This is more
  expensive but the only correct option for state-dependent
  criteria.

The two channels are not mutually exclusive.  A production setup
often uses *both*: a position-based pre-initial pass to build the
geometric skeleton (moving boxes around the stars, refinement
shells around extraction radii), followed by a state-dependent
post-initial pass to tighten the refinement around the stellar
surfaces or the orbital plane.

A note on the workflow this enables: **FMR boxes are static — they
are declared at startup and never touched by the AMR criterion.**
If you want, say, two levels of moving boxes around each neutron
star, the FMR alone is not enough; you need the AMR subsystem to
add those levels.  The standard pattern is to declare a small FMR
base — coarse enough to be cheap, refined enough to give the
criterion a useful starting point — and to rely on the
initial-time regrid passes to *warm-start* the AMR by adding the
remaining levels.  Skipping the warm-start means the first
evolution-time regrid has to do all the work at once, which is
both expensive and risks evolving for several substeps on an
under-refined mesh before the criterion catches up.


The transaction object
----------------------

The regrid is wrapped in a stack-local ``regrid_transaction_t``
(declared in ``include/grace/amr/regrid/regrid_transaction.hh``,
implemented in ``src/amr/regrid/regrid_transaction.cpp``).  It
has the lifetime of a single regrid call and owns:

- the *outgoing* / *incoming* lists for the three quadrant
  classes (refine, coarsen, keep);
- the *minimum-allowed-level* array used to enforce FMR floors;
- the *device-side scratch space* for state migration;
- the per-rank MPI buffers and the descriptor lists for the
  face copies that the div-preserving prolongation needs;
- the GPU / MPI task DAG that executes the actual migration.

The top-level ``regrid()`` constructs the transaction, asks it
whether the grid changed (which also performs the p4est
refine / coarsen / balance), and if so calls ``execute()`` to
run the migration:

.. code-block:: cpp

   bool regrid() {
       regrid_transaction_t trx{} ;
       bool grid_has_changed = trx.grid_has_changed() ;
       Kokkos::fence() ;
       trx.execute() ;
       return grid_has_changed ;
   }

The transaction is destroyed at the end of the call; its scratch
storage and descriptor lists go away with it.


The four phases
---------------

A regrid runs through four logical phases.  Each is short on its
own but they compose with subtle ordering constraints.

Phase 1: criterion evaluation
*****************************

Before any p4est mutation, the per-quadrant *refine flag* in
``user_data`` (see :ref:`grace-p4est-wrapper`) is updated by the
refinement-criterion pass.  The criterion is a free function
selected at runtime by the active *regridding policy*
(see ``include/grace/amr/regrid/regridding_policy_kernels.tpp``);
it inspects the state of each quadrant and writes one of the
``quadrant_flags_t`` values:

.. code-block:: cpp

   enum quadrant_flags_t : int8_t {
       DEFAULT_STATE = 0,
       NEED_PROLONGATION,   // refine this quadrant on the next pass
       NEED_RESTRICTION,    // coarsen this family of quadrants
       INVALID_STATE = -1,
   } ;

Policies range from simple FMR-only (refine inside a static box,
coarsen outside) to physically-driven (refine where ``|∇ρ|``
exceeds a threshold, where the lapse is below a threshold,
where the matter is above the floor by some factor).  Multiple
criteria are composed by OR: a quadrant is refined if *any*
criterion asks for refinement.


Phase 2: p4est refine / coarsen / balance
*****************************************

With flags in place, the transaction calls p4est in sequence:

.. code-block:: cpp

   p4est_refine_ext(  forest, /*recursive*/ 0, grace_maxlevel,
                      amr::refine_cback,
                      amr::initialize_quadrant,
                      amr::set_quadrant_flag ) ;

   p4est_coarsen_ext( forest, /*recursive*/ 0, /*callback_orphans*/ 0,
                      amr::coarsen_cback,
                      amr::initialize_quadrant,
                      amr::set_quadrant_flag ) ;

   p4est_balance_ext( forest, P4EST_CONNECT_FULL,
                      amr::initialize_quadrant,
                      amr::set_quadrant_flag ) ;

Three points worth knowing:

- **Recursive refinement is disabled.**  ``refine_recursive = 0``
  means a quadrant can only refine by *one* level per regrid call.
  Multi-level jumps require multiple regrids (e.g. during
  initial-data settling).  This keeps the migration phase
  tractable: the field-data prolongation only ever has to
  produce data for one level finer, not many.
- **Balance is full** (``P4EST_CONNECT_FULL``), meaning the 2:1
  balance condition is enforced not just across faces but also
  edges and corners.  This is the level-difference invariant
  every downstream descriptor relies on.
- **Whether the grid changed** is decided by comparing the
  p4est revision number before and after the three calls.  The
  ``grid_has_changed`` return propagates back through ``regrid()``
  and tells the main loop whether to fire the downstream
  rebuilds.

Phase 3: state migration
************************

If the grid changed, the transaction has to move field data from
the old mesh layout onto the new one.  This is the structurally
hardest phase.

The first step is to classify every new-mesh quadrant against the
old mesh:

.. code-block:: text

   for each new quadrant iq_new:
       flag = new_quad.get_regrid_flag()
       if   flag == NEED_PROLONGATION  ->  refine: 1 old → 8 new
       elif flag == NEED_RESTRICTION   ->  coarsen: 8 old → 1 new
       else                             ->  keep:    1 old → 1 new

The transaction fills three pairs of vectors —
``refine_incoming / refine_outgoing``,
``coarsen_incoming / coarsen_outgoing``,
``keep_incoming / keep_outgoing`` — pairing every new quadrant's
local index with the corresponding old quadrant index(es).  These
vectors drive the migration kernels.

The next step is to reallocate the scratch (second-time-level)
View to the new ``nq_regrid`` quadrant count.  ``state_p`` is the
scratch View from :ref:`grace-variable-layout`, used here as the
migration destination.  The old ``state`` keeps its current size
during the migration, gets read from, and is then
``Kokkos::resize``-d to the new shape after the migration kernels
have run.  Aux variables are reallocated to the new shape but
*not* re-computed — the chapter docstring on ``regrid()`` makes
this explicit, and downstream code is responsible for re-running
``compute_auxiliary_quantities()`` after the regrid returns.

The three migration patterns:

- **Keep.**  Pure local copy from old quadrant slot to new
  quadrant slot.  Same cell data, same staggered face B, same
  ghost zones (which are about to be re-filled anyway by the
  first post-regrid ghost exchange).
- **Coarsen.**  8 old child quadrants restrict to 1 new
  parent quadrant.  Cell-centred fields use volume-conservative
  averaging (sum over 2³ child cells, divide by 8).  Staggered
  face B uses a div-preserving restriction that respects
  ``∇·B = 0``.  All restriction kernels live in
  ``include/grace/amr/regrid/restrict_kernels.hh``.
- **Refine.**  1 old parent quadrant prolongs to 8 new child
  quadrants.  Cell-centred fields use the slope-limited or
  high-order Lagrange interpolant (selected per variable);
  staggered face B uses the Tóth–Roe div-preserving prolongation
  in two phases.  Prolong kernels live in
  ``include/grace/amr/regrid/prolong_kernels.hh``.

The div-preserving prolongation deserves its own subsection.

Phase 4: repartition
********************

Once every new-mesh quadrant has received its data, the
transaction calls

.. code-block:: cpp

   size_t transfer_count = p4est_partition_ext( forest, ... ) ;

which redistributes the quadrants across MPI ranks so the
per-rank workloads are roughly balanced.  ``p4est_partition_ext``
returns the number of quadrants that changed owner; if non-zero,
a follow-up MPI communication phase moves the associated field
data between ranks.

After partition, the forest is in its final shape and the
transaction can be torn down.


The MPI face copy and div-preserving prolongation
-------------------------------------------------

The most intricate piece of the migration is what happens to the
staggered face B when an old quadrant refines next to a neighbour
that was *already* at the finer level before this regrid call.
The reason this case is special — and the reason a whole face
copy / send / receive pipeline exists to handle it — is flux
consistency at the now-shared fine-level face.

Consider a refining quadrant at level :math:`\ell` whose neighbour
across a given face is already at level :math:`\ell + 1`.  Before
the regrid this was a hanging face; after the regrid it is a
flush same-level interface between two fine quadrants.  The
neighbour already carries the *correct* fine-level B-face values
on its side of the shared face — values that have been evolved by
constrained transport, that satisfy ``∇·B = 0`` exactly, and that
the previous EMF reflux pass has made consistent with the coarse
side.  If we now prolong the newly-refined quadrant's outer face
from its own coarse parent, the two sides of the shared face will
disagree on the B-normal value at the fine level, breaking the
constrained-transport invariant.

The correct treatment is to read the neighbour's existing fine
face data and *use it* as the outer-face boundary condition on
the newly-refined quadrant's Tóth–Roe prolongation.  Only the
*inner* faces of the new fine block are then filled by the
divergence-preserving solve.  This is the song and dance: locate
the neighbour, copy (or MPI-send) its fine face values into the
newly-refined quadrant's outer-face slots, then run the prolong.

The complementary case — refining next to a *coarse* neighbour —
is much simpler.  There is no pre-existing fine data on the
other side to preserve; the newly-refined quadrant simply
prolongs from its own coarse parent into all eight fine children,
and a fresh coarse–fine interface is created which the next EMF
reflux pass will then maintain consistent on subsequent substeps.
No face copy is needed, no MPI traffic is involved, and Phase 1
of Tóth–Roe fills the outer faces from the coarse-parent
interpolation as in the standard construction.

So the rule is: **the face copy only fires when the refining
quadrant has a pre-existing fine neighbour on that face**.  Other
configurations (neighbour at the same coarse level, neighbour
also refining, neighbour at the new coarse–fine interface
boundary) are handled by ordinary Tóth–Roe prolongation without
external face input.

The pipeline that handles the pre-existing-fine-neighbour case
lives in ``src/amr/regrid/regrid_build_buffers.cpp`` and
``src/amr/regrid/regrid_transaction.cpp``.  Three structures
matter:

**``fine_face_data_desc_t``** is the per-rank record exchanged
via ``MPI_Alltoallv``.  Each record carries:

- ``qid_local`` — the receiving rank's local fine-child quadrant id;
- ``qid_remote`` — the sending rank's local quadrant id (the p4est
  ``piggy3.local_num``, cumulative over trees);
- ``fid_local`` / ``fid_remote`` — face ids on the two sides;
- ``which_tree`` — the p4est tree index of the remote quadrant.

**``fine_interface_desc_t``** is the per-rank GPU-friendly list
that drives pack and unpack kernels.  Two flavours coexist:

- A *send* descriptor is built on the sending rank from the
  alltoallv output; its ``qid_src`` indexes the local sending
  quadrant, ``fid_src`` selects the face to read.
- A *recv* descriptor is built on the receiving rank; its
  ``qid_dst`` indexes the local fine child to write into,
  ``fid_dst`` selects the face to write.

The naming is initially confusing because the alltoallv exchange
inverts roles: rank A sends its *recv* descriptors (what it
needs), rank B receives them into its *send* buffer (what it
must provide):

.. code-block:: cpp

   MPI_Alltoallv(
       recvbuf.data(), recvcounts, rdispls, MPI_BYTE,  // what WE need
       sendbuf.data(), sendcounts, sdispls, MPI_BYTE,  // what THEY need from us
       MPI_COMM_WORLD
   ) ;

The "recvbuf" / "sendbuf" names refer to *data-flow roles* on
each rank, not MPI verbs.  This is intentional, and is the
source of more than one swap bug in the history of this
subsystem.

**``have_fine_data_x/y/z``** are per-rank boolean flag arrays
indexed by ``(iq, side)`` where ``iq`` is a fine-child absolute
index and ``side ∈ {0=lower, 1=upper}``.  A flag is set exactly
when the corresponding outer face is shared with a pre-existing
fine neighbour and will therefore be filled by a local fine-fine
copy or an MPI unpack *before* the prolongation kernel runs.
Phase 1 of Tóth–Roe consults these flags to skip the standard
coarse-parent interpolation on faces that already carry the
correct fine-level data; faces without the flag are filled from
the coarse parent in the normal way.

**The Tóth–Roe kernel** runs in two phases per fine block,
separated by a ``team.team_barrier()``:

- **Phase 1** fills the *outer* (even-indexed) faces shared with
  the coarse parent.  Loop over coarse cells ``(i, j, k)``;
  fine index is ``i_f = ngz + 2*i``.  Skip the lower face fill
  when ``have_fine_data`` is already set, otherwise interpolate
  the coarse face value.  The upper outer face is filled by a
  special branch that triggers when ``i_f == nx + ngz - 2``.
- **Phase 2** fills the *inner* (odd-indexed) faces by solving
  the discrete divergence constraint at each fine cell using
  the Phase-1 outer face values.  Uses Tóth & Roe equations
  8–10.

The crucial subtlety lives in Phase 1's last-face check.  It
must read ``i_f == nx + ngz - 2``, *not* ``i_c == ngz +
nx/2 - 1``: the latter form fires only for the upper child in
each pair, leaves the lower children's upper face uninitialised,
and breaks ``∇·B`` in Phase 2.  This was one of the historical
bugs found and fixed during the April 2026 div-B investigation.


The task DAG
------------

State migration runs on its own custom task DAG, separate from
the ghost-exchange one but built on the same
``include/grace/utils/task_queue.hh`` executor.  The build order
in ``regrid_transaction_t::build_task_list()`` is:

1. **MPI recv tasks** — immediately ready.
2. **MPI send tasks** — blocked on the pack kernels (which post
   GPU events the send tasks wait on).
3. **Local fine face copies** — same-rank fine-to-fine, runs on
   ``copy_stream``; these contribute to ``prolong_fs_dependencies``,
   the list of edges into the prolongation kernels.
4. **Keep-quadrant copies** — same-rank, same-level, run on
   ``copy_stream``.
5. **Restrict tasks** — same-rank coarsening, runs on
   ``interp_stream``.
6. **Pack kernels** — fill MPI send buffers from the fine-side
   data; runs on ``copy_stream``, fires a GPU event the
   corresponding MPI send waits on.
7. **Unpack kernels** — read MPI recv buffers into the
   destination fine ghost faces; runs on ``copy_stream``,
   depends on the MPI recv, contributes to
   ``prolong_fs_dependencies``.
8. **Cell-centred prolongation** — runs on ``interp_stream``.
9. **Div-preserving prolongation** — runs on ``interp_stream``,
   depends on every ``prolong_fs_dependencies`` task being
   complete (every outer face filled, both same-rank and MPI).

The executor follows the same pattern as the ghost-exchange
executor (:ref:`grace-ghost-exchange-pipeline`): dispatch ready,
poll GPU events, poll ``MPI_Test``, release dependents on
completion.  The MPI dependency structure is the same as ghost
exchange — one send and one recv task per rank pair, with packs
and unpacks fanning in and out — and the determinism guarantee
is the same: a slow rank pair delays only its own downstream
tasks, not the whole regrid.


Downstream rebuilds
-------------------

A grid that has changed invalidates most of the static state
documented in earlier chapters.  ``regrid()`` itself does not
rebuild these — its return value is the trigger for the caller
to do so.  The minimum set of post-regrid rebuilds is:

- **Ghost-exchange descriptors and task graph**
  (``amr_ghosts::get().update()``).  Every per-quadrant
  ``quad_neighbors_descriptor_t``, every face / edge / corner
  slot, every MPI send / receive buffer is rebuilt.
- **Reflux descriptors** (``_reflux_*_descs`` vectors and the
  associated buffers).  Same lifecycle as ghost descriptors.
- **Coordinate system** (``coord_system``).  The per-quadrant
  logical-to-physical extents are reconstituted from the new
  forest and the unchanged connectivity.
- **Spherical surfaces**.  Every registered detector recomputes
  its octree intersection and interpolation weights against the
  new quadrant layout
  (:ref:`grace-io-and-diagnostics`).
- **Auxiliary state.**  ``compute_auxiliary_quantities()`` must
  run before the next RK substep, because the regrid leaves the
  ``_aux`` View allocated-but-empty.

In production ``grace_initialize.cpp`` and the main evolution
loop know to do all of this when ``regrid()`` returns ``true``.
External code paths that bypass the main loop (a Catch2 test
that drives regrid directly, a diagnostic that calls regrid as
part of a special-case operation) must follow the same checklist
or end up reading stale auxiliary state.


Pitfalls
--------

- **The transaction object is the only place to look for migration
  state.**  Everything that has to survive between
  ``grid_has_changed()`` and ``execute()`` lives there; nothing is
  hidden in singletons or globals.  Adding new migration
  metadata should extend the transaction, not other state holders.
- **Recursive refinement is intentionally disabled.**  If your
  refinement criterion wants a quadrant to jump multiple levels,
  it has to be allowed to do so over multiple regrid calls.  This
  matters most during initial-data settling, where
  ``grace_initialize.cpp`` calls ``regrid()`` repeatedly in a
  loop.
- **2:1 balance is non-negotiable.**  Every descriptor downstream
  assumes ``level_diff ∈ {-1, 0, +1}``.  Building a custom
  refinement workflow that bypasses ``p4est_balance_ext`` and
  produces a deeper level jump will pass the migration phase
  silently but break ghost-exchange and reflux descriptors that
  cannot represent the larger gap.
- **Auxiliary state is invalid after regrid until you recompute it.**
  The ``_aux`` View is reallocated to the new shape, not populated.
  Anything that reads auxiliaries between ``regrid()`` returning
  and ``compute_auxiliary_quantities()`` running is reading
  uninitialised memory.
- **The alltoallv inversion is intentional and not a bug.**
  ``recvbuf`` and ``sendbuf`` in ``regrid_build_buffers.cpp``
  refer to *data-flow roles*, not MPI verbs; the
  ``MPI_Alltoallv`` swaps them because each rank knows what *it
  needs* but not what *other ranks need from it*.  Anyone who
  "fixes" this by renaming or re-swapping will produce data
  corruption that hides at small rank counts and explodes on
  many ranks.
- **The Tóth–Roe Phase 1 last-face condition must be on the
  fine index** (``i_f == nx + ngz - 2``), not the coarse index.
  The coarse-index form fires only for the upper child of each
  pair, leaves lower-child upper faces uninitialised, and shows
  up downstream as ``∇·B ≠ 0`` at AMR boundaries that aren't
  obviously related to the prolongation.
- **The** ``have_fine_data_*`` **flag arrays must be in sync with
  the actual pack/unpack/copy descriptor lists.**  ``build_buffers()``
  populates them once, before ``build_task_list()`` constructs the
  dependency graph; any later change to which descriptors will
  fire must update the flags or Phase 1 will skip faces that no
  longer get filled.
- **The output of regrid is non-deterministic across rank counts
  *unless* you preserve the in-place / canonical-order patterns
  used elsewhere in the code.**  The migration kernels themselves
  are race-free by construction (each (qid, ivar, i, j, k) is
  written by exactly one descriptor), but any caller-side
  reduction or diagnostic over the new mesh state must use the
  ``MPI_Allreduce`` + sort patterns documented in
  :ref:`grace-refluxing` to stay rank-invariant.
