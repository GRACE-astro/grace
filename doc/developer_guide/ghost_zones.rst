.. _grace-ghost-zones:

Ghost zones and descriptors
===========================

Every stencil in GRACE that reads beyond a quadrant's interior
cells expects those cells to be already filled with valid data: the
``ngz``-deep ghost layer wrapping each quadrant block (see
:ref:`grace-variable-layout`).  Filling those ghosts is the job of
the ghost-exchange subsystem.

This chapter describes the *static* side of that subsystem — the
question "where could a given ghost cell's data come from, and how
is that recorded?" — and leaves the question "in what order is the
work actually executed?" for the next chapter.  Concretely, this
chapter covers:

- the five categories of ghost-data source GRACE has to handle;
- the descriptor abstraction that records what each ghost zone of
  each local quadrant needs;
- the ``ghost_array_t`` storage layer that backs the send / receive
  / coarse buffers;
- the lifecycle of all of this across regrids;
- the physical boundary-condition kinds and how reflection symmetry
  composes with them;
- the few sharp edges that make this subsystem notoriously hard to
  modify safely.

The companion chapter :ref:`grace-ghost-exchange-pipeline` documents
the per-substep task graph, the pack / unpack kernels, the MPI
ordering, and the prolongation / restriction kernels.


Five sources of ghost data
--------------------------

For any given ghost cell of a local quadrant, the data has to come
from exactly one of five places:

1. **Same-level, same-rank neighbour.**  The simplest case: an
   adjacent quadrant on the same refinement level, owned by this
   rank.  Filled by a local Kokkos copy from the neighbour's
   interior to this quadrant's ghost layer.

2. **Same-level, different-rank neighbour.**  Same refinement
   level, but the neighbour lives on another MPI rank.  The remote
   rank packs its interior cells into a send buffer; we receive
   them into our receive buffer and unpack into the ghost layer.

3. **Coarser-level neighbour (prolongation).**  At an AMR
   coarse–fine interface, the coarse side has fewer cells than the
   fine side needs.  A high-order interpolant fills the fine
   ghosts from the coarse cells.  For metric variables this
   is plain Lagrange interpolation; for hydro this is slope limited, conservative 
   2nd order interpolation; for the staggered magnetic
   field it is a div-preserving prolongation (Tóth–Roe) that
   respects ``∇·B = 0`` to round-off.

4. **Finer-level neighbour (restriction).**  The complementary
   case: a coarse quadrant whose ghost layer needs data from finer
   neighbours.  Averaging fills cell-centred ghosts; a
   div-preserving restriction handles the staggered face-B case.

5. **Physical domain boundary.**  At the outer edges of the
   computational domain (and at any reflection plane) there is no
   neighbour at all.  A boundary-condition kernel fills the ghosts
   from this quadrant's own interior: outflow, Lagrange
   extrapolation, Sommerfeld, or reflection.

Every ghost-zone slot of every local quadrant falls into exactly
one of these five categories.  The descriptor system records *which*.


The descriptor abstraction
--------------------------

For each local quadrant, GRACE maintains one
``quad_neighbors_descriptor_t`` instance describing how to fill
its ghost layer.  In 3D the descriptor is

.. code-block:: cpp

   struct quad_neighbors_descriptor_t {
       std::array< face_descriptor_t,    6  > faces ;
       std::array< edge_descriptor_t,    12 > edges ;
       std::array< corner_descriptor_t,  8  > corners ;
       size_t quad_id ;     // local index of this quadrant
       size_t cbuf_id ;     // coarse-buffer slot (fine-side quadrants only)
       int8_t local_child_id ;
       // ...
   } ;

— one slot per face, edge, and corner of the quadrant.  Faces are
indexed by the six p4est face codes (±x, ±y, ±z), edges by the
twelve p4est edge codes, corners by the eight p4est child codes.

Each per-slot descriptor is a tagged structure recording two
critical fields, plus a data union:

.. code-block:: cpp

   struct face_descriptor_t {
       interface_kind_t  kind ;        // PHYS or INTERNAL
       int8_t            level_diff ;  // FINER=-1, SAME=0, COARSER=+1
       face_data_t       data ;        // union { full, hanging, phys }
       int8_t            face ;        // face code seen from the other side
       int8_t            child_id ;
   } ;

The two tags answer the categorisation question:

- ``kind == PHYS`` means there is no neighbour on this side and the
  slot will be filled by a physical-BC kernel.  ``kind == INTERNAL``
  means there is a neighbour somewhere — possibly same rank,
  possibly remote, possibly at a different level.
- ``level_diff`` distinguishes the three internal cases:
  ``SAME = 0`` (same level → copy or MPI), ``COARSER = +1``
  (neighbour is one level coarser → prolongation), ``FINER = -1``
  (neighbour is one level finer → restriction).

The remaining variant data lives in the union ``face_data_t``
(and analogously ``edge_data_t`` / ``corner_data_t``):

- ``full_face_t`` — one same-level neighbour: its quadrant id,
  send / receive buffer indices (used only when remote), the
  ``is_remote`` flag, the owner rank, and a per-staggering
  task-id slot used to wire dependencies into the task graph.
- ``hanging_face_t`` — multiple fine-level neighbours (four in 3D):
  arrays of quadrant ids, buffer indices, remote flags, and
  per-staggering task ids.
- ``physical_face_t`` — no neighbour: the outward normal
  direction, a flag indicating whether this physical face also
  needs to be filled inside the coarse buffer used by
  prolongation, the element kind, and task ids.

Edge and corner descriptors follow the same template; the only
substantive difference is that corners and 2D-AMR edges have at
most one hanging neighbour each, so their hanging variants are
single-entry rather than array-valued.

Two practical points:

- The descriptor is *all* that the per-substep kernels need to
  know.  Pack / unpack / copy / prolong / restrict / phys-BC
  kernels are dispatched purely off ``kind`` and ``level_diff``;
  none of them look at the underlying p4est forest at run time.
- The slot index *is* the geometric position.  Face slot ``0`` is
  always the ``-x`` face, slot ``5`` is ``+z``, edge slot ``0`` is
  the ``-y/-z`` edge of the quadrant, and so on, following p4est
  conventions.  The descriptor stores *what to do* at each slot;
  the index already tells the kernel *where*.


The ``ghost_array_t`` storage layer
-----------------------------------

All buffer storage used by ghost exchange lives in flat
``Kokkos::View<double*, default_space>`` allocations, indexed by
hand-computed offsets rather than by multidimensional View extents.
The wrapper type ``ghost_array_t``
(``include/grace/amr/ghostzone_kernels/ghost_array.hh``) holds the
View plus the offset tables; per-element accessors compute the
flat index from ``(rank, element kind, i, j, k, ivar, ielem)``.

The layout has three nested tiers:

- **Per-rank base offsets.**  ``rank_offsets[rank]`` gives the
  starting position in the View for all data exchanged with that
  rank.  Both send and receive buffers use this same model, with
  separate offset tables.
- **Per-element-kind sub-ranges.**  Within a rank's slab, the
  data is stored *faces*, then *edges*, then *corners*.  The
  accessor methods ``at_faces``, ``at_edges``, ``at_corners``
  apply the right sub-offset.
- **Per-element strides.**  Inside an element kind, the stride
  pattern is ``(i, j, k, ivar, ielem)`` with strides set once at
  buffer construction by ``set_strides()``.

For the AMR coarse-fine path, a parallel set of strides
(``cfstrides``, ``cestrides``) addresses the *coarse buffer*
backing prolongation.  The coarse buffer holds a downsampled copy
of fine-side data that the prolongation kernels then expand back
into the fine ghost layer, accessed through ``at_cbuf()``.

The buffers materially used by the ghost subsystem are:

- ``_send_buffer`` — outgoing MPI data, packed per substep.
- ``_recv_buffer`` — incoming MPI data, unpacked per substep.
- ``_coarse_buffers`` — cell-centred coarse data for cell-centred
  prolongation.
- ``_stag_coarse_buffers`` — three face-staggered coarse buffers
  (one per staggering) for div-preserving prolongation of B.

The buffer envelopes are sized to the *worst-case* element
footprint across all three staggerings.  This costs roughly 30 %
extra memory but keeps the offset arithmetic uniform across
staggerings — a deliberate readability-vs-memory trade.


Physical boundary conditions
----------------------------

Where ``kind == PHYS`` in a descriptor, a BC kernel runs.  GRACE
ships four BC kinds, declared as an enum in
``include/grace/data_structures/variable_properties.hh``:

.. code-block:: cpp

   enum bc_t : uint8_t {
       BC_OUTFLOW = 0,        // zero-gradient copy
       BC_LAGRANGE_EXTRAP,    // higher-order polynomial extrapolation
       BC_SOMMERFELD,         // outgoing-wave radiation BC, FD derivatives
       BC_NONE,               // no-op (variable is not extended into ghosts)
   } ;

Each evolved variable carries its choice of BC kind: the variable
list maintains a per-variable lookup
``Kokkos::View<bc_t*> var_bc_kind`` (and a face-staggered
counterpart ``var_bc_kind_f``).  The corresponding BC kernels live
in ``include/grace/amr/ghostzone_kernels/phys_bc_kernels.hh`` as
small functor structs:

- ``outflow_bc_t`` — zero-order extrapolation; copy the
  outermost interior cell value across all ghost cells.
- ``extrap_bc_t<N>`` — polynomial extrapolation of order ``N``;
  GRACE uses ``extrap_bc_t<3>`` (cubic Lagrange) in production
  when ``BC_LAGRANGE_EXTRAP`` is selected.
- ``sommerfeld_bc_t`` — Sommerfeld outgoing-wave BC implemented
  via finite-difference derivatives of the wave operator at the
  boundary.  Used for the Z4c metric fields.
- ``reflect_bc_t`` — overlay (see below).

There is no ``BC_DIV_PRESERVING`` enumerator: the divergence-
preserving treatment of staggered face-B at AMR interfaces is not
a *physical* BC but a *coarse–fine interface* operation, and it
runs through the prolongation / restriction kernels, not the
``phys_bc_kernels`` bucket.


Reflection symmetry is an overlay, not a BC choice
**************************************************

Reflection symmetry is treated separately from the four BC kinds
above.  It is controlled by the boolean parameters
``amr.reflection_symmetries.{x,y,z}``: whenever any of them is
``true``, the corresponding lower-side face of the domain becomes
a mirror plane and the ``reflect_bc_t`` kernel is applied *in
addition* to the per-variable BC kind.

Concretely, at any descriptor with ``kind == PHYS`` whose normal
points along a reflection-active axis at the lower side, the
reflection kernel mirrors interior values across the reflection
plane with a per-variable parity factor (scalar = +1, vector
components flip according to the axis direction, pseudovectors
such as ``B`` flip according to pseudovector parity).  The
parity factor is determined at variable-registration time.

For cell-centred variables the mirroring is across the cell-face
plane; for face-staggered variables there is a half-cell shift
(see :ref:`grace-variable-layout`) and the reflection kernel
applies ``ijk_s += (stag == STAG_FACEd ? 1 : 0)`` along the
reflection axis to put the mirrored index in the right place.
The pseudovector convention bit you on the BNS sim once already:
``B`` is even under z-reflection in the z-component, odd in the
others, opposite for true vectors.

The compositional order is fixed: physical-BC kernels (including
reflection) run first, then pack / MPI / unpack, then prolongation.
This sequence is enforced by the task graph and is not a knob.


Descriptor lifecycle
--------------------

Descriptors are built once per regrid and reused across all
substeps until the next regrid.  The lifecycle:

1. **Construction.**  After every regrid (and at startup),
   ``amr_ghosts_impl_t::update()``
   (``src/amr/amr_ghosts.cpp``) traverses the p4est forest and
   the p4est ghost layer (``p4est_ghost_t * p4est_ghost_layer``),
   filling a fresh ``quad_neighbors_descriptor_t`` for every local
   quadrant.  Same-rank vs remote, level-diff classification, and
   buffer-index assignment all happen here.
2. **Buffer allocation.**  ``build_remote_buffers()`` and
   ``build_coarse_buffers()`` size and allocate the send / receive
   / coarse buffer Views from the descriptor totals.
3. **Task graph construction.**  ``build_task_list<stag>()`` and
   ``build_task_list_face_stag<stag>()`` walk the descriptors and
   emit copy / pack / MPI / unpack / prolong / phys-BC tasks into
   the per-staggering task graph (see
   :ref:`grace-ghost-exchange-pipeline` for the operational view).
4. **Per-substep replay.**  ``apply_boundary_conditions()`` (in
   ``src/amr/boundary_conditions.cpp``) calls the executor, which
   walks the precomputed task graph.  No descriptor traversal is
   done at this point — all of that has been compiled into the
   graph.
5. **Teardown.**  The next regrid invalidates everything; the
   above starts over.

Two consequences worth internalising:

- **Anything cached against a descriptor is invalidated by
  regrid.**  Pointer-into-buffer optimisations, cached task ids,
  and so on must all be refreshed.
- **The per-substep cost is dominated by the task graph, not
  the descriptors.**  Tuning ghost exchange means tuning the
  pipeline in chapter 7, not the descriptor builder.


Pitfalls
--------

- **Descriptor staleness across regrid.**  Holding any descriptor
  pointer, quad id, or buffer id past a regrid call is a
  use-after-free in disguise.  The only safe way to retain
  ghost-related state across a regrid is by stable identifier
  (e.g. a Morton id) and to re-resolve to a fresh descriptor
  afterwards.
- **The union members are not interchangeable.**  ``full``,
  ``hanging``, and ``phys`` in ``face_data_t`` etc. share storage;
  reading the wrong one is undefined behaviour, not "zero".
  Always inspect ``kind`` and ``level_diff`` first, then access
  only the matching union arm.
- **Buffer envelopes are worst-case-sized.**  ``at_faces`` and
  friends do not bounds-check; if you compute an offset using the
  wrong staggering's strides, you will silently land in another
  variable's slot.  Always derive strides from the same
  ``set_strides()`` call that the buffer was built with.
- **The compositional order is reflection → BC → ghost copy → MPI
  → unpack → prolongation.**  Inserting work in the middle of
  this sequence (for example, a fix-up pass on the ghost layer
  before prolongation runs) breaks the divergence-cleaning
  guarantees on staggered B.  If you need an extra pass, put it
  before phys-BC or after prolongation, not between them.
- **Reflection parity is per-variable, not per-component.**  A
  vector field is represented by a contiguous run of three scalar
  variables; the parity for each component is registered
  independently at variable-registration time.  Forgetting to
  declare the parity for a new vector variable defaults it to
  ``+1`` for every component, which is wrong for any vector
  except a *pure* scalar triple.
- **Div-preserving prolongation assumes** ``ngz`` **is even.**
  The Tóth–Roe ghost-zone prolongation has a phase 1 loop with
  ``i_f = 2*i`` whose last-face check fires only when
  ``i_f == ngz - 2``; with odd ``ngz`` the boundary face is never
  filled and phase 2 reads uninitialised data.  GRACE uses
  ``ngz = 2`` or ``ngz = 4`` so the assumption is always
  satisfied, but it is a hidden dependency of the scheme.
- **Sommerfeld for staggered fields is technically wrong.**  The
  Sommerfeld kernel hard-codes cell-centre offsets
  (``ccoords = {0.5, 0.5, 0.5}``) when computing physical
  coordinates.  This is incorrect for face-staggered ``B``, but
  Sommerfeld is never applied to ``B`` in production — staggered
  ``B`` always uses ``BC_NONE`` plus the div-preserving
  interface treatment — so the latent bug is moot in practice.
  If you ever wire Sommerfeld to a face-staggered field, this is
  the first thing to fix.
