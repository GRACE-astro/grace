.. _grace-variable-layout:

Variable layout and per-quadrant storage
========================================

The previous chapter described the topological skeleton — forest,
trees, quadrants — that p4est gives us, and noted that GRACE stores
field data in Kokkos ``View``\ s indexed by a flat quadrant id ``q``
rather than in p4est's per-quadrant ``user_data`` slot.  This
chapter describes that storage: the per-quadrant cell block, the
``View`` rank and layout, the staggered counterparts used for the
constrained-transport magnetic field, how physical coordinates are
recovered from logical indices, and the few sharp edges that
accumulate around all of this.


The cell block
--------------

Each leaf quadrant in the forest carries a fixed-size block of
cells.  The block edge length is the build-time / runtime constant
``npoints_block`` (read from the parameter store as
``amr.npoints_block_{x,y,z}``); the block is wrapped on every side
by ``ngz`` ghost cells (``amr.n_ghostzones``).  So a single
quadrant's data extents are:

.. code-block:: text

   nx_tot = npoints_block_x + 2 * ngz
   ny_tot = npoints_block_y + 2 * ngz
   nz_tot = npoints_block_z + 2 * ngz

The first ``ngz`` cells along each axis are the lower-side ghost
zone, the next ``npoints_block`` are the *interior*, and the last
``ngz`` are the upper-side ghost zone.  Every operation in GRACE
that loops over a quadrant uses the convention that interior
indices run from ``ngz`` to ``ngz + npoints_block``, with ghosts
addressed by stepping outside that range.

Two consequences are worth absorbing up front:

- **A quadrant occupies the same memory regardless of refinement
  level.**  Its physical size changes (halves on refinement), but
  the ``View`` extents do not.  The same kernel runs on a
  level-0 quadrant and on a level-8 quadrant.
- **Ghost zones are part of the quadrant.**  They are not stored in
  a separate auxiliary array; they live inside the same
  ``(nx_tot, ny_tot, nz_tot)`` block.  Filling them is the job of
  the ghost-exchange pipeline (see the ghost-exchange chapter).


The cell-centred View
---------------------

All cell-centred evolved and auxiliary variables live in a single
Kokkos ``View``:

.. code-block:: cpp

   using var_array_t = Kokkos::View<
       double*** /* (i,j,k) within a quadrant block */
              ** /* variable index, local quadrant index q */ ,
       Kokkos::LayoutLeft,
       default_space > ;

So the index order is ``(i, j, k, var, q)``.  Because the layout is
``LayoutLeft``, ``i`` is the fastest-running index in memory and
``q`` is the slowest.  This matches the cell-block stride pattern
that the flux loops and stencils want.

The cell-centred storage of a run is held by the variable-list
singleton, which actually carries **three** ``var_array_t`` Views,
not one:

.. code-block:: cpp

   auto& vars    = grace::variable_list::get() ;
   auto  state   = vars.getstate() ;     // evolved state
   auto  aux     = vars.getaux() ;       // auxiliary variables
   auto  scratch = vars.getscratch() ;   // second time-level / RK scratch

The separation reflects how the variables are used:

- **State** holds the variables that are advanced by the RHS at
  every substep (the evolved conservatives, the Z4c metric set,
  the magnetic-driver shift, etc.).  These are what RK schemes
  integrate, what gets checkpointed by default, and what physical
  conservation laws act on.
- **Auxiliaries** hold variables that are derived from the state
  and recomputed (or partially recomputed) each step: primitive
  variables from the c2p inversion, constraints violation (if Einstein equations are enabled)
  the cell-centred B-field reconstruction, c2p error bitmasks, and
  the various diagnostic fields written to disk.  Auxiliaries are
  not integrated; they are functions of the state at the current
  time, refreshed as needed.
- **Scratch** is the second RK time level, plus working space for
  flux-divergence updates and other transient storage.  It is the
  same shape as ``state`` and is allocated at all times to avoid
  realloc churn across substeps.

All three Views share the same ``(i, j, k, var, q)`` layout; they
differ only in the ``var`` dimension, which counts the number of
fields each one holds.

``state(i, j, k, var, q)`` is the value of variable ``var`` at cell
``(i, j, k)`` of the local quadrant with flat index ``q``, where
``q`` runs over the ``forest_impl_t::local_num_quadrants()`` leaves
this rank owns.  The mapping from ``q`` back to ``(tree, local
quadrant within tree)`` goes through ``forest_impl_t``:

.. code-block:: cpp

   auto& forest = grace::forest::get() ;
   for (size_t itree = forest.first_local_tree() ;
                itree <= forest.last_local_tree() ; ++itree) {
       auto tree = forest.tree(itree) ;
       size_t const q0 = tree.quadrants_offset() ;     // global q at start of this tree
       for (size_t iq = 0 ; iq < tree.num_quadrants() ; ++iq) {
           size_t const q = q0 + iq ;                  // flat index into state(...,q)
           auto quad = tree.quadrant(iq) ;
           // ... use state(...,q) and quad ...
       }
   }

In practice almost no GRACE kernel walks the forest by hand like
this — most use the Kokkos parallel constructs over ``q`` (and
``(i, j, k)``) directly, picking up the quadrant metadata from a
parallel device-side lookup.  The example is shown here only to
make the ``q``-to-quadrant mapping explicit.


Staggered storage for the magnetic field
----------------------------------------

The constrained-transport scheme requires the normal component of
**B** to live on cell faces rather than at cell centres.  GRACE
keeps face-staggered storage in a separate aggregate alongside the
cell-centred Views:

.. code-block:: cpp

   struct staggered_variable_arrays_t {
       var_array_t face_staggered_fields_x ;   // (nx_tot+1, ny_tot, nz_tot, var, q)
       var_array_t face_staggered_fields_y ;   // (nx_tot, ny_tot+1, nz_tot, var, q)
       var_array_t face_staggered_fields_z ;   // (nx_tot, ny_tot, nz_tot+1, var, q)

       var_array_t edge_staggered_fields_xy ;  // (nx_tot+1, ny_tot+1, nz_tot, var, q)
       var_array_t edge_staggered_fields_xz ;  // (nx_tot+1, ny_tot, nz_tot+1, var, q)
       var_array_t edge_staggered_fields_yz ;  // (nx_tot, ny_tot+1, nz_tot+1, var, q)

       var_array_t corner_staggered_fields ;   // (nx_tot+1, ny_tot+1, nz_tot+1, var, q)
   } ;

   auto& vars   = grace::variable_list::get() ;
   auto  sstate = vars.getstaggeredstate() ;

Three things to know about this storage.

**The shift is backwards by half a cell.**  An index ``i`` in
``face_staggered_fields_x`` refers to the face at physical position
``x_{i - 1/2}`` — i.e. the face between cells ``i - 1`` and ``i`` of
the cell-centred View.  The same convention applies to ``y`` and
``z``: a staggered index always points to the position half a cell
*below* the corresponding cell-centred index.  This is what the
"+1" in the staggered extents buys: with ``nx_tot`` cells in a row,
faces ``i = 0 … nx_tot`` cover both the lower-side face of cell 0
(at ``i = 0``) and the upper-side face of the last cell (at
``i = nx_tot``).

**Only the face-staggered slots are currently populated.**  In
production, GRACE uses ``face_staggered_fields_{x,y,z}`` for the
staggered magnetic field only — ``B^x`` on x-faces, ``B^y`` on
y-faces, ``B^z`` on z-faces.  The ``edge_staggered_fields_*`` and
``corner_staggered_fields`` members are present as storage slots
for completeness, but no physics module writes to them: GRACE has
no edge-centred or vertex-centred *evolved* quantity at present.
Edge-centred auxiliaries that one might expect to land in those
slots — fluxes and EMFs — actually live in their own dedicated
Views (``flux_array_t`` for the conservative fluxes,
``emf_array_t`` for the EMFs), allocated alongside but outside of
``staggered_variable_arrays_t``.  This separation keeps the
hot-path scratch storage and the persistent staggered fields on
distinct allocations with distinct lifetimes.

**The naming is direction-aware.**  A ``face_*_x`` field lives on
faces normal to ``x``; if the edge slots were ever used, an
``edge_*_xy`` field would live on edges parallel to ``z`` (the
axis *not* in the suffix), and ``corner_*`` would sit at quadrant
vertices.  The convention is stable even though most of these slots
are currently empty.


From logical indices to physical coordinates
--------------------------------------------

Recovering the physical position of cell ``(i, j, k)`` of quadrant
``q`` requires three pieces:

1. The interior cell index within the block,
   ``i_int = i - ngz``, ranging from 0 to ``npoints_block - 1``.
2. The quadrant's logical placement and extent within its parent
   tree, obtained from ``quadrant_t``:

   - ``quad.qcoords()`` — integer logical coordinates, in units of
     the quadrant's spacing.
   - ``quad.spacing()`` — the quadrant edge length in *logical*
     units, i.e. as a fraction of the parent tree's unit-cube
     extent.  This is ``1.0 / 2^level``.

3. The tree's *physical* extents, obtained from the connectivity:

   - ``conn.tree_coordinate_extents(itree)`` — the physical-space
     bounding box of the root cube.

Concretely:

.. code-block:: cpp

   // physical edge length of a single cell in this quadrant, per axis
   double const dx_phys_x = quad.spacing() * tree_size_x / npoints_block ;
   double const dx_phys_y = quad.spacing() * tree_size_y / npoints_block ;
   double const dx_phys_z = quad.spacing() * tree_size_z / npoints_block ;

   // physical position of the cell centre
   double const x = tree_x_min + ( quad.qcoords()[0] + i_int + 0.5 ) * dx_phys_x ;
   double const y = tree_y_min + ( quad.qcoords()[1] + j_int + 0.5 ) * dx_phys_y ;
   double const z = tree_z_min + ( quad.qcoords()[2] + k_int + 0.5 ) * dx_phys_z ;

GRACE provides helpers that fold these together — see
``include/grace/coordinates/`` for the canonical accessors.  The
important point is the structure: **the quadrant gives you a
logical fraction; the connectivity gives you the physical scale.**

This is also where the most common single mistake in the code base
lives.  ``coord_system::get_spacing(q)`` returns the *logical*
spacing ``1 / npoints_block`` of a cell within its parent tree, not
the physical ``dx``.  To get physical spacing, multiply by the
tree's physical extent:

.. code-block:: cpp

   auto& cs = grace::coord_system::get() ;
   double const dx_logical  = cs.get_spacing(q)[0] ;       // 1/n, NOT physical dx
   auto   tree_ext_x        = cs.get_tree_spacing(itree)[0] ;
   double const dx_physical = dx_logical * tree_ext_x ;    // this is what you want

For a cubic brick where every tree has the same extent on every
axis, the distinction is mostly cosmetic.  For a *non-cubic* brick
(say a slab much wider in ``x`` than in ``z``), getting this wrong
silently gives anisotropic spacing in any quantity that uses ``dx``
without the tree-extent multiplier — gradients, stencils,
interpolators, mass deposit.


The discrete update form
------------------------

A natural follow-up question to "where are cell volumes and face
areas?" is: how is the conservative update written without them?
GRACE evolves the **densitized** conservative variables.  For a
Cartesian quadrant the densitized variables and fluxes are

.. math::

   \tilde U   = \sqrt{\gamma}\, U,
   \qquad
   \tilde F^i = \sqrt{\gamma}\, F^i ,

so that the conservation law

.. math::

   \partial_t (\sqrt{\gamma}\, U) + \partial_i (\sqrt{\gamma}\, F^i)
       = \sqrt{\gamma}\, S

becomes simply

.. math::

   \partial_t \tilde U + \partial_i \tilde F^i = \tilde S ,

with all metric factors absorbed into the field definitions.  At
the discrete level this gives the familiar update

.. math::

   \tilde U^{n+1}_{i,j,k} =
       \tilde U^{n}_{i,j,k}
       - \frac{\Delta t}{\Delta x}\left(
            \tilde F^{x}_{i+1/2,j,k} - \tilde F^{x}_{i-1/2,j,k}
         \right)
       - \frac{\Delta t}{\Delta y}\left(
            \tilde F^{y}_{i,j+1/2,k} - \tilde F^{y}_{i,j-1/2,k}
         \right)
       - \frac{\Delta t}{\Delta z}\left(
            \tilde F^{z}_{i,j,k+1/2} - \tilde F^{z}_{i,j,k-1/2}
         \right)
       + \Delta t\, \tilde S_{i,j,k} ,

with ``Δx`` / ``Δy`` / ``Δz`` the *coordinate* spacings (uniform
within a quadrant, computed once from the tree extents and the
quadrant level — see the previous section).  No cell volume and no
face-area weighting appear, because the ``√γ`` factors that *would*
generate them are already inside ``Ũ`` and ``F̃ᵢ``.  The bookkeeping
is identical for the staggered constrained-transport step on the
face-staggered ``B`` field: the discrete induction equation uses
``Δx``, ``Δy``, ``Δz`` directly, with ``√γ`` carried by the
``F̃ᵢ`` already.

The earlier versions of GRACE carried explicit per-cell volume,
face-surface, and edge-length arrays as a hedge against
non-Cartesian grids.  Those arrays are no longer kept: the
densitized form makes them redundant on Cartesian quadrants, and
the wrapper currently supports only brick (Cartesian) connectivity.
If non-Cartesian connectivity is ever added, the volumes and
surface elements will return — but the place to add them is here,
in the per-quadrant precomputation, and the discrete update form
itself stays the same.


A tour of the variable-list Views
---------------------------------

Beyond the three Views introduced above (``state``, ``aux``,
``scratch``), the variable-list singleton owns a substantial set of
auxiliary storage that comes up routinely in kernels.  This section
is a flat catalogue so a developer can know what is available
without grepping the header.

**Coordinate spacings**

- ``getspacings()`` — a ``scalar_array_t`` keyed by ``(axis, q)``
  giving the logical cell spacing along each axis of quadrant
  ``q``.  Used by every stencil that needs ``1/Δx``.  Recall that
  this is the *logical* spacing; multiply by the tree extent for
  physical ``Δx``.
- ``getispacings()`` — the precomputed inverse, also keyed by
  ``(axis, q)``.  Provided separately because stencils that
  multiply by ``1/Δx`` once per cell prefer not to pay the divide
  on device; precomputing the inverse trades a small memory cost
  for a measurable speed-up in the FD-heavy Z4c and reconstruction
  kernels.

**Cell-centred state**

- ``getstate()`` — the evolved state (RK-integrated).
- ``getscratch()`` — second time level / RK working space, same
  shape as ``state``.
- ``getstagingbuffer()`` — a ``std::vector<var_array_t>`` of
  additional RK stage buffers.  The vector length depends on the
  active time-stepper (RK3 needs none, RK4 needs one, etc.).
- ``getaux()`` — auxiliary cell-centred variables (primitives,
  metric derivatives, c2p bitmasks, …).

**Staggered state**

- ``getstaggeredstate()`` — the face / edge / corner-staggered
  fields (see the staggered storage section); only the face slots
  are populated in production, holding the staggered ``B``.
- ``getstaggeredscratch()`` — second time level for the staggered
  fields, same shape as the state.
- ``getstagstagingbuffer()`` — staggered RK stage buffers, vector
  length tied to the time-stepper.

**Fluxes and EMFs**

- ``getfluxesarray()`` — the conservative fluxes ``F̃^x``, ``F̃^y``,
  ``F̃^z`` computed at cell faces by the Riemann solver and
  consumed by the cons-update kernel.  This is ``flux_array_t``
  (a separate allocation from the staggered state).
- ``getemfarray()`` — the EMFs at cell edges, used by the CT
  update of the staggered ``B``.  This is ``emf_array_t``.
- ``getvbararray()`` — the HLL-averaged transport velocities at
  cell faces (when the EMF scheme is UCT).
- ``getecarray()`` / ``getefarray()`` — cell-centred and
  face-staggered electric field (only present under the GS EMF
  scheme; conditional on ``GRACE_EMF_SCHEME``).

**FOFC scratch**

- ``getfofcfacetags()`` / ``getfofcedgetags()`` — 5D byte tag
  arrays (``int*****``) flagging which faces and edges have been
  marked for first-order fallback by ``flag_fofc_cells``.
- ``getfofcfx()`` / ``getfofcfy()`` / ``getfofcfz()`` /
  ``getfofceyz()`` / ``getfofcexz()`` / ``getfofcexy()`` —
  compacted index lists of the flagged faces and edges, populated
  atomically and consumed in linear order by
  ``apply_fofc_correction``.
- ``getfofcfcnt()`` / ``getfofcecnt()`` — small device-side
  counters used during the atomic compact step.

**Z4c curvature scratch** (under ``GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4``)

- ``getz4ccurvscratch()`` — per-cell scratch produced by the Z4c
  curvature-pre kernel, holding the Ricci tensor, Ricci trace, matter source terms and
  second Christoffel symbols.  Indexed in the same ``(i, j, k, var, q)``
  layout as ``state`` but with its own ``var`` enumeration
  (``z4c_curv_scratch_idx`` in ``variable_indices.hh``).

**Lifetime semantics**

Most of the above are allocated once and resized at every regrid
through ``realloc_state`` / ``resize_aux_staging_and_flux_buffers``;
they are not torn down across a regrid.  The exceptions are the
FOFC tag arrays and counters, which are ``deep_copy``-reset to zero
at the start of every substep before the dry-run kernel populates
them.

The ``allocate_state`` and ``allocate_staggered_state`` helpers
return *fresh* Views of the same shape, useful when a subsystem
needs its own temporary state-shaped buffer (for example for an
implicit solve, or a smoothing pass) without trampling the
production storage.


Adding a new variable
---------------------

Variables register themselves at startup.  The mechanism is a
free function ``register_variables()`` (in
``src/data_structures/variable_indices.cpp``) that runs through
each module and calls ``variable_list::register_variable(...)``
with:

- a string name (used in checkpoints, IO, log messages)
- the variable's centring (cell, face-x, face-y, face-z, edge-xy,
  edge-xz, edge-yz, corner)
- the variable's boundary-condition kind (outflow, Sommerfeld,
  Lagrange-extrap, divergence-preserving for staggered B, none)
- whether the variable is evolved, auxiliary, or persistent
  diagnostic

The registration call returns an integer index that the
registering module stores in a global ``enum`` (``RHO_``,
``BX_``, ``Z4C_THETA_``, …).  Downstream kernels then index
``state(..., RHO_, q)`` or ``sstate.face_staggered_fields_x(..., BX_, q)``
through that enum.

To add a new cell-centred variable to an existing module:

1. Pick an enum slot in the module's index header (e.g.
   ``include/grace/physics/grmhd_helpers.hh`` for GRMHD).
2. In ``register_variables()`` (or the module's section thereof),
   add a ``register_variable("name", cell_centred, bc_kind, ...)``
   call.
3. Update any IO descriptor lists that should expose the new field
   (volume / surface output, checkpoints).  Checkpoint inclusion is
   automatic for evolved variables; auxiliaries that you want
   restored on reload need to be flagged through the ``co_tracker``
   mechanism (see the checkpoint chapter).
4. Use it in your kernels via ``state(i, j, k, NEW_VAR_, q)``.

The same flow applies for staggered variables, with the centring
argument changed and the access going through the corresponding
member of ``staggered_variable_arrays_t``.


Pitfalls
--------

- **LayoutLeft vs LayoutRight**.  GRACE's ``var_array_t`` is
  ``LayoutLeft`` (``i`` is fastest, ``q`` is slowest).  Iterating
  in the wrong order will produce strided access that is correct
  but slow on GPU.  Use Kokkos's ``MDRangePolicy`` with default
  iteration patterns; do not transpose loops by hand.
- **Ghost zones included in the View extents**.  ``state.extent(0)``
  is ``npoints_block_x + 2 * ngz``, not ``npoints_block_x``.
  Interior loops should index ``[ngz, ngz + npoints_block)``, not
  ``[0, extent(0))``.
- **Logical vs physical spacing**.  ``coord_system::get_spacing(q)``
  is logical; multiply by ``get_tree_spacing(itree)`` for physical.
  See above.
- **Non-cubic bricks**.  All per-axis spacings are independent.  Do
  not assume ``dx == dy == dz`` anywhere — even on a cubic brick,
  defending against the assumption costs nothing.
- **Staggered "+1"**.  The staggered ``View`` extents differ from
  the cell-centred ones by ``+1`` on the staggered axis.  Loops
  over staggered fields must use the staggered extents, not the
  cell-centred ones, or they will read past the end of the storage.
- **Stale ``q`` after regrid**.  The flat quadrant index ``q``
  refers to a specific quadrant in a specific forest state.  After
  any regrid, partition, or refine/coarsen call, the ``q`` values
  are invalidated along with the ``tree_t`` / ``quadrant_t`` views.
  Code that holds a ``q`` across a regrid must re-resolve it (for
  particles this happens through the migration step; for diagnostic
  bookkeeping it is the developer's responsibility).
- **Index enum drift**.  The enum-based variable indices
  (``RHO_``, ``BX_``, …) are global integers; adding a new variable
  in the middle of an existing enum block shifts every subsequent
  index.  Append new variables at the end of the relevant block
  unless you are willing to chase the renumbering through every
  consumer.  Checkpoints written before such a renumber will be
  unloadable afterwards.
