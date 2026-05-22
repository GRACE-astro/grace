.. _grace-userguide:

User Guide
==========

This page is the runtime reference for GRACE: parameter schemas grouped
by module, variable name conventions, the grid model, and output data
formats.  For build-time options see the
:doc:`building guide <../code_building_guide/index>`; for an end-to-end
walkthrough that gets you from a clone to your first simulation, see the
:doc:`quickstart <../quickstart/index>`.


Input parameters
*****************

GRACE reads `YAML <https://en.wikipedia.org/wiki/YAML>`__ parameter files
split into sections — one per module.  The reference pages below list
every supported key, its type, default, and allowed range, grouped by
the module that owns it.

.. toctree::
   :maxdepth: 1
   :caption: Code Modules

   params-amr
   params-system
   params-IO
   params-evolution
   params-grmhd
   params-eos
   params-z4c
   params-coordinate_system
   params-checkpoints
   params-spherical_surfaces
   params-bh_diagnostics
   params-co_tracker
   params-mhd_diagnostics
   params-outflows
   params-gw_integrals
   params-adm_integrals
   params-nan_check
   params-b_field_injection
   params-m1
   params-particles


Output Data Formats 
*********************

There are a few different kinds of output in GRACE. 

First, there is scalar output. This encompasses reductions of any registered variable (auxiliary or evolved) in GRACE.
Possible reductions are: coordinate volume integral, L2 norm, max and min. The output frequency of all scalar quantities is controlled by a single parameter, 
and all output can be found in the same directory. Output files contain three columns: iteration, simulation time, and the value. 

The second kind of output is also in the form of a timeseries and comprises all spherical surface integrals. These are implemented 
as custom modules in GRACE which have their own parameters. One example is the ``gw_integrals`` module, which simply takes the names 
of registered surfaces where integrals should be performed, and outputs the spherical harmonic decomposition of the Penrose-Newman scalar 
onto these surfaces. The output frequency here is controlled by the diagnostic output frequency, and the corresponding files are placed in 
the same directory as scalars. 

Other outputs are performed in HDF5 by default, with the legacy (deprecated) option of native VTK. GRACE supports volume and plane surface output (in xy, xz, and yz planes), as
well as output of point data on spherical surfaces. All this data can be easily visualized in `ParaView <https://www.paraview.org/>`_ through the use of XDMF descriptors (see the :doc:`related page <../python_interface/index>` on how to generate them),
as well as in python through the vtk Python interface.

Reflection symmetries and diagnostic output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``amr::reflection_symmetries`` is enabled along one or more axes, GRACE only stores and evolves the active subdomain (e.g. the upper hemisphere :math:`z \geq 0` for :math:`z` reflection symmetry).
For diagnostics that are written to dedicated output files, namely the ones produced by the ``mhd_diagnostics`` module (``disk_mass.dat``), the magnetic energy diagnostic (``E_em.dat``),
the ``outflows`` module (``Mdot_*.dat``), and the ``gw_integrals`` / ``adm_integrals`` spherical decompositions, GRACE automatically rescales the reported values to the
full physical domain. The user therefore does not need to apply any post-processing correction: the numbers in these files are already full-domain physical values.

This is correct as long as the integrand is invariant under each active reflection axis (rest mass, energies, fluxes through equatorially-symmetric surfaces, :math:`J_z` under :math:`z`-reflection, etc.), which covers all the named-file diagnostics shipped with GRACE.
Generic scalar reductions (volume integrals, L2 norms, max/min) requested through the registered-variables mechanism are **not** corrected: their values reflect the integral over the active subdomain only. If the integrand is parity-even, multiply
by :math:`2^{N}` in post-processing, where :math:`N` is the number of active reflection axes; if it is odd under any active axis, the physical value is zero by symmetry.

Variable names and definitions
*******************************

GRACE evolves the equations of GRMHD on static or dynamical curved backgrounds.
The MHD equations are written in Heaviside-Lorentz units.

GRMHD conserved variables
~~~~~~~~~~~~~~~~~~~~~~~~~

The conserved variables follow the standard Valencia 3+1 formulation:

- ``dens`` — conserved rest-mass density
- ``tau`` — conserved energy
- ``stilde[0..2]`` — conserved momentum density

The magnetic field is evolved separately as a **face-staggered, densitized
magnetic field** that lives on cell faces and is updated through the
constrained-transport (CT) machinery; divergence-free preservation is by
construction.  Cell-centered values used by the c2p inversion and the
reconstruction (``Bvec[0..2]``, see primitives below) are obtained from
the face-staggered field by averaging.

GRMHD primitives
~~~~~~~~~~~~~~~~

- ``rho`` — rest-mass density
- ``press`` — pressure
- ``temperature`` — temperature
- ``eps`` — specific internal energy
- ``entropy`` — entropy per particle
- ``ye`` — electron fraction
- ``zvec[0..2]`` — the velocity-like variable :math:`z^i = W v^i`, where
  :math:`W` is the Lorentz factor and :math:`v^i` the standard Eulerian
  velocity.  GRACE stores :math:`z^i` (not :math:`v^i`) because it removes
  the unbounded :math:`W` singularity from primitive recovery.
- ``Bvec[0..2]`` — **cell-centered** magnetic field, derived from the
  face-staggered evolved field by face-to-cell averaging.

Auxiliary diagnostics
~~~~~~~~~~~~~~~~~~~~~

- ``Bdiv`` — discrete divergence of the magnetic field on the cell-centered
  grid, computed from the face-staggered field.  By construction this is
  zero to round-off when CT is used; non-zero values flag CT pipeline bugs
  rather than physics.
- ``c2p_err`` — packed bit-pattern (stored as a double, decoded via integer
  cast) recording which conservative-to-primitive failure modes fired in a
  given cell during the timestep.  The bits are sticky-OR'd over each step
  and reset at the start of the next one:

  - bits 0-4 — which conserved variables were rewritten
    (``dens`` / ``tau`` / ``stilde`` / ``entropy`` / ``ye``)
  - bits 5-18 — raw signal mirrors: which c2p or EOS error bit fired
    (rho out of range, eps out of range, NR failed to converge, ...).
    Purely diagnostic.
  - bit 19 — the entropy-backup c2p path saved a distrusted primary
    inversion.
  - bit 20 — the cell fell through to a full atmosphere reset
    (most severe).
  - bit 21 — T-only floor: rho was fine but T fell below the atmosphere
    temperature; T (and derived eps, press, entropy) were reset while
    velocity was preserved.

  The full bit layout and a Python decoding recipe live in
  ``include/grace/physics/eos/c2p.hh``.

- ``c2p_dens_corr`` — signed mass-error accumulator: the per-cell sum of
  conserved-density adjustments applied by the c2p flooring logic over the
  timestep.  Tells you *how much* mass was added/removed by the c2p, not
  just whether a reset fired.

Spacetime variables (Cowling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the metric is not evolved, GRACE stores the physical 3+1 fields
directly:

- ``gamma[i,j]`` — spatial metric (6 components)
- ``ext_curv[i,j]`` — extrinsic curvature (6 components)
- ``beta[0..2]`` — shift vector
- ``alp`` — lapse function

Spacetime variables (Z4c)
~~~~~~~~~~~~~~~~~~~~~~~~~

When the metric is evolved (``GRACE_METRIC_EVOL=Z4``), GRACE evolves the
Z4c form, storing the conformal fields:

- ``gamma_tilde[i,j]`` — conformal spatial metric
- ``A_tilde[i,j]`` — conformal traceless extrinsic curvature
- ``conf_fact`` — conformal factor (W)
- ``z4c_Khat`` — trace of extrinsic curvature minus the Z4c constraint
- ``z4c_theta`` — Z4c constraint-propagating degree of freedom
- ``z4c_Gamma[i]`` — contracted conformal connection (BSSN-style
  :math:`\tilde\Gamma^i`)
- ``z4c_Bdriver`` — hyperbolic gamma-driver auxiliary
- ``alp``, ``beta[i]`` — lapse and shift, as in the Cowling case

Z4c diagnostics:

- ``z4c_H``, ``z4c_M[i]`` — Hamiltonian and momentum constraint violations
- ``PsiRe``, ``PsiIm`` — real and imaginary parts of the Weyl scalar
  :math:`\Psi_4` used for GW extraction.  Not updated each step; computed
  on demand when diagnostic output fires.


Grid setup
**********

The grid in GRACE is represented as a *forest of oct-trees*.  Each tree is
a cube in Cartesian coordinates, recursively subdivided into eight children
per refinement; the leaves of the trees — called *quadrants* by ``p4est``
convention (technically ``octants`` in the 3D ``p8est``) — form the actual
grid.  Each quadrant is a regular grid block of equally spaced cells
plus a ghost-zone halo on every side, on which differential operators are
evaluated locally.

Operationally, GRACE splits the user-requested domain into trees of size
equal to the smallest grid extent along any axis, stacking equal-shaped
trees along the longer axes.  For example, if the domain spans
:math:`(x,y,z) \in [-1024, 1024] \times [-1024, 1024] \times [0, 1024]`,
each tree has side :math:`1024` and two trees are stacked along
:math:`x` and :math:`y`.

After the trees are set up, the grid is initialized at a uniform refinement
level :math:`l` controlled by ``amr::initial_refinement_level``, so each
tree contains :math:`2^l` quadrants per direction.  Continuing the example,
:math:`l = 1` gives quadrants of side :math:`512`; with
``amr::npoints_block_x = 32`` the resulting base cell size is
:math:`1024 / 2 / 32 = 16`.

On top of this uniform base grid GRACE supports an arbitrary number of
**fixed mesh refinement (FMR) boxes**, each specified by coordinate
extents; every level inside an FMR box halves the grid spacing.

Finally, the grid can be adapted during the simulation with **AMR**.  The
base grid (uniform + FMR) is never touched by AMR; AMR can only add
quadrants on top of it (or remove ones it previously added).  AMR is
controlled by:

- ``amr::regrid_every`` — regrid frequency in iterations
- the refinement criterion, which produces a per-quadrant error estimate
  :math:`\epsilon(q)`
- ``amr::refinement_criterion_CTORE`` — refine if :math:`\epsilon(q)`
  exceeds this threshold
- ``amr::refinement_criterion_CTODE`` — coarsen if :math:`\epsilon(q)`
  falls below this threshold

For runs that benefit from extra refinement already at the initial-data
stage, ``amr::regrid_at_postinitial`` plus
``amr::postinitial_regrid_depth`` apply one or more AMR passes immediately
after IC setup before the evolution loop begins.
