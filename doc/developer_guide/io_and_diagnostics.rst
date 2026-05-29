.. _grace-io-and-diagnostics:

I/O and diagnostics
===================

GRACE writes output through three distinct channels (volume,
surface, spherical), and produces diagnostics through two distinct
patterns (a shared spherical-surface common interface, and one-off
classes that manage their own output).  This chapter documents
both halves of the system: the IO machinery that physics modules
plug into, and the design split that distinguishes "anything that
lives on a sphere" diagnostics from custom diagnostics that need
bespoke code.


The three output channels
-------------------------

GRACE has three logically distinct IO targets, each with its own
file convention, its own writer code, and its own user-facing
configuration block:

**Volume output** writes the full 3-D state of evolved and
auxiliary variables on the AMR mesh.  These are the big files,
suitable for visualisation and 3-D analysis.  Each rank writes its
own quadrants into one HDF5 file, and an XMF descriptor can be emitted alongside (see the GRACEpy documentation) so
ParaView / VisIt can read the result through a standard XDMF
reader.  Volume output is comparatively expensive and is usually
scheduled at coarse intervals.

**Surface output** writes the same variables, but restricted to
one or more *coordinate planes* (xy, xz, yz at user-specified
offsets).  This is the 2-D dump used for time-resolved plotting:
density evolution in the orbital plane, B-field topology in a
poloidal slice, and so on.  Surface output is cheap relative to
volume — only the cells intersected by the plane are written — and
typical production cadence is one surface dump every 100-500 iterationss.

**Spherical-surface output** writes data interpolated onto
2-spheres of user-specified radius, centred on user-specified
origins, sampled by a quadrature (equiangular only for now).  This
is the channel used by every diagnostic that integrates over a
2-sphere: ADM-mass integrals, gravitational-wave extraction at
detector radii, black-hole apparent-horizon quantities, outflow
fluxes through extraction spheres.  The output here is *not* the
sphere itself as a mesh — it is the *scalar time series* of the
integrated quantities, one ``.dat`` file per (quantity, detector)
pair.

The three channels share the same on-disk root (set by the
``IO.basepath`` parameter) and the same scheduling driver
(``output_diagnostics()``), but their writers, descriptors, and
on-disk formats are independent.


File formats and the XMF descriptor
-----------------------------------

The volume and surface channels write HDF5 with optional XMF
sidecars.  HDF5 carries the bulk data — one dataset per variable,
plus the coordinate arrays and connectivity — and the XMF (XDMF)
file is a small XML document that names the HDF5 datasets and
describes their geometric layout to the reader.  The XMF emitter
lives in ``include/grace/IO/xmf_utilities.hh``; you almost never
need to touch it directly. Note that emitting xmf directly from
``GRACE`` is quite fragile, since it relies on absolute paths of data
files. The official way of inspecting data is instead to generate
the XDMF headers by hand using the ``create_descriptor`` cli which is 
part of ``GRACEpy``. 

The spherical channel writes plain text ``.dat`` files, one per
``(flux_name, detector_name)`` pair, in the path
``runtime.scalar_io_basepath()``.  Each file has a small header
(``Iteration``, ``Time``, ``Value``) followed by one row per
output step.  The convention is fixed format width = 20 and
``std::fixed std::setprecision(15)`` — small enough to parse by
hand, precise enough that round-trip through ``np.loadtxt``
recovers the value to floating-point floor.

Lookup and scheduling go through the runtime singleton
``grace::runtime::get()``:

- ``scalar_io_basepath()`` — directory for ``.dat`` time series.
- ``volume_io_basepath()`` / ``surface_io_basepath()`` — HDF5
  output directories.
- ``iteration()`` / ``time()`` — current substep coordinates,
  used as the leading two columns of every scalar file.

All three channels are driven by ``output_diagnostics()``, a
free function in ``include/grace/IO/output_diagnostics.hh``,
called from the main loop after every evolution step.  It
consults the IO schedule for each channel and dispatches the
write only when the channel's interval condition is met.


How a variable becomes output
-----------------------------

Volume and surface output do not write every variable in
``variable_list``; they write only variables selected at runtime
by the ``IO.volume_output_cell_variables``, supplemented by some 
additional output (rank decomposition, quadrant id, quadrant level) which 
can be enabled via the ``IO.output_extra_quantities`` parameters.  The selection is by
name (a string) and is resolved against the per-variable
registration done at startup (see :ref:`grace-variable-layout`).


Note that staggered field output is not supported.  The
exception is checkpoints, which preserve the staggered storage
verbatim (see the checkpoint chapter).


The two diagnostic patterns
---------------------------

GRACE diagnostics split cleanly into two architectural shapes,
and the choice between them is determined entirely by *whether
the diagnostic lives on a sphere*:

**Spherical-surface diagnostics** integrate a flux through a
fixed 2-sphere (or evaluate a quantity at one).  Gravitational-wave extraction 
at coordinate detector radii, ADM-mass integrals, outflow fluxes through an
extraction radius, ejecta mass measurements — all of these share
exactly the same machinery: a centred sphere of given radius is
sampled by a quadrature, fields are interpolated to the sample
points, and a weighted sum produces one scalar per timestep per
flux.

**Custom diagnostics** are anything that doesn't have this shape:
mass accumulated across the outer boundary, NaN checks across the
whole grid, the FOFC firing counter, the c2p iteration histogram.
Each is a one-off class that manages its own output, its own MPI
reduction, and its own lifecycle.

There is no third pattern in current GRACE production code.  When
adding a new diagnostic, the right first question is "is this a
sphere integral?".  If yes, use the common interface; if no,
write a custom class.


The spherical-surface common interface
--------------------------------------

The common interface has three layers.

**The surface manager** (``IO/spherical_surfaces.hh``) owns every
2-sphere active in the run.  It is a singleton
(``spherical_surface_manager = utils::singleton_holder<spherical_surface_manager_impl_t>``)
populated at startup from the ``spherical_surfaces`` yaml block.
Each surface gets a string name (e.g. ``"AH"``, ``"detector_50"``,
``"extraction_100"``), a radius, a centre, a sampling resolution,
and is stored as a ``std::unique_ptr<spherical_surface_iface>``.

The manager exposes:

.. code-block:: cpp

   auto& mgr = grace::spherical_surface_manager::get() ;

   spherical_surface_iface&       s    = mgr.get(idx) ;       // by index
   spherical_surface_iface&       s    = mgr.get("AH") ;       // by name
   int                            idx  = mgr.get_index("AH") ; // -1 if missing
   mgr.update(mesh_changed) ;                                  // re-sample after regrid

**The surface itself** (``spherical_surface_t<SamplingPolicy,
TrackingPolicy>``) is a CRTP-style template parameterised on two
policy types:

- ``SamplingPolicy`` chooses how points are placed on the sphere
  (geodesic icosahedral, equiangular Gauss-Legendre, ...) and
  computes the quadrature weights.  The geodesic helper lives in
  ``IO/geodesic_sphere_helpers.hh``.
- ``TrackingPolicy`` chooses how the centre / radius evolve over
  time (static, AH-tracked, BH-co-moving, ...).  Each step the
  manager calls ``update_if_needed`` which delegates to the
  tracker; only if the tracker reports a change (or the mesh has
  changed) does the surface re-sample and re-compute interpolation
  weights.

The surface owns its sample points, weights, an
``octree_search`` that locates which p4est quadrant owns each
sample, and an ``interpolator`` that pre-computes interpolation
stencils from the owning quadrants' cell-centred storage to the
sphere sample positions.

**The diagnostic base class** (``IO/diagnostics/diagnostic_base.hh``)
is a CRTP template that turns "I have a list of spheres" into "I
have one ``.dat`` file per (flux, detector) and a write loop":

.. code-block:: cpp

   template <typename derived_t>
   struct diagnostic_base_t {
       diagnostic_base_t(std::string const& diag_name) ;  // read detector list from yaml
       void initialize_files() ;                          // open + header
       void write_fluxes() ;                              // append a row
       void compute_and_write() ;                         // resets, calls derived::compute(), writes
       std::vector<size_t> sphere_indices ;
       std::vector<std::vector<double>> fluxes ;
   } ;

The derived class must supply:

- ``static constexpr int n_fluxes`` — number of scalar integrals
  per surface.
- ``static constexpr std::array<const char*, n_fluxes> flux_names``
  — names used for filenames and column headers.
- ``void compute()`` — populate ``fluxes[i][j]`` for each active
  sphere ``i`` and each flux ``j``.  The base class calls this
  inside ``compute_and_write()`` after resetting the ``fluxes``
  array.

Concrete diagnostics that follow this pattern are in
``IO/diagnostics/``: ``black_hole_diagnostics.hh``,
``outflow_diagnostics.hh``, ``grmhd_diagnostics.hh``.  Each is a
~100-line class whose ``compute()`` does a host seerial reduction over
sample points using the surface's interpolator.

The base class also reads ``detector_names`` from the
diagnostic's own yaml block, resolves the names through the
surface manager's ``get_index()``, sorts and deduplicates, and
stores the resulting integer indices.  Missing surfaces produce
warnings, not errors — so a diagnostic listed in the yaml that
points to a surface no longer in the active list is silently
skipped.

Note that the integral performed by the diagnostic base class is over 
the angles, so that if the diagnostic being implemented ought to be 
a surface integral it has to carry the ``r^2`` measure explicitly in 
the flux definition.

Adding a new spherical-surface diagnostic
*****************************************

Concretely:

1. **Decide which fluxes you want.**  A spherical diagnostic
   produces one scalar per (sphere, flux) per step.  Count them:
   that's ``n_fluxes``.
2. **Create a new header** under ``IO/diagnostics/``, e.g.
   ``new_diagnostics.hh``.  Derive from
   ``diagnostic_base_t<new_diagnostics_t>`` and supply
   ``n_fluxes``, ``flux_names``, and ``compute()``.
3. **Inside ``compute()``**, use the sphere manager and the
   interpolator helpers to interpolate the fields you need from
   the variable list onto the sample points, then do a Kokkos
   reduction with the quadrature weights to produce the scalar.
4. **Register the diagnostic in the runtime** so
   ``output_diagnostics()`` calls ``compute_and_write()`` at the
   right cadence.
5. **Add a yaml schema** for the diagnostic with at least
   ``detector_names`` (a list of surface names declared in
   ``spherical_surfaces``).

You get the file management, MPI reduction, header writing, and
detector validation for free from the base class.  The only
physics-specific code is ``compute()``.


Custom diagnostics
------------------

When a diagnostic doesn't live on a sphere, the right pattern is
a singleton class that manages its own state and its own output
file.  ``boundary_outflow_t`` in
``include/grace/evolution/boundary_outflow.hh`` is the canonical
example and is worth reading as a template before writing a new
custom diagnostic.

The pattern has three pieces:

**1. A singleton with persistent accumulators.**

.. code-block:: cpp

   class boundary_outflow_t {
       Kokkos::View<double[N]> outflow_mass ;   // device-side accumulator
       /* … */
     public:
       static boundary_outflow_t& get() ;
       void accumulate(double dt, double dtfact) ;
       double flush_to_host() ;
   } ;

The accumulator lives on device — a single ``Kokkos::View`` of
the right shape — so the per-substep updates do not require any
host-device traffic.

**2. A per-substep call site that updates the accumulator.**

In the case of ``boundary_outflow_t``, that call site is in
``evolve.cpp`` between ``reflux_correct_fluxes`` and
``add_fluxes_and_source_terms``:

.. code-block:: cpp

   reflux_correct_fluxes(flux_context) ;
   boundary_outflow_t::get().accumulate(dt, dtfact) ;
   add_fluxes_and_source_terms(...) ;

This is the critical detail of any custom diagnostic: the call
site must be positioned in the loop *exactly* where the state it
reads is correct.  For a mass-conservation diagnostic that reads
the corrected flux, the call must be after ``reflux_correct_*``
and before the kernel that consumes the flux array (otherwise the
kernel's writes would alias your reads).

**3. A flush + write step at output time.**

When the output cadence triggers, the diagnostic flushes its
device accumulator to host, MPI-reduces across ranks, formats the
result, and appends one row to its ``.dat`` file:

.. code-block:: cpp

   double boundary_outflow_t::flush_to_host() {
       double host_val = 0.0 ;
       Kokkos::deep_copy(host_val, outflow_mass) ;
       parallel::mpi_allreduce_sum(host_val) ;
       return host_val ;
   }

The flush is rank-0-writes-the-file by convention.  The MPI
reduction handles the cross-rank aggregation; the file write
handles the format.

**The "sticky" pattern.**  Some diagnostics need to integrate
*across* output intervals: total mass crossed during one output
window, total c2p failures since last reset, etc.  For these the
accumulator is *not* reset on every flush; it is reset only when
a new output window opens, and the substep-level ``accumulate``
calls weight contributions by ``dt * dtfact`` so each RK substage
contributes correctly.  ``boundary_outflow_t`` uses this pattern.

Custom diagnostics share three idioms with the spherical pattern:

- **All file writes are rank-0 only.**  Per-rank file writes
  would multiply the output and require post-processing to
  reassemble; rank-0-after-reduction keeps the on-disk format
  simple.
- **All scalar output is appended to a ``.dat`` file**, one row
  per timestep, with leading ``Iteration`` and ``Time`` columns.
  This is by convention, not enforced by the code; following it
  keeps the post-processing scripts uniform.
- **The output directory is queried from
  ``runtime::scalar_io_basepath()``** rather than hard-coded.
  This lets the user redirect via yaml without touching any
  diagnostic code.


When to use which pattern
*************************

The decision is purely structural, not aesthetic:

- **Spherical pattern** if your diagnostic samples a 2-sphere (or
  a small set of them), evaluates a per-point integrand, and
  produces a quadrature integral.  AH-related quantities,
  GW detector channels, ADM integrals, mass-extraction fluxes.
  Cost: 100-line derived class.
- **Custom pattern** otherwise.  A scalar accumulator, a
  reduction over the whole grid, a histogram of cell counts, a
  per-rank tally that needs an MPI reduction with a different
  reduction operation than sum.  Cost: ~150-line class with its
  own singleton boilerplate.

When in doubt, ask whether your "detectors" have radii.  If yes,
use the spherical pattern; if no, write a custom class.


Pitfalls
--------

- **Don't write per-rank files for scalar output.**  Surface and
  volume HDF5 output is naturally per-rank, but scalar / time-
  series output must be rank-0-only.  A per-rank ``.dat`` produces
  ``N`` copies of the same time series after reduction, and any
  post-processing script will either crash or silently average
  them in some non-obvious way.
- **Custom diagnostic call-site positioning is load-bearing.**
  A diagnostic that reads the flux array before
  ``reflux_correct_fluxes`` sees the uncorrected flux and reports
  a spurious conservation violation at coarse–fine interfaces;
  one that reads after ``add_fluxes_and_source_terms`` sees a
  flux array that has been mutated by the consumer kernel.  The
  call site sits in a narrow window and a code reviewer must
  understand why.
- **The diagnostic base class silently skips missing detectors.**
  If your yaml lists a detector that wasn't registered in
  ``spherical_surfaces``, the base class warns and continues with
  the surfaces it found.  Output for the missing detector simply
  never appears.  Check the log for ``"Spherical detector …
  not found"`` if an expected ``.dat`` file is empty.
- **Sphere updates after regrid are non-trivial.**  Every
  registered sphere recomputes its interpolation weights when
  either the tracker reports a change *or* the mesh has changed.
  After every regrid the manager's ``update(true)`` is called and
  every sphere re-samples.  Diagnostics that hold cached state
  derived from sphere geometry must subscribe to that update
  pathway or be prepared to recompute.
- **Volume output dwarfs everything else in cost.**  At
  production HR resolution a single 3-D dump can be tens of GB
  per output and minutes of wall time including IO.  Schedule
  volume output sparingly; lean on surface (cheap) and spherical
  (cheaper) for time-resolved analysis.
- **Adding a new spherical diagnostic requires four files
  touched, not one.**  The derived class, its registration with
  ``output_diagnostics()``, its yaml schema entry, and the user
  yaml that names the actual detectors.  Forgetting the schema
  entry means the parameter parser will warn about unknown
  parameters; forgetting the runtime registration means
  ``compute()`` never runs; forgetting the yaml detector list
  means ``sphere_indices`` is empty and the diagnostic silently
  no-ops.  Verify in the startup log that the diagnostic
  appears with the expected detector count.
- **The on-disk ``.dat`` format is not a stable API.**  Adding
  a new column to an existing diagnostic breaks post-processing
  scripts that read by column index.  Either keep columns
  append-only (new columns to the right) or change the file name
  so old readers fail loudly.
