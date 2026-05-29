.. _grace-unit-testing:

The unit testing framework
==========================

GRACE's tests are organised as Catch2 binaries under ``test/``,
registered with CTest, and grouped into *labels* that select
subsets for different execution contexts.  The hot lane runs on
every push and pull request; the heavier MPI / boundary /
conservation lanes run locally and are required before merging
PRs that touch the matching machinery.  This chapter documents
the layout, the labels, the GitHub Actions CI, the
not-in-CI-but-required tests, and the conventions for adding new
tests cleanly.


The test directory
------------------

Tests live under ``test/`` with a flat structure:

::

   test/
   ├── CMakeLists.txt
   ├── configs/                  -- per-test YAML configs
   │   ├── basic_config.yaml
   │   ├── bc_test_fmr.yaml
   │   ├── checkpoint_roundtrip_test.yaml
   │   ├── ct_flux_conservation_test.yaml
   │   └── …
   ├── mains/                    -- per-framework main() files
   │   ├── kokkos_tests_main.cc
   │   ├── mpi_tests_main.cc
   │   ├── p4est_tests_main.cc
   │   └── parser_tests_main.cc
   └── test_*.cpp                -- the test files themselves

The per-test YAML configs are the smallest possible GRACE
configurations that exercise the relevant code path:
``checkpoint_roundtrip_test.yaml`` sets up a 4³-quadrant TOV grid
with FMR refinement at one corner so save/load touches all
descriptor kinds; ``bc_test_fmr.yaml`` builds the minimum mesh
that produces every (kind, level_diff) ghost-descriptor variant;
and so on.  The configs are deliberately small so the tests
finish in seconds.

The four ``mains/`` files are framework-specific entry points
that bring up the right singletons before Catch2 takes over:

- ``kokkos_tests_main.cc`` — initialises Kokkos.
- ``mpi_tests_main.cc`` — initialises MPI then Kokkos.
- ``p4est_tests_main.cc`` — initialises MPI, Kokkos, p4est.
- ``parser_tests_main.cc`` — initialises only the parameter
  store (for tests that exercise the schema validator without
  any compute backend).

A new test picks the smallest main that brings up everything it
needs.  Tests that touch the AMR machinery use the p4est main;
tests that only do single-cell c2p arithmetic use the kokkos
main; tests that read schemas only use the parser main.


Test labels and CTest selection
-------------------------------

Every test in ``test/CMakeLists.txt`` is registered via
``add_test(NAME … COMMAND mpirun -n N ./binary)`` and assigned one
or more CTest labels.  The labels group tests by execution
profile rather than by what they test:

==============  ==============================================
Label           Meaning
==============  ==============================================
``fast``        Single-host, seconds-each, no heavy MPI.  Run on every push by CI.
``mpi``         Multi-rank tests (typically ``-n 2`` to exercise the comm path).
``bc``          Boundary-condition and ghost-fill tests.  Multi-rank, mid-cost.
``conservation`` Bit-exact conservation / FP-floor tests for refluxing, CT, FOFC.
``symmetry``    Discrete-symmetry equivariance tests for Z4c, GRMHD, FD stencils.
``checkpoint``  Save / load / cross-rank-restart round-trip tests.
``regrid``      Regrid migration + state-preservation tests.
==============  ==============================================

A test can carry multiple labels.  The selection is done at
``ctest`` invocation time with ``-L <pattern>`` or
``-L "label1|label2"``.  The standard local-development invocations:

- ``ctest -L fast`` — the same set the CI runs.  Seconds.
- ``ctest -L "fast|mpi"`` — adds the multi-rank lane.
- ``ctest -L bc`` — the ghost-zone / boundary-condition lane.
- ``ctest`` (no filter) — everything that builds.  Minutes on a
  decent workstation.


Continuous integration
----------------------

The CI workflow lives in ``.github/workflows/ci.yml`` and runs on
every push to ``main`` and every pull request targeting ``main``.
It is intentionally a *narrow* lane: a single
``ubuntu-latest`` runner with the Serial Kokkos backend, full
Z4c + FOFC physics, the in-tree bundled dependencies, and
``ctest -L fast`` as the test step.

Concretely, the build is configured with:

.. code-block:: cmake

   -DGRACE_USE_BUNDLED_DEPS=ON
   -DGRACE_ENABLE_SERIAL=ON
   -DKokkos_ENABLE_SERIAL=ON
   -DKokkos_ENABLE_OPENMP=ON
   -DGRACE_METRIC_EVOL=Z4
   -DGRACE_ENABLE_FOFC=ON
   -DGRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS=ON
   -DGRACE_SAME_LEVEL_FLUX_AVERAGE=ON
   -DGRACE_ENABLE_TESTING=ON

and the test step is

.. code-block:: bash

   ctest -L fast --output-on-failure --schedule-random

The ``--schedule-random`` flag is deliberate: it shakes out any
hidden order-dependence between tests, which would otherwise
appear only when the suite happens to be re-ordered for unrelated
reasons.  ``--output-on-failure`` keeps the green-path log
short while still capturing diagnostics on any red.

On failure the workflow uploads the CTest temporary logs
(``LastTest.log``, ``LastTestsFailed.log``) as a GitHub artefact
so the source of a surprise can be inspected without
re-running locally.

The narrow CI is a deliberate choice.  The full test suite —
notably the ghost-fill, conservation, and symmetry lanes —
requires multi-rank execution and a meaningful build matrix
(Serial / OpenMP / CUDA / HIP), neither of which is cheap on
GitHub-hosted runners.  Those lanes are run locally before PRs;
see the next section.


Lanes not in CI — required before PRs
-------------------------------------

The CI does not run:

- the ``bc`` lane (ghost-zone filling tests)
- the ``conservation`` lane (CT-flux / FOFC / EMF reflux
  bit-exactness)
- the ``symmetry`` lane (Z4c / FD equivariance)
- the ``regrid`` lane (state migration round-trips)
- the ``checkpoint`` lane (save/load + cross-rank restart)

These are nevertheless **required to pass before a PR that
touches the corresponding code path is merged**.  In practice
this means:

- **PRs touching ghost-exchange, prolongation, restriction, or
  any boundary-condition code must run** ``ctest -L bc``
  **locally and report the result in the PR.**  These tests
  exercise every (kind, level_diff) descriptor variant on a
  small FMR grid and are the only thing that will catch a
  broken pack/unpack stride or a swapped recv/send descriptor
  before it lands.  Running them takes a few minutes on a
  workstation; running them is not optional.
- **PRs touching reflux, CT, or FOFC must run**
  ``ctest -L conservation``.  These tests assert bit-exact
  identity of conserved quantities across rank counts and FP
  floor for the per-cell residuals; they catch the
  non-associativity hazards documented in
  :ref:`grace-refluxing`.
- **PRs touching Z4c, the FD stencils, or the codegen pipeline
  must run** ``ctest -L symmetry``.  These confirm discrete
  symmetry equivariance under reflection at machine precision.
- **PRs touching AMR regrid, partition, or the prolongation
  kernels must run** ``ctest -L regrid`` **and**
  ``ctest -L checkpoint``.  Both exercise state migration and
  catch the most subtle classes of bugs (the alltoallv-inversion
  family and the post-load BC family).

Local execution is the right venue for these for two reasons
beyond cost.  First, the multi-rank tests want a meaningful
backend — running them in Serial on a CI runner exercises
neither the threading nor the device path that production uses.
Second, the developer who wrote the change is the right person
to triage a failure: the heavy lanes' failures are usually
informative rather than mysterious, and walking the stack from a
local run is much faster than uploading artefacts from a CI run.

If you are about to open a PR and you are unsure which lanes
apply, run ``ctest`` with no filter and include the elapsed time
in the PR description.  Reviewers will not require the full
matrix on every PR — they will require the lanes that matter for
the code under review.


Test conventions
----------------

Three conventions recur across the suite and are worth
internalising before writing new tests.

**Bit-exact assertions over FP-floor tolerances.**  When a
property is supposed to hold exactly — conservation across MPI
repartition, restart round-trip, symmetry under reflection of
equivariant inputs — the test asserts bit-exactness with
``CHECK(a == b)`` or its array-aware equivalent, not
``CHECK(std::abs(a - b) < tol)``.  Tolerance-based assertions are
reserved for genuinely approximate quantities (truncation-error
limits, convergence orders, EOS lookups with finite-table
precision).  This discipline is what catches the FP-non-associativity
hazards before they cause a paper retraction.

**Catch2 SECTION blocks re-run the whole test body.**
``SECTION`` is a Catch2-ism that lets a single ``TEST_CASE``
sample multiple sub-paths; it works by re-executing the entire
test body once per section.  For tests that involve singleton
setup (anything that goes through ``grace_initialize`` /
``grace_finalize``), the re-execution will attempt to
double-initialise singletons, trip the lifetime assertions
documented in :ref:`grace-singleton-holder`, and abort.  The
recipe is to use sequential ``CHECK`` calls instead of
``SECTION`` blocks for any state that crosses a setup boundary;
``test_checkpoint_roundtrip`` is the worked example.

**Tests own their YAML configs.**  Every test that needs a GRACE
configuration ships one under ``test/configs/`` named after the
test.  Sharing a config between tests is fine when they exercise
the same setup, but the config must be checked into the repo —
runtime-generated configs are forbidden because they make CI
failures un-reproducible.


Adding a new test
-----------------

The minimum recipe:

1. **Write the source** as ``test/test_<name>.cpp``.  Use Catch2's
   ``TEST_CASE`` / ``SECTION`` / ``CHECK`` / ``REQUIRE`` macros.
   Include the appropriate main from ``mains/`` via the
   ``add_executable`` setup in ``CMakeLists.txt``.
2. **Write the config** as ``test/configs/<name>_test.yaml``.
   Keep it the smallest mesh and physics set that exercises the
   property under test — seconds of runtime, not minutes.
3. **Register the binary and the test** in
   ``test/CMakeLists.txt`` using ``add_executable`` for the build
   target and ``add_test(NAME … COMMAND mpirun -n N ./<binary>)``
   for the CTest entry.  Assign labels with
   ``set_tests_properties(<name> PROPERTIES LABELS "<labels>")``.
4. **Choose the right labels.**  Use ``fast`` only if the test
   genuinely runs in seconds and is single-rank.  Multi-rank
   tests go in ``mpi``.  Anything that touches ghost descriptors
   or the BC kernels goes in ``bc``.  Bit-exactness-on-rank-count
   tests go in ``conservation`` or ``checkpoint``.  Symmetry
   tests go in ``symmetry``.

The CI will pick up the ``fast``-labelled subset automatically.
The non-fast labels are picked up by local invocations.


Pitfalls
--------

- **Singletons don't survive multiple Catch2 ``TEST_CASE`` calls
  in one binary without explicit re-initialisation.**  If you
  put more than one ``TEST_CASE`` in a single test binary,
  either each test must drive its own ``initialize`` /
  ``finalize`` cycle, or the binary must use the ``mains/``
  helper that sets up the singletons once and the
  ``TEST_CASE`` bodies must agree on the state they leave
  behind.  In practice most GRACE tests use a single
  ``TEST_CASE`` per binary for this reason.
- **``--schedule-random`` is a feature, not a nuisance.**  A
  test that passes on one ordering and fails on another has a
  hidden state dependency; the fix is in the test (or in the
  code it exercises), not in pinning the order.
- **Catch2's `SECTION` is not a fixture.**  See above.  Sequential
  ``CHECK`` calls are the right tool when the test body has
  irreversible side effects (a checkpoint write, a regrid call,
  any singleton state mutation).
- **The "fast" lane is not the bar.**  Passing ``ctest -L fast``
  is necessary but not sufficient for merging anything that
  touches AMR, ghost exchange, refluxing, or the C2P inverter.
  The lane that applies to the code under review must pass too.
- **Local multi-rank execution requires an MPI runtime.**  On
  developer machines without a system MPI, the heavy lanes will
  not run.  GRACE's bundled OpenMPI is the supported fallback;
  Homebrew's ``open-mpi`` works on macOS.  Tests that need
  more than the default ``-n 2`` declare so in their
  ``add_test`` invocation.
- **CI runs only with the Serial Kokkos backend.**  A test that
  passes in CI but fails on CUDA / HIP is not a CI bug, it is a
  test bug — typically a missing ``Kokkos::fence`` or a host-side
  read of a device-only view.  Run the heavy lanes against the
  same backend the failure was reported on before assuming the
  test itself is fine.
- **YAML config drift breaks tests silently.**  A new schema
  field added to a module with a non-trivial default will
  silently change the behaviour of every test config that
  doesn't explicitly set it.  Re-running ``ctest -L bc`` and
  ``ctest -L conservation`` after touching any parameter schema
  is a cheap insurance policy.


Wrap-up
-------

This is the closing chapter of the developer guide.  Across the
preceding chapters we have walked the static infrastructure
(singleton holder, config parser, p4est wrapper, variable
layout), the per-substep AMR machinery (ghost zones, ghost
exchange, refluxing), the supporting subsystems (codegen, I/O
and diagnostics), the dynamic AMR machinery (regrid,
checkpointing), and now the testing harness that holds it all
together.  A contributor who has read this guide should be able
to land a non-trivial change to any one of those subsystems
without re-deriving the conventions from the source.

The companion :doc:`../userguide/index` covers the runtime
configuration surface from the operator perspective; the
auto-generated Doxygen reference covers the public API.  When in
doubt about a subsystem not covered here in detail, the memory
notes under
``.claude/projects/-Users-cmusolino-grace-viper-grace-private-src/memory/``
preserve the diagnostic history of every bug that motivated a
design choice — they are not authoritative documentation, but
they are the highest-fidelity record of *why* the code looks the
way it does.
