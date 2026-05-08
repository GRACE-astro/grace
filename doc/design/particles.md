# GRACE Particles — Design & Implementation Plan

**Status:** Draft, 2026-04-27 (revised 2026-04-28: Cabana dependency dropped)
**Author:** Carlo Musolino (with planning assistance)

## Dependency policy

The particle subsystem builds **directly on Kokkos and MPI** with no external
particle library. We previously sketched this on top of Cabana but dropped
that dependency for two reasons: (1) Kokkos ≥ 4.7 requires Cabana to be built
against `Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON` until Cabana ports its custom
AoSoA layouts to mdspan-based Views, with no public timeline; (2) post-ECP
Cabana maintenance is uncertain. The Cabana value-add for our v1 use case
(AoSoA storage + Distributor + HDF5 helper) is small enough that owning
~500 LOC of Kokkos+MPI primitives wins on long-term maintenance.

Build flag: `GRACE_ENABLE_PARTICLES=ON`. No external dependency beyond the
existing GRACE stack.

## Goals

Add a particle subsystem to GRACE. v1: passive Lagrangian tracers advected by the
fluid 3-velocity. The architecture must extend cleanly to:
- Monte Carlo radiation transport (per-particle RNG, scattering rates, source-term
  feedback to fluid)
- Particle-in-cell (shape functions, current deposition, force interpolation from
  staggered fields)

Key non-goals for v1: PIC current deposition, MC interaction kernels, particle
self-interactions, particle-AMR coupling beyond what migration requires.

## Non-negotiable requirements driving the design

1. **Performance-portable**: Kokkos memory spaces, GPU-resident particle storage,
   GPU-aware MPI for migration and aux fetch.
2. **Decoupled load balance**: particle ownership independent of fluid ownership.
   A fluid quad in a hot region (disk midplane, near horizon) does not impose its
   particle workload on the rank that owns its fluid.
3. **AMR-aware**: particles survive regrid (quad ids change → forced re-search +
   re-migration).
4. **Restart-safe**: checkpoint partition-independent; on load with a different
   `mpisize`, particles redistribute correctly.
5. **No fluid-side cost when particles disabled**: gated on
   `GRACE_ENABLE_CABANA` at compile time *and* on a runtime config flag.

## Architecture

### Storage

One `tracer_container_t<MemorySpace>` per rank — a struct of `Kokkos::View`s,
one per field (SoA layout). Sorted by `(global_quad_id, intra_quad_idx)`
where `global_quad_id` is the p4est SFC index of the owning quad. Because
p4est's SFC traversal is z-order, this sort *is* particle Morton order at the
quad's level — no separate position-Morton step needed.

SoA is sufficient for tracers (G2P touches ~3 fields per particle, push
touches ~5). For PIC kernels that touch ≥4 fields per particle in tight
loops, AoSoA gives a single-digit-percent improvement in published
benchmarks; we'd migrate then by replacing the per-field Views with a single
char-buffer + offset math, keeping the same accessor API. Phase 5
optimization, not a requirement.

Slices (v1) split into three blocks: identity/topology, advection inputs, and
hydro samples for output.

| Block | Field | Type | Purpose |
|---|---|---|---|
| identity | `pos[3]` | `double` | physical position |
| identity | `id` | `uint64` | immortal global id (preserved across migration/regrid/restart) |
| identity | `status` | `uint8` | `particle_status_flag_t` |
| topology | `owner_rank` | `int32` | cached owning fluid rank |
| topology | `owner_local_quad` | `int32` | cached owning fluid quad (local index on owner) |
| advection | `sample_alpha` | `double` | lapse |
| advection | `sample_beta[3]` | `double` | shift |
| advection | `sample_v[3]` | `double` | 3-velocity |
| advection | `sample_W` | `double` | Lorentz factor |
| hydro | `sample_rho` | `double` | rest-mass density |
| hydro | `sample_temp` | `double` | temperature |
| hydro | `sample_ye` | `double` | electron fraction |
| hydro | `sample_entropy` | `double` | specific entropy |
| hydro | `sample_press` | `double` | pressure |
| hydro | `sample_eps` | `double` | specific internal energy |
| hydro | `sample_B[3]` | `double` | 3-magnetic-field |

All sample fields are populated by the per-substep fetch protocol in a
**single** round-trip — fetch latency is dominated by MPI handshake, not
payload size, so adding sample fields costs zero substeps. The advection block
is what the RK update reads; the hydro block is what tracer output writes.

Source aux indices (see `data_structures/variable_indices.hh`): `RHO_`,
`ZVECX/Y/Z_` (→ `v^i` after dividing by `W`), `BX/Y/Z_`, `YE_`, `TEMP_`,
`ENTROPY_`, `EPS_`, `PRESS_`. Lapse + shift come from the metric arrays
(separate from aux).

Future fields (MC weight, RNG state, photon energy, ...) are added by
extending `tracer_member_types` or by introducing additional species AoSoAs.

### Partition

- Each rank holds a contiguous range of particle indices in the globally sorted
  array.
- Cuts are placed at `total_count · r / nproc` (perfect balance by particle count).
- Cuts may land mid-quad — that's allowed; multiple ranks can own particles from
  the same quad. They all fetch fluid aux from the (single) fluid owner of that
  quad.

### Per-substep protocol

```
1. fast-path bbox check (on-device)
   for each particle:
     if pos inside cached owner-quad bbox: keep cached (rank, quad), update
       base_cell_ijk via floor into local coords
     elif pos inside (cached owner-quad bbox + ghost halo): keep cached owner,
       flag for end-of-step slow-path re-search
     else: HARD FLAG (drift exceeded ghost width)

2. forward distribution_plan_t: ship "give me aux at (quad, position)"
   request POD tuples to fluid owners (MPI_Alltoallv under the hood, with
   GPU-aware MPI; same path GRACE uses for fluid halos)

3. fluid rank fulfills: kernel computes the trilinear stencil locally from
   per-quad geometry, reads 8 cell-centered values per requested field from
   its state/aux arrays, packs interpolated values into a response POD

4. reverse distribution_plan_t: response back to particle ranks

5. local trilinear G2P + RK substep update:
     dst_pos = src_pos + dtfact * dt * (alpha * v - beta)

6. (MC/PIC: deposition via reverse fetch — same protocol direction-flipped)
```

### End-of-step

```
- Slow-path re-search for flagged particles via the bbox shadow
  (host-side p4est_search_partition); update cached (rank, quad)
- Mark out-of-domain particles (status = PARTICLE_OUTSIDE_DOMAIN)
- Mark inside-BH particles via lapse threshold (status = PARTICLE_INSIDE_BH)
```

### Rebalance / redistribute

Triggered when **any** of:
- `max_local_count / mean_local_count > τ` (default τ = 1.3)
- Every N steps (config-driven safety net)
- Immediately after a regrid (quad ids changed; cached lookup invalid)

Steps:
1. `MPI_Allreduce` per-quad histogram → global counts vector
2. Local prefix-sum → cut points (deterministic, every rank gets same cuts)
3. Build `Cabana::Distributor`, migrate AoSoA
4. After regrid: forced full re-search of every cached `(rank, quad)`

### Drift regimes (decision matrix for fast-path / slow-path)

Let δ be per-substep position drift, ngz the fluid ghost depth.

| Regime | δ range | Policy | Cost |
|---|---|---|---|
| 1 | δ ≤ ngz − 0.5 | **Lazy**: keep cached owner; trilinear stencil reads through cached quad's ghost layer (refreshed by fluid BCs each substep). Flag only. | zero extra per substep |
| 2 | ngz − 0.5 < δ ≤ a few cells | **Eager**: inline slow-path search before next substep's fetch | one host lookup per crossed particle per substep |
| 3 | δ > "a lot" (≥ ½ quad-width) | **Hard error**: log + abort/freeze | — |

Tracers default to lazy (regime 1). MC/PIC default to eager (regime 2). Threshold
exposed as config.

### Boundary handling within a substep

If a particle's `dst_pos` ends up:
- **Outside global domain**: slow-path search returns no owner → `PARTICLE_OUTSIDE_DOMAIN`,
  freeze position, mask out of subsequent kernels.
- **Inside BH** (fetched α < threshold): `PARTICLE_INSIDE_BH`, freeze, mask out.

Both retain their slot in the AoSoA (preserves global id ordering for output) but
are skipped via a status mask in kernels.

### Fluid topology shadow

Each rank holds a replicated table:
- `global_first_quadrant[]` (already maintained by p4est)
- per-quad bbox `[xlo, xhi, ylo, yhi, zlo, zhi]` for every quad globally

Refreshed via `MPI_Allgatherv` after each regrid. Size: `~6·n_global_quads`
doubles. For 64 GPUs × 10k quads/rank ≈ 5 MB/rank — affordable.

Owner lookup for a point:
1. (Optional global pre-filter via Hilbert/Morton key range)
2. Brute or octree-walk through ranks' bboxes → candidate ranks
3. `p4est_search_partition` for the actual quad index

Host-resident initially. Move to device only if it shows up in profiles.

## Cadence summary

```
every substep:
  fast-path bbox check + cell-floor              [on-device]
  forward Distributor: aux requests
  fluid-side fulfillment                         [on-device on fluid ranks]
  reverse Distributor: aux responses
  G2P trilinear + RK push                        [on-device]
  (MC/PIC: deposit via reverse)

every step:
  slow-path re-search for flagged                [host]
  classify out-of-domain / inside-BH

every N steps OR imbalance OR after regrid:
  global per-quad histogram → cut points
  Cabana::Distributor migrate
  (after regrid: full re-search for everyone)
  refresh fluid bbox shadow
```

## File layout

**New:**
```
include/grace/particles/
  particles_module.hh         singleton + main API entry
  particle_storage.hh         tracer_container_t (SoA Kokkos::Views per field)
  particle_distributor.hh     distribution_plan_t (per-rank MPI plan + migrate())
  particle_advance.hh         per-substep RK push + G2P trilinear
  particle_owner_search.hh    fluid bbox shadow + p4est_search wrapper
  particle_aux_fetch.hh/.tpp  per-substep fetch protocol (templated on N_FIELDS)
  particle_migration.hh       end-of-step redistribute (uses distributor)
  particle_io.hh              HDF5 output + checkpoint hooks

src/particles/
  particles_module.cpp
  particle_distributor.cpp
  particle_advance.cpp
  particle_owner_search.cpp
  particle_migration.cpp
  particle_io.cpp
  CMakeLists.txt
```

**Modify:**
- `include/grace/utils/particle_utilities.hh` → keep as the shared types/enums
  header (status enum lives here)
- `src/evolution/evolve.cpp` → call particle advance per substep (gated)
- `src/amr/regrid.cpp` → post-regrid re-search + bbox-shadow refresh
- `src/IO/hdf5_output.cpp` + checkpoint handler → I/O hooks
- `src/grace.cpp` → init/teardown
- `src/CMakeLists.txt`, root `CMakeLists.txt` → register module, link Cabana
- `parameters/` → new `particles.yaml` schema
- `include/grace/grace_config.h.in` → already has `GRACE_ENABLE_CABANA`

## Phasing

| Phase | Scope | Validation |
|---|---|---|
| **0** | CMake plumbing + module skeleton; AoSoA hello-world; `GRACE_ENABLE_CABANA={ON,OFF}` both build clean | existing tests pass with OFF; new test creates 1k particles, fills positions, writes HDF5 |
| **1a** | Fluid topology shadow: per-rank bbox table, allgather on regrid, batched `p4est_search_partition` wrapper | static fluid + N injected query points; lookup matches brute-force search |
| **1b** | Aux-fetch protocol: forward+reverse `Cabana::Distributor`, pack/unpack kernels | fetch aux at positions identical to cell centers; round-trip values match originals |
| **1c** | RK-coherent advance via fetch protocol; trilinear G2P; hooked into `advance_substep` | single-rank case: tracers in a uniform flow track analytic streamlines |
| **2a** | Particle migration: per-quad histogram, cut points, `Cabana::Distributor` redistribute | multi-rank Bondi: tracer count conservation; trajectories smooth across rank boundaries |
| **2b** | Boundary handling: out-of-domain freeze, BH lapse-threshold detection; regrid hook | AMR test: particles near refinement boundary survive a regrid intact |
| **2c** | Drift regimes 2/3: eager slow-path inline; hard-error detection | inject high-velocity particles; verify no silent staleness |
| **3a** | HDF5 output via `Cabana::Experimental::HDF5ParticleOutput::writeTimeStep` + XMF | open output in ParaView, animate trajectories |
| **3b** | Checkpoint save/load via `co_tracker`-pattern block in `checkpoint_handler` | checkpoint mid-run, restart on different mpisize, ids+positions match |
| **4** | YAML config, parfile examples, validation tests (Bondi tracers, geodesic comparison in Schwarzschild) | parfile drives end-to-end |
| **5+** | MC: per-particle RNG, interaction rates, source feedback. PIC: shape functions, P2G/G2P consistency. | future |

## Open / undecided

- **Particle ghost halo**: do we need particles to be visible to neighbor ranks
  during the substep (for P2G in PIC)? For tracers v1: no. For PIC: probably yes —
  defer until Phase 5 design.
- **Per-species AoSoA vs single AoSoA with species tag**: defer until v2 (MC).
  Current design: one AoSoA per "particle species" registered with the module;
  v1 has a single hardcoded "tracers" species.
- **Fluid bbox shadow on device**: host-only is the default; revisit if profile
  shows the search dominating.
- **Output cadence (Phase 3 decision)**:
  1. *Snapshot* — HDF5 dump at output cadence; one row per particle including
     all sampled fields. Default; misses sub-cadence trajectory detail.
  2. *Trajectory* — per-particle ring buffer of K most-recent samples, flushed
     to per-particle datasets at output cadence. For nucleosynthesis post-proc.
     Memory cost K × ~150 B per particle; config knobs: ring depth, flush.
  3. *Append* — every substep, append one row per particle to a growing dataset.
     Simplest, biggest file footprint. Useful for short calibration runs.

## Risks

| Risk | Mitigation |
|---|---|
| Fetch latency dominates substep | Overlap with fluid flux compute; coalesce per-quad requests; fallback to local fetch when particle inside fluid-owned quad on this rank |
| Hot fluid quad serializes responses | Intrinsic; mitigation via ghost-fan-out is a future optimization |
| GPU-aware MPI failure modes | Already exercised by fluid halos; same path |
| Regrid + particle migration ordering bugs | Make particle re-search part of the regrid `update()` task graph, not a separate phase |
| SoA vs AoSoA perf gap at PIC time | Container API hides storage layout; Phase 5 swap to AoSoA is local change |
