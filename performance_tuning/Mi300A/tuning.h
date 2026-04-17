// performance_tuning/Mi300A/tuning.h
//
// Tuning for AMD MI300A (gfx942, CDNA3).  Selected automatically when Kokkos
// is configured with Kokkos_ARCH_AMD_GFX942=ON.  Each macro overrides the
// source-level default in evolve.cpp (all `#ifndef GRACE_* / #define ...`
// guarded).
//
// LaunchBounds<BLOCK, MIN_WAVES_PER_EU>: second arg caps VGPR/lane at
// 512/MIN_WAVES.  Smaller MIN_WAVES ⇒ more registers, lower occupancy.
//
// See performance_tuning/Mi300A/README.md for provenance / how to re-measure.

// ---- GRMHD flux kernels --------------------------------------------------
// WENO5 + HLL/LLF mix.  Compute-heavy (Riemann branches, smoothness
// indicators), not bandwidth-bound.  128 VGPR is the allocator's Pareto
// choice even with budget room, so the LB mostly documents the intent;
// combined with the PPLIM gating it yields the measured speedup.
#define GRACE_FLUX_LB Kokkos::LaunchBounds<256, 2>

// ---- Z4c RHS kernels -----------------------------------------------------
// advective : bandwidth-bound, want occupancy ⇒ 4 waves/EU.
// curv_pre  : 2nd derivatives heavy (ddgtdd_dx2[36], ddchi_dx2[6]) ⇒ 1 wave.
// curv      : still register-pressured after the split, keep 1 wave.
#define GRACE_Z4C_ADV_LB      Kokkos::LaunchBounds<256, 4>
#define GRACE_Z4C_CURV_PRE_LB Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_LB     Kokkos::LaunchBounds<256, 1>
