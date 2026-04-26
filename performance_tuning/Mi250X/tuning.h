// performance_tuning/Mi250X/tuning.h
//
// Tuning for AMD MI250X (gfx90a, CDNA2).  Selected when Kokkos is
// configured with Kokkos_ARCH_AMD_GFX90A=ON.
//
// Starting point: inherit MI300A values (same 512 VGPR/SIMD layout, same
// occupancy tradeoffs).  Re-measure with the performance_tuning/Mi300A/disasm.sh
// workflow on gfx90a hardware and adjust if the PBQP allocator lands on
// different Pareto points.

#define GRACE_FLUX_LB Kokkos::LaunchBounds<256, 2>

#define GRACE_Z4C_ADV_LB      Kokkos::LaunchBounds<256, 4>
#define GRACE_Z4C_CURV_PRE_LB Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_LB     Kokkos::LaunchBounds<256, 1>
