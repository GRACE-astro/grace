// performance_tuning/H100/tuning.h
//
// Tuning for NVIDIA H100 (sm_90, Hopper).  Selected when Kokkos is
// configured with Kokkos_ARCH_HOPPER90=ON.
//
// Register file: 64K 32-bit regs per SM, 255-reg/thread cap, 2048
// threads/SM max.  LaunchBounds<BLOCK, MIN_BLOCKS_PER_SM> on CUDA caps
// per-thread regs at 65536 / (BLOCK * MIN_BLOCKS_PER_SM).  The source
// default of GRACE_FLUX_LB = LaunchBounds<256, 2> was tuned for AMD
// (where 2 waves/EU lifts to 256 VGPR); on NVIDIA the same setting
// caps regs at 128, which ptxas rejects because getflux + WENO5/PPM
// reconstruction inlines to ~255 regs.
//
// Z4c kernels have the same problem documented at the source defaults.
// Lower MIN_BLOCKS_PER_SM to 1 across all heavy GRMHD / Z4c kernels so
// the 255-reg/thread cap kicks in.  Costs occupancy (12.5%), gained
// back as ptxas-successful builds.  Re-measure with nsight-compute and
// split lambdas further if higher occupancy is needed.

#define GRACE_FLUX_LB         Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_ADV_LB      Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_PRE_LB Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_LB     Kokkos::LaunchBounds<256, 1>
