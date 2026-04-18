// performance_tuning/A100/tuning.h
//
// Tuning for NVIDIA A100 (sm_80, Ampere).  Selected when Kokkos is
// configured with Kokkos_ARCH_AMPERE80=ON.
//
// Register file: 64K 32-bit registers per SM, 255-reg/thread cap, 2048
// threads/SM max.  LaunchBounds<BLOCK, MIN_BLOCKS_PER_SM> on CUDA sets
// __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM), which caps per-thread regs
// at  65536 / (BLOCK * MIN_BLOCKS_PER_SM).  A too-tight cap (e.g. 64 from
// <256, 4>) makes ptxas error out with
//   "Entry function ... with max regcount of 64 calls function ... with
//    regcount of 222"
// because the inner tile iterator (`exec_range`) needs more registers than
// the entry kernel is allowed to carry.
//
// Z4c advective kernel: NVCC's inliner on fill_deriv_*_upw +
// kreiss_olinger_operator lifts live regs to ~222 even though the body is
// slim.  Lowering MIN_BLOCKS_PER_SM to 1 gives 255 regs/thread — enough to
// compile, at the cost of occupancy (12.5%).  Bandwidth-bound kernel, so
// low occupancy is the lesser evil on A100.  Re-measure with nsight-compute
// and split the lambda further if we want more headroom.

#define GRACE_Z4C_ADV_LB      Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_PRE_LB Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_LB     Kokkos::LaunchBounds<256, 1>
