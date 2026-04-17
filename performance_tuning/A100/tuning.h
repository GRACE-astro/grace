// performance_tuning/A100/tuning.h
//
// Tuning for NVIDIA A100 (sm_80, Ampere).  Selected when Kokkos is
// configured with Kokkos_ARCH_AMPERE80=ON.
//
// Register file: 64K 32-bit registers per SM, 255-reg/thread cap, 2048
// threads/SM max.  LaunchBounds<BLOCK, MIN_BLOCKS_PER_SM> here controls
// register budget indirectly via CUDA's __launch_bounds__.
//
// Not measured yet: left as stubs, fall through to source defaults for now.
// Re-measure with nsight-compute / nvdisasm and fill in.
