// performance_tuning/generic.h
//
// Neutral / fallback tuning header.  Used on CPU backends (Serial, OpenMP,
// SYCL), on GPUs without a dedicated tuning header, and when the build
// system cannot detect the architecture.
//
// LaunchBounds template tags are silently ignored by the non-GPU Kokkos
// backends, so the values below remain correct when this header is active
// on CPU.
//
// On CUDA, the source defaults are AMD-tuned and ptxas rejects them
// (GRACE_FLUX_LB = LaunchBounds<256, 2> and GRACE_Z4C_ADV_LB =
// LaunchBounds<256, 4> both cap regs/thread below what the kernels
// inline to).  Since the generic header by definition has no
// arch-specific knowledge, the safest fallback is to *omit* the
// launch_bounds template parameter entirely — the policy templates
// then default to no __launch_bounds__ attribute on the kernel and the
// device compiler picks its own heuristic occupancy.  This guarantees
// the code compiles on any CUDA GPU.  Performance on a real target
// should be reclaimed by selecting a dedicated tuning header via
// -DGRACE_PERF_TUNING=<name>.
//
// The GRACE_NO_LB sentinel below is consumed by src/evolution/evolve.cpp:
// when defined, all flux + Z4c policies drop the LaunchBounds template
// argument entirely instead of passing Kokkos::LaunchBounds<0, 0>
// (whose behaviour is unspecified per NVIDIA's CUDA C++ Programming
// Guide, even though de-facto compilers treat it as a no-op).

#define GRACE_NO_LB
