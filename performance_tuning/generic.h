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
// On CUDA, the source default of GRACE_Z4C_ADV_LB = LaunchBounds<256, 4>
// caps per-thread regs at 64, which ptxas rejects because the Z4c
// advective lambda inlines ~222 regs of stencil state.  Override here
// with <256, 1> so the generic path still *builds* on unknown CUDA GPUs;
// performance on a real target should be reclaimed with a dedicated
// tuning header selected via -DGRACE_PERF_TUNING=<name>.

#define GRACE_Z4C_ADV_LB      Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_PRE_LB Kokkos::LaunchBounds<256, 1>
#define GRACE_Z4C_CURV_LB     Kokkos::LaunchBounds<256, 1>
