// performance_tuning/generic.h
//
// Neutral / fallback tuning header.  Intentionally empty: every perf macro
// guarded by `#ifndef GRACE_*` in the sources will fall through to its
// source-level default.  This is what you get on CPU backends (Serial,
// OpenMP, SYCL), on GPUs without a dedicated tuning header, and when the
// build system cannot detect the architecture.
//
// LaunchBounds template tags are silently ignored by the non-GPU Kokkos
// backends, so source defaults that mention LaunchBounds<256, N> remain
// correct when this header is active on CPU.
