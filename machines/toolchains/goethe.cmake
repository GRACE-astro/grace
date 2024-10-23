# CMake flags
# Set compilers only if not already specified by the user
set(CMAKE_C_COMPILER "hipcc" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE)

set(GRACE_ENABLE_HIP ON)

# Libs and dependencies
set(P4EST_ROOT  "/home/astro/musolino/libs/p4est-install-system-mpi")
set(CATCH2_ROOT "/home/astro/musolino/libs/catch2-install")
set(YAML_ROOT "/home/astro/musolino/libs/yaml-cpp-install")
set(KOKKOS_ROOT "/home/astro/musolino/libs/kokkos4-install")
set(VTK_ROOT "/home/astro/musolino/libs/VTK-install")
set(SPDLOG_ROOT "/home/astro/musolino/libs/spdlog-install")
set(HDF5_ROOT "/home/astro/musolino/libs/hdf5-install-system-mpi")
set(KOKKOS_TOOLS_ROOT "/home/astro/musolino/libs/kokkos-tools-install")