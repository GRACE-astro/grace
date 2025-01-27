# CMake flags
# Set compilers only if not already specified by the user
set(CMAKE_C_COMPILER "hipcc" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE)

set(GRACE_ENABLE_HIP ON)

# Libs and dependencies
set(P4EST_ROOT  "/mnt/rafast/musolino/libs/p4est-install-release")
set(CATCH2_ROOT "/mnt/rafast/musolino/libs/Catch2-install-gnu")
set(YAML_ROOT "/mnt/rafast/musolino/libs/yaml-cpp-install-gnu")
set(KOKKOS_ROOT "/mnt/rafast/musolino/libs/kokkos-hip-rdc")
set(VTK_ROOT "/mnt/rafast/musolino/libs/vtk/install-gnu-openmpi")
set(SPDLOG_ROOT "/mnt/rafast/musolino/libs/spdlog-install")
set(HDF5_ROOT "/mnt/rafast/musolino/libs/hdf5-rocm-install")
set(KOKKOS_TOOLS_ROOT "/mnt/rafast/musolino/libs/kokkos-tools-install")