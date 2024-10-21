# Goethe is a HIP machine

# Load specific system modules using the 'module load' command
execute_process(COMMAND bash -c "module load mpi/openmpi/5.0.5-rocm ucx/1.17.0 rocm"
                RESULT_VARIABLE MODULE_LOAD_RESULT)

if(NOT MODULE_LOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to load necessary system modules!")
endif()

# CMake flags
set(CMAKE_C_COMPILER "hipcc")
set(CMAKE_CXX_COMPILER "hipcc")

set(CMAKE_BUILD_TYPE "Release")

# Grace configs 
set(GRACE_NSPACEDIM 3)
set(GRACE_ENABLE_HIP ON)
set(GRACE_ENABLE_GRMHD ON)
set(GRACE_ENABLE_COWLING_METRIC ON)
set(GRACE_CARTESIAN_COORDINATES ON)

# Libs and dependencies
set(P4EST_ROOT  "/home/astro/musolino/libs/p4est-install-system-mpi")
set(CATCH2_ROOT "/home/astro/musolino/libs/catch2-install")
set(YAML_ROOT "/home/astro/musolino/libs/yaml-cpp-install")
set(KOKKOS_ROOT "/home/astro/musolino/libs/kokkos4-install")
set(VTK_ROOT "/home/astro/musolino/libs/VTK-install")
set(SPDLOG_ROOT "/home/astro/musolino/libs/spdlog-install")
set(HDF5_ROOT "/home/astro/musolino/libs/hdf5-install-system-mpi")
set(KOKKOS_TOOLS_ROOT "/home/astro/musolino/libs/kokkos-tools-install")



