#!/bin/bash
# ==========================
# Environment setup for Grace
# ==========================
ml OpenMPI/5.0.5-NVHPC-24.3-CUDA-12.3.0
# --------------------------
# Compilers
# --------------------------
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx
export CC=mpicc
export CXX=mpicxx

# --------------------------
# Library roots
# --------------------------
export GRACE_LIBS="/mnt/proj3/open-30-28/Kpierre/libs"

export KOKKOS_ROOT="$GRACE_LIBS/kokkos-install"
export YAML_ROOT="$GRACE_LIBS/yaml-cpp-install"
export SPDLOG_ROOT="$GRACE_LIBS/spdlog-install"
export P4EST_ROOT="$GRACE_LIBS/p4est-install"
export CATCH2_ROOT="$GRACE_LIBS/Catch2-install"
export HDF5_ROOT="$GRACE_LIBS/hdf5-install"
export VTK_ROOT="$GRACE_LIBS/VTK-install"

# --------------------------
# CMake discoverability
# --------------------------
# This lets CMake automatically find your installed libraries
export CMAKE_PREFIX_PATH="$KOKKOS_ROOT:$YAML_ROOT:$SPDLOG_ROOT:$P4EST_ROOT:$CATCH2_ROOT:$HDF5_ROOT:$VTK_ROOT"

# --------------------------
# Architecture tweaks
# --------------------------
# Convert old-style zen2 flags if present
while read -r e; do
    NAME=$(cut -d= -f1 <<< "$e")
    VALUE=$(printenv "$NAME")
    export $NAME="${VALUE/tp=zen2/march=znver2}"
done < <(env | grep 'tp=zen2')

echo "Environment for Grace setup loaded."

