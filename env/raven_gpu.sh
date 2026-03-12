#!/bin/bash

module load cmake/4.0 cuda/13.0 gcc/15 openmpi/5.0 openmpi_gpu/5.0 hdf5-mpi/1.12.2 mkl mkl_parts-mpi/1 gsl

export LIB_BASEDIR=/u/cmusolino/grace-libs

export HOME_LORENE=/u/cmusolino/lorene/Lorene
export LORENE_ROOT=/u/cmusolino/lorene/Lorene

export TwoPunctures_ROOT=/u/cmusolino/Standalone-TwoPunctures-C-Cpp

export KOKKOS_TOOLS_ROOT=${LIB_BASEDIR}/install/kokkos_tools
export KOKKOS_ROOT=${LIB_BASEDIR}/install/kokkos
export P4EST_ROOT=${LIB_BASEDIR}/install/p4est
export YAML_ROOT=${LIB_BASEDIR}/install/yamlcpp
export CATCH2_ROOT=${LIB_BASEDIR}/install/catch2
export SPDLOG_ROOT=${LIB_BASEDIR}/install/spdlog

export KOKKOS_TOOLS_PATH=${KOKKOS_TOOLS_ROOT}/lib64

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin
export PATH=${PATH}:${KOKKOS_ROOT}/bin
