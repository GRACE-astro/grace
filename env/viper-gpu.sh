#!/bin/zsh

module load gcc/14 rocm/6.3 openmpi/5.0 hdf5-mpi
module load gsl mkl boost fftw-mpi

export LIB_BASEDIR=/u/cmusolino/grace-libs

export HOME_LORENE=/u/cmusolino/lorene/Lorene
export HOME_KADATH=/u/cmusolino/fuka

export LORENE_ROOT=${HOME_LORENE}
export TwoPunctures_ROOT=/u/cmusolino/Standalone-TwoPunctures-C-Cpp

export KOKKOS_TOOLS_ROOT=${LIB_BASEDIR}/install/kokkos-tools
export KOKKOS_ROOT=${LIB_BASEDIR}/install/kokkos
export P4EST_ROOT=${LIB_BASEDIR}/install/p4est
export YAML_ROOT=${LIB_BASEDIR}/install/yamlcpp
export CATCH2_ROOT=${LIB_BASEDIR}/install/catch2
export SPDLOG_ROOT=${LIB_BASEDIR}/install/spdlog

export KOKKOS_TOOLS_PATH=${KOKKOS_TOOLS_ROOT}/lib64

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin
