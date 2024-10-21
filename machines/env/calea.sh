#!/bin/bash

module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-mkl
module load gsl 

export LIBXSMM_ROOT=/mnt/rafast/musolino/libs/libxsmm 
export GOOGLE_BENCHMARK_ROOT=/mnt/rafast/musolino/benchmark 
export P4EST_ROOT=/mnt/rafast/musolino/p4est-install 
export CATCH2_ROOT=/mnt/rafast/musolino/libs/Catch2-install
export BLAZE_ROOT=/mnt/rafast/musolino/libs/blaze-install
export YAML_ROOT=/mnt/rafast/musolino/libs/yaml-cpp-install
export KOKKOS_ROOT=/mnt/rafast/musolino/libs/kokkos-install-intel