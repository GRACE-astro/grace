#!/bin/bash

module load gcc
module load intel-oneapi-mpi
module load hdf5 
module load gsl

export GOOGLE_BENCHMARK_ROOT=/mnt/rafast/musolino/benchmark 
export P4EST_ROOT=/mnt/rafast/musolino/p4est-install 
export CATCH2_ROOT=/mnt/rafast/musolino/libs/Catch2-install-gnu
export YAML_ROOT=/mnt/rafast/musolino/libs/yaml-cpp-install-gnu
export KOKKOS_ROOT=/mnt/rafast/musolino/libs/kokkos-install-gnu-serial-omp