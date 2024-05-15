#!/bin/bash

export PATH=/mnt/rafast/relastro-shared/non-spack/rocm/ompi-install/bin/:${PATH}
export LD_LIBRARY_PATH=/mnt/rafast/relastro-shared/non-spack/rocm/ompi-install/lib/:${LD_LIBRARY_PATH}

export OMPI_CXX=hipcc 
export MPI_CXX=/mnt/rafast/relastro-shared/non-spack/rocm/ompi-install/bin/mpicxx

alias mpirun='mpirun --mca pml ucx --mca osc ucx \
              --mca coll_ucc_enable 1 \
              --mca coll_ucc_priority 100'

export P4EST_ROOT=/mnt/rafast/musolino/libs/p4est-install-hip-openmpi
export CATCH2_ROOT=/mnt/rafast/musolino/libs/Catch2-install-gnu
export YAML_ROOT=/mnt/rafast/musolino/libs/yaml-cpp-install-gnu
export KOKKOS_ROOT=/mnt/rafast/musolino/libs/kokkos-hip-rdc
export VTK_ROOT=/mnt/rafast/musolino/libs/vtk/install-gnu-openmpi
export SPDLOG_ROOT=/mnt/rafast/musolino/libs/spdlog-install
export KOKKOS_TOOLS_LIB=/mnt/rafast/musolino/libs/kokkos-tools-install/lib


export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}

export PATH=/mnt/rafast/musolino/libs/valgrind-install:/mnt/rafast/musolino/libs/kokkos-tools-install/bin:${PATH}
