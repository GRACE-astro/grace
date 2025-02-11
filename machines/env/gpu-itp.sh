#!/bin/bash
export OMPI_ROOT=/mnt/rafast/musolino/libs/ompi-rocm-install/
export PATH=${OMPI_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${OMPI_ROOT}/lib:${LD_LIBRARY_PATH}
export PATH=/opt/rocm-6.2.1/bin/:${PATH}

export OMPI_CXX=hipcc 
export MPI_CXX=${OMPI_ROOT}/bin/mpicxx

alias mpirun='mpirun --mca pml ucx --mca osc ucx \
              --mca coll_ucc_enable 1 \
              --mca coll_ucc_priority 100'

export P4EST_ROOT=/mnt/rafast/musolino/libs/p4est-install-release
export CATCH2_ROOT=/mnt/rafast/musolino/libs/Catch2-install-gnu
export YAML_ROOT=/mnt/rafast/musolino/libs/yaml-cpp-install-gnu
export KOKKOS_ROOT=/mnt/rafast/musolino/libs/kokkos-hip-rdc
export VTK_ROOT=/mnt/rafast/musolino/libs/vtk/install-gnu-openmpi
export SPDLOG_ROOT=/mnt/rafast/musolino/libs/spdlog-install
export KOKKOS_TOOLS_ROOT=/mnt/rafast/musolino/libs/kokkos-tools-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib
export HDF5_ROOT=/mnt/rafast/musolino/libs/hdf5-rocm-install

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}

export PATH=/mnt/rafast/musolino/libs/valgrind-install/bin:${KOKKOS_TOOLS_ROOT}/bin:${PATH}
