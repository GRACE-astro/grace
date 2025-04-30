#!/bin/bash
module load mpi/openmpi/5.0.5-rocm ucx/1.17.0 rocm
export PATH=${ROCM_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

export OMPI_CXX=hipcc 

alias mpirun='mpirun --mca pml ucx --mca osc ucx \
              --mca coll_ucc_enable 1 \
              --mca coll_ucc_priority 100'

export LIBSROOT=/home/astro/musolino/libs/
export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib64


export P4EST_ROOT=/home/astro/musolino/libs/p4est-install-system-mpi
export CATCH2_ROOT=/home/astro/musolino/libs/catch2-install
export YAML_ROOT=/home/astro/musolino/libs/yaml-cpp-install
export KOKKOS_ROOT=/home/astro/musolino/libs/kokkos4-install
export VTK_ROOT=/home/astro/musolino/libs/VTK-install
export SPDLOG_ROOT=/home/astro/musolino/libs/spdlog-install
export HDF5_ROOT=/home/astro/musolino/libs/hdf5-install-system-mpi
export KOKKOS_TOOLS_ROOT=/home/astro/musolino/libs/kokkos-tools-install



export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin
