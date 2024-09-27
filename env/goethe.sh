#!/bin/bash
module load mpi/openmpi/5.0.5-rocm ucx/1.17.0 rocm
export PATH=${ROCM_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

export OMPI_CXX=hipcc 

alias mpirun='mpirun --mca pml ucx --mca osc ucx \
              --mca coll_ucc_enable 1 \
              --mca coll_ucc_priority 100'

export LIBSROOT=/home/astro/musolino/libs/
export P4EST_ROOT=${LIBSROOT}/p4est-install-system-mpi
export CATCH2_ROOT=${LIBSROOT}/catch2-install
export YAML_ROOT=${LIBSROOT}/yaml-cpp-install
export KOKKOS_ROOT=${LIBSROOT}/kokkos-install-system-mpi
export VTK_ROOT=${LIBSROOT}/VTK-install
export SPDLOG_ROOT=${LIBSROOT}/spdlog-install
export HDF5_ROOT=${LIBSROOT}/hdf5-install-system-mpi
export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib64


export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin
