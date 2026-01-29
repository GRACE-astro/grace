#!/bin/bash

export PATH=${EBROOTCUDA}/bin:${PATH}
export LD_LIBRARY_PATH=${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH}

export OMPI_CXX=nvcc

alias mpirun='mpirun --mca pml ucx --mca osc ucx \
              --mca coll_ucc_enable 1 \
              --mca coll_ucc_priority 100'

export LIBSROOT=/home/it4i-kpierre/codes/libs
export P4EST_ROOT=${LIBSROOT}/p4est-install
export CATCH2_ROOT=${LIBSROOT}/Catch2-install
export YAML_ROOT=${LIBSROOT}/yaml-cpp-install
export KOKKOS_ROOT=${LIBSROOT}/kokkos-install
export VTK_ROOT=${LIBSROOT}/VTK-install
export SPDLOG_ROOT=${LIBSROOT}/spdlog-install
export HDF5_ROOT=${LIBSROOT}/hdf5-install-new
export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib64


export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin

while read -r e; do
        NAME=$(cut -d= -f1 <<< "$e")
        VALUE=$(printenv $NAME)
        export $NAME="${VALUE/tp=zen2/march=znver2}"
done< <(env | grep 'tp=zen2')

                                               
