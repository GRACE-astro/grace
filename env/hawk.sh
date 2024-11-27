#!/bin/bash
module load CPE PrgEnv-nvidia cray-hdf5-parallel


alias mpirun='mpirun --mca pml ucx --mca osc ucx \
              --mca coll_ucc_enable 1 \
              --mca coll_ucc_priority 100'

export LIBSROOT=/zhome/academic/HLRS/xfp/xfpmusol/grace-libs
export P4EST_ROOT=${LIBSROOT}/p4est-install
export CATCH2_ROOT=${LIBSROOT}/catch2-install
export YAML_ROOT=${LIBSROOT}/yaml-cpp-install
export KOKKOS_ROOT=${LIBSROOT}/kokkos-install
export VTK_ROOT=${LIBSROOT}/VTK-install
export SPDLOG_ROOT=${LIBSROOT}/spdlog-install
export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib64

export PATH=${KOKKOS_ROOT}/bin:${PATH}

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin
