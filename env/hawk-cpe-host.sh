#!/bin/bash

export LIB_BASE=/zhome/academic/HLRS/xfp/xfpmusol/HunterWS/GRACE/libs
export P4EST_ROOT=${LIB_BASE}/p4est-install
export CATCH2_ROOT=${LIB_BASE}/catch2-install
export YAML_ROOT=${LIB_BASE}/yaml-cpp-install
export KOKKOS_ROOT=${LIB_BASE}/kokkos-omp-install
export VTK_ROOT=${LIB_BASE}/vtk-install
export SPDLOG_ROOT=${LIB_BASE}/spdlog-install
export KOKKOS_TOOLS_ROOT=${LIB_BASE}/kokkos-tools-omp-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}

export PATH=/mnt/rafast/musolino/libs/valgrind-install/bin:${KOKKOS_TOOLS_ROOT}/bin:${PATH}

