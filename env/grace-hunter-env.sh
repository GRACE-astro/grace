#!/bin/bash

# Set linking to dynamic libraries
export CRAYPE_LINK_TYPE=dynamic

# Set compilers to use
export CMAKE_C_COMPILER=hipcc            # Use Cray CPU C compiler for host CPU code
export CMAKE_CXX_COMPILER=hipcc       # Use HIP compiler for GPU-enabled C++ code

# Set root directories for various third-party libraries built with Hunter package manager
export LIBSROOT=/zhome/academic/HLRS/xfp/xfpolski/libs/
export P4EST_ROOT=${LIBSROOT}/p4est-hunter-install
export CATCH2_ROOT=${LIBSROOT}/Catch2-hunter-install
export YAML_ROOT=${LIBSROOT}/yaml-cpp-hunter-install
export YAML_CPP_LIBRARIES="${YAML_ROOT}/lib64/libyaml-cpp.a"
export YAML_CPP_INCLUDE_DIRS="${YAML_ROOT}/include"
export KOKKOS_ROOT=${LIBSROOT}/kokkos-hunter-install
export VTK_ROOT=${LIBSROOT}/VTK-hunter-install
export SPDLOG_ROOT=${LIBSROOT}/spdlog-hunter-install
export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-hunter-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib64
export HDF5_ROOT=${LIBSROOT}/hdf5-hunter-install

# Add Kokkos binaries and tools to PATH and LD_LIBRARY_PATH
export PATH=${KOKKOS_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin

# Enable GPU support in MPICH for GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

export PATH="${LIBSROOT}/hdf5-hunter-install/bin:$PATH"



# Kadath support: 
module load cray-fftw/3.3.10.10
module load gsl 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${FFTW_ROOT}lib 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GSL_ROOT_LIB}lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CRAY_LIBSCI_PREFIX_DIR}/lib

# Add Cray LibSci include and lib paths
#export CRAY_LIBSCI_PREFIX_DIR=/opt/cray/pe/libsci/24.07.0/CRAYCLANG/18.0/x86_64
export CBLAS_INCLUDE_DIR=${CRAY_LIBSCI_PREFIX_DIR}/include
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CRAY_LIBSCI_PREFIX_DIR}/lib

