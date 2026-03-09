#!/bin/bash
#export OMPI_CXX=icpx       # DPC++ compiler

# MPI wrapper around the DPC++ (GPU-aware) compiler
export OMPI_CXX=mpiicpx

#export LIBSROOT=/home/astro/musolino/libs

export LIBSROOT=/dss/dsshome1/0B/di75rur/libs

#export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-install
export KOKKOS_TOOLS_ROOT=${LIBSROOT}/kokkos-tools-4.6.01-dpcpp-install
export KOKKOS_TOOLS_LIB=${KOKKOS_TOOLS_ROOT}/lib64

export P4EST_ROOT=${LIBSROOT}/p4est-install-system-mpi
export CATCH2_ROOT=${LIBSROOT}/catch2-install
export YAML_ROOT=${LIBSROOT}/yaml-cpp-install
#export KOKKOS_ROOT=${LIBSROOT}/kokkos-4.4.0-install
#export KOKKOS_ROOT=${LIBSROOT}/kokkos-4.5.0-install
export KOKKOS_ROOT=${LIBSROOT}/kokkos-4.6.01-install

#export VTK_ROOT=${LIBSROOT}/VTK-install
export VTK_ROOT=/dss/dsshome1/0B/di75rur/user_spack/24.1.0/opt/linux-sles15-sapphirerapids/vtk/9.0.0-gcc-13.2.0-6n7oyu5

export SPDLOG_ROOT=${LIBSROOT}/spdlog-install
export HDF5_ROOT=/dss/lrzsys/sys/spack/release/24.1.0/opt/sapphirerapids/hdf5/1.14.3-oneapi-z5yvd3l

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin

# GPU offload tuning
export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_RDMA=1
export I_MPI_OFFLOAD_IPC=1
export I_MPI_OFFLOAD_L0_D2D_ENGINE_TYPE=1
export I_MPI_OFFLOAD_CELL=device
export I_MPI_DEBUG=3


# extra setup to be removed later 
module load user_spack
spack load libxml2@2.10.3%oneapi@2024.1.0 arch=linux-sles15-sapphirerapids


export TwoPunctures_ROOT=/hppfs/work/pn67xi/di75rur/Standalone-TwoPunctures-C-Cpp/
export FFTW_BASE="${LIBSROOT}/fftw-3.3.10-install"
export GSL_BASE="${LIBSROOT}/gsl-2.8-install"
export FFTW_ROOT="${FFTW_BASE}"
export FFTW_DIR="${FFTW_BASE}"
#export FFTW_LIBRARIES="fftw3"
export GSL_ROOT_DIR=${GSL_BASE}
export BOOST_ROOT_DIR=${BOOST_BASE}
export BOOST_ROOT=${BOOST_BASE}


