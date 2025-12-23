#!/bin/bash


export KAD_CC=hipcc
export KAD_CXX=hipcc
export KAD_NUMC=

export CC=hipcc
export CXX=hipcc
export MPICH_CC=hipcc
export MPICH_CXX=hipcc
export MPI_CC=hipcc
export MPI_CXX=hipcc

# Kadath support:
module load cray-fftw/3.3.10.10
module load gsl

export CPATH="${FFTW_INC}:${GSL_ROOT_DIR}/include:${CPATH}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${FFTW_ROOT}lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GSL_ROOT_LIB}lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CRAY_LIBSCI_PREFIX_DIR}/lib

# Add Cray LibSci include and lib paths
#export CRAY_LIBSCI_PREFIX_DIR=/opt/cray/pe/libsci/24.07.0/CRAYCLANG/18.0/x86_64
export CBLAS_INCLUDE_DIR=${CRAY_LIBSCI_PREFIX_DIR}/include
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CRAY_LIBSCI_PREFIX_DIR}/lib

# One needs to change the cmakelocal in KADATH
#set (GSL_LIBRARIES "$ENV{GSL_ROOT_DIR}/lib/libgsl.so")
#set (FFTW_LIBRARIES "$ENV{FFTW_ROOT}/lib/libfftw3.so")
#set (SCALAPACK_LIBRARIES "$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_cray_mpi.so")


