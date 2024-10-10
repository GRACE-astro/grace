#!/bin/bash

module load mpi/openmpi/5.0.5-rocm ucx/1.17.0 rocm
export PATH=${ROCM_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
#export OMPI_ROOT=/home/astro/musolino/libs/ompi-install
#export PATH=${OMPI_ROOT}/bin:${PATH}
#export LD_LIBRARY_PATH=${OMPI_ROOT}/lib:${LD_LIBRARY_PATH}

export OMPI_CXX=hipcc 
#export MPI_CXX=${OMPI_ROOT}/bin/mpicxx

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

#export FFTW_ROOT=/home/astro/topolski/spack/share/spack/modules/linux-almalinux9-zen2/fftw/3.3.10-gcc-11.4.1-v7ksedl
export FFTW_ROOT=/scratch/astro/topolski/spack/linux-almalinux9-zen2/gcc-11.4.1/fftw-3.3.10-v7ksedlil647yr3a2oqp6sac4vzup2g3
export GSL_ROOT_DIR=/scratch/astro/topolski/spack/linux-almalinux9-zen2/gcc-11.4.1/gsl-2.7.1-ezhua3c3r7qqiw6rjimrhaek66nrtac7/
export SCALAPACK_DIR=/scratch/astro/topolski/spack/linux-almalinux9-zen2/gcc-11.4.1/netlib-scalapack-2.2.0-l7jrva7huea5eqywa3upnskjbnraxgm3/
export SCALAPACKDIR=/scratch/astro/topolski/spack/linux-almalinux9-zen2/gcc-11.4.1/netlib-scalapack-2.2.0-l7jrva7huea5eqywa3upnskjbnraxgm3/
export OPENBLAS_DIR=/scratch/astro/topolski/spack/linux-almalinux9-zen2/gcc-11.4.1/openblas-0.3.27-u2a64nqatx5hpmbsz2ifnj6jixjsefak/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OPENBLAS_DIR}lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SCALAPACK_DIR}lib
#export NETLIB_SCALAPACK_ROOT=/mnt/rafast/relastro-shared/new-spack/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-12.2.0/netlib-scalapack-2.2.0-dbrdsn63x4nd7zvasmw4a27iip7kktrg
#export INTEL_MKL_ROOT=/cluster/intel/oneapi/2023.2.0/mkl/2023.2.0
#export MKL_ROOT=$INTEL_MKL_ROOT
#export MKLROOT=$MKL_ROOT

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIBDIR}


export ZLIB_ROOT=/home/astro/musolino/libs/zlib/build
export LD_LIBRARY_PATH=${GSL_ROOT}/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=${KOKKOS_TOOLS_LIB}:${LD_LIBRARY_PATH}
export PATH=${PATH}:${KOKKOS_TOOLS_LIB}/../bin
