export MODULEPATH="/mnt/proj3/open-30-28/gkatev/easybuild/modules/all:$MODULEPATH"

ml CUDA/12.8.0 NVHPC/25.3-CUDA-12.8.0 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0 UCX UCX-CUDA

export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export GRACE_LIBS="/mnt/proj3/open-30-28/gkatev/grace-libs"

export KOKKOS_ROOT="$GRACE_LIBS/kokkos-4.6.00"
export YAML_ROOT="$GRACE_LIBS/yaml-cpp-2f86d13"
export SPDLOG_ROOT="$GRACE_LIBS/spdlog-1.15.2"
export P4EST_ROOT="$GRACE_LIBS/p4est-2.8.7"
export CATCH2_ROOT="$GRACE_LIBS/Catch2-3.8.1"
export HDF5_ROOT="$GRACE_LIBS/hdf5-1.14.6"
export VTK_ROOT="$GRACE_LIBS/vtk-9.4.2"

while read -r e; do
	NAME=$(cut -d= -f1 <<< "$e")
	VALUE=$(printenv $NAME)
	export $NAME="${VALUE/tp=zen2/march=znver2}"
done< <(env | grep 'tp=zen2')

# cmake -Bbuild -S./ -DGRACE_NSPACEDIM=3 -DGRACE_ENABLE_CUDA=ON -Wno-dev \
#   -DCMAKE_CXX_COMPILER="$KOKKOS_ROOT/bin/nvcc_wrapper" -DCMAKE_C_COMPILER=nvc
