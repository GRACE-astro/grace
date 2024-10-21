
#!/bin/bash
spack load gcc@11.4.0 cmake openmpi%gcc@11.4.0

export KOKKOS_ROOT=${HOME}/libs/install/kokkos
export P4EST_ROOT=${HOME}/libs/install/p4est-dbg
export YAML_ROOT=${HOME}/libs/install/yaml-cpp
export CATCH2_ROOT=${HOME}/libs/install/catch2
export MPI_ROOT=$(spack location -i openmpi%gcc@11.4.0)