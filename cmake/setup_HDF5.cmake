# Prefer the MPI-parallel HDF5 build when both serial and parallel installs
# are present on the system (e.g. Ubuntu's `libhdf5-dev` + `libhdf5-openmpi-dev`
# co-install, or HPC modules that ship both side-by-side).  Without this hint
# FindHDF5 may pick serial HDF5 first and the GRACE configure succeeds but the
# parallel I/O paths fail to link.  HDF5_PREFER_PARALLEL is a documented
# FindHDF5 cache variable (CMake >= 3.6).
set(HDF5_PREFER_PARALLEL ON)

find_package(HDF5 REQUIRED COMPONENTS C)

if ( NOT HDF5_FOUND )
  if( NOT HDF5_ROOT )
    set(HDF5_ROOT "$ENV{HDF5_ROOT}")
  endif()

  message(STATUS "Looking for HDF5 in ${HDF5_ROOT}")

  find_package(HDF5 PATHS "${HDF5_ROOT}")
endif()
