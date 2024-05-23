if( NOT HDF5_ROOT )
    set(HDF5_ROOT "$ENV{HDF5_ROOT}")
endif()
message(STATUS "Looking for HDF5 in ${HDF5_ROOT}")

find_package(HDF5 PATHS "${HDF5_ROOT}")
find_package(HDF5 REQUIRED COMPONENTS C)