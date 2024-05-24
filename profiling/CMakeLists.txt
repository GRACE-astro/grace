add_executable(bm_IO bm_IO.cpp)
target_include_directories(bm_IO PRIVATE "${HEADER_DIR}")
target_link_libraries(bm_IO PRIVATE 
        thunder_objects
        $<TARGET_OBJECTS:thunder_vtk_output>
        $<TARGET_OBJECTS:thunder_hdf5_output>
        $<TARGET_OBJECTS:thunder_cell_output>
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        VTK::VTK
        spdlog::spdlog
        HDF5::HDF5 
        )
vtk_module_autoinit(
    TARGETS bm_IO
    MODULES ${VTK_LIBRARIES}
)