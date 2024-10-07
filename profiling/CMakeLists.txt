add_executable(bm_IO bm_IO.cpp)
target_include_directories(bm_IO PRIVATE "${HEADER_DIR}")
target_link_libraries(bm_IO PRIVATE 
        grace_objects
        $<TARGET_OBJECTS:grace_vtk_output>
        $<TARGET_OBJECTS:grace_hdf5_output>
        $<TARGET_OBJECTS:grace_cell_output>
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        VTK::VTK
        spdlog::spdlog
        HDF5::HDF5 
        $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
        )
vtk_module_autoinit(
    TARGETS bm_IO
    MODULES ${VTK_LIBRARIES}
)

add_executable(bm_fluxes bm_flux_computation.cpp)
target_include_directories(bm_fluxes PRIVATE "${HEADER_DIR}")
target_link_libraries(bm_fluxes PRIVATE 
        yaml_cpp::yaml
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        spdlog::spdlog
        $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
        )