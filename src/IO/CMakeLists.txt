add_library(thunder_vtk_output OBJECT vtk_output.cpp vtk_setup_grid.cpp vtk_volume_output.cpp vtk_surface_output.cpp scalar_output.cpp)
target_include_directories( thunder_vtk_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_vtk_output PRIVATE
                        $<TARGET_OBJECTS:thunder_parallel>  
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        $<TARGET_OBJECTS:thunder_coordinates>
                        $<TARGET_OBJECTS:thunder_variable_list>
                        $<TARGET_OBJECTS:thunder_amr_functions>
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        VTK::VTK  
                        spdlog::spdlog )

vtk_module_autoinit(
    TARGETS thunder_vtk_output
    MODULES ${VTK_LIBRARIES}
    )

add_library(thunder_hdf5_output OBJECT hdf5_output.cpp xmf_utilities.cpp )
target_include_directories( thunder_hdf5_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_hdf5_output PRIVATE
                        $<TARGET_OBJECTS:thunder_parallel>  
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        $<TARGET_OBJECTS:thunder_coordinates>
                        $<TARGET_OBJECTS:thunder_variable_list>
                        $<TARGET_OBJECTS:thunder_amr_functions>
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        VTK::VTK  
                        HDF5::HDF5 
                        spdlog::spdlog )
vtk_module_autoinit(
    TARGETS thunder_hdf5_output
    MODULES ${VTK_LIBRARIES}
    )

add_library(thunder_cell_output OBJECT cell_output.cpp)
target_include_directories( thunder_cell_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_cell_output PRIVATE
                        $<TARGET_OBJECTS:thunder_parallel>  
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        $<TARGET_OBJECTS:thunder_coordinates>
                        $<TARGET_OBJECTS:thunder_variable_list>
                        $<TARGET_OBJECTS:thunder_amr_functions>
                        $<TARGET_OBJECTS:thunder_hdf5_output>
                        $<TARGET_OBJECTS:thunder_vtk_output>
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        VTK::VTK  
                        HDF5::HDF5 
                        OpenMP::OpenMP_CXX
                        spdlog::spdlog )
vtk_module_autoinit(
    TARGETS thunder_cell_output
    MODULES ${VTK_LIBRARIES}
    )


add_library(thunder_output_aux OBJECT vtk_output_auxiliaries.cpp )
target_include_directories( thunder_output_aux PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_output_aux PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc
                        Kokkos::kokkos
                        spdlog::spdlog )