add_library(thunder_volume_output OBJECT vtk_volume_output.cpp)
target_include_directories( thunder_volume_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_volume_output PRIVATE
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
                        VTK::VTK  )

vtk_module_autoinit(
    TARGETS thunder_volume_output
    MODULES ${VTK_LIBRARIES}
    )