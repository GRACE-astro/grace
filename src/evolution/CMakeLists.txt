add_library( thunder_evolution OBJECT auxiliaries.cpp evolve.cpp find_stable_timestep.cpp )
target_include_directories( thunder_evolution PRIVATE "${HEADER_DIR}")
target_link_libraries( thunder_evolution PRIVATE 
                       $<TARGET_OBJECTS:thunder_error_handlers>
                       $<TARGET_OBJECTS:thunder_parallel>
                       $<TARGET_OBJECTS:thunder_singleton>
                       $<TARGET_OBJECTS:thunder_connectivity> 
                       $<TARGET_OBJECTS:thunder_forest> 
                       $<TARGET_OBJECTS:thunder_amr_functions>
                       $<TARGET_OBJECTS:thunder_BC_helpers>
                       $<TARGET_OBJECTS:thunder_BC>
                       $<TARGET_OBJECTS:thunder_variable_list>
                       $<TARGET_OBJECTS:thunder_variable_registration>
                       $<TARGET_OBJECTS:thunder_variable_utils>
                       yaml_cpp::yaml
                       MPI::MPI_CXX 
                       p4est::sc 
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )
add_library( thunder_initial_data OBJECT initial_data.cpp )
target_include_directories( thunder_initial_data PRIVATE "${HEADER_DIR}")
target_link_libraries( thunder_initial_data PRIVATE 
                        $<TARGET_OBJECTS:thunder_error_handlers>
                        $<TARGET_OBJECTS:thunder_parallel>
                        $<TARGET_OBJECTS:thunder_singleton>
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest> 
                        $<TARGET_OBJECTS:thunder_amr_functions>
                        $<TARGET_OBJECTS:thunder_BC_helpers>
                        $<TARGET_OBJECTS:thunder_BC>
                        $<TARGET_OBJECTS:thunder_variable_list>
                        $<TARGET_OBJECTS:thunder_variable_registration>
                        $<TARGET_OBJECTS:thunder_variable_utils>
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )