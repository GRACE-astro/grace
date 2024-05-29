add_library( grace_evolution OBJECT auxiliaries.cpp evolve.cpp find_stable_timestep.cpp )
target_include_directories( grace_evolution PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_evolution PRIVATE 
                       $<TARGET_OBJECTS:grace_error_handlers>
                       $<TARGET_OBJECTS:grace_parallel>
                       $<TARGET_OBJECTS:grace_singleton>
                       $<TARGET_OBJECTS:grace_connectivity> 
                       $<TARGET_OBJECTS:grace_forest> 
                       $<TARGET_OBJECTS:grace_amr_functions>
                       $<TARGET_OBJECTS:grace_BC_helpers>
                       $<TARGET_OBJECTS:grace_BC>
                       $<TARGET_OBJECTS:grace_variable_list>
                       $<TARGET_OBJECTS:grace_variable_registration>
                       $<TARGET_OBJECTS:grace_variable_utils>
                       yaml_cpp::yaml
                       MPI::MPI_CXX 
                       p4est::sc 
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )
add_library( grace_initial_data OBJECT initial_data.cpp )
target_include_directories( grace_initial_data PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_initial_data PRIVATE 
                        $<TARGET_OBJECTS:grace_error_handlers>
                        $<TARGET_OBJECTS:grace_parallel>
                        $<TARGET_OBJECTS:grace_singleton>
                        $<TARGET_OBJECTS:grace_connectivity> 
                        $<TARGET_OBJECTS:grace_forest> 
                        $<TARGET_OBJECTS:grace_amr_functions>
                        $<TARGET_OBJECTS:grace_BC_helpers>
                        $<TARGET_OBJECTS:grace_BC>
                        $<TARGET_OBJECTS:grace_variable_list>
                        $<TARGET_OBJECTS:grace_variable_registration>
                        $<TARGET_OBJECTS:grace_variable_utils>
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )