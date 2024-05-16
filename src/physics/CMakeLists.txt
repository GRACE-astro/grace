add_library(thunder_scalar_adv OBJECT scalar_advection.cpp )
target_include_directories(  thunder_scalar_adv PRIVATE "${HEADER_DIR}")
target_link_libraries( thunder_scalar_adv PRIVATE
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
add_library(thunder_burgers_eq OBJECT burgers_equation.cpp )
target_include_directories(  thunder_burgers_eq PRIVATE "${HEADER_DIR}")
target_link_libraries( thunder_burgers_eq PRIVATE
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