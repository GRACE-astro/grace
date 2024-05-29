add_library(grace_eos OBJECT eos_storage.cpp )
target_include_directories(  grace_eos PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_eos PRIVATE
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