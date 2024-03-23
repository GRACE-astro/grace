add_library(thunder_variable_list OBJECT variables.cpp)

target_include_directories( thunder_variable_list PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_variable_list PRIVATE
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        $<TARGET_OBJECTS:thunder_coordinates>
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos )


add_library(thunder_variable_registration OBJECT variable_indices.cpp)

target_include_directories( thunder_variable_registration PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_variable_registration PRIVATE
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        $<TARGET_OBJECTS:thunder_variable_list>
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos )


add_library(thunder_variable_utils OBJECT variable_utils.cpp)

target_include_directories( thunder_variable_utils PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_variable_utils PRIVATE
                        $<TARGET_OBJECTS:thunder_error_handlers>
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        $<TARGET_OBJECTS:thunder_variable_list>
                        $<TARGET_OBJECTS:thunder_variable_registration>
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos )

