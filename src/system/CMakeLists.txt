add_library( thunder_runtime_utilities OBJECT print.cpp )
target_include_directories( thunder_runtime_utilities PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_runtime_utilities PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc)

add_library( thunder_init OBJECT thunder_initialize.cpp )
target_include_directories( thunder_init PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_init PRIVATE 
                        $<TARGET_OBJECTS:thunder_parallel> 
                        $<TARGET_OBJECTS:thunder_singleton>
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest> 
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos )