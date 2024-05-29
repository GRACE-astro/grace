add_library( grace_runtime_utilities OBJECT runtime_functions.cpp )
target_include_directories( grace_runtime_utilities PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_runtime_utilities PRIVATE 
                       $<TARGET_OBJECTS:grace_parallel> 
                       $<TARGET_OBJECTS:grace_singleton> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       Kokkos::kokkos
                       spdlog::spdlog )

add_library( grace_init OBJECT grace_initialize.cpp )
target_include_directories( grace_init PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_init PRIVATE 
                        $<TARGET_OBJECTS:grace_parallel> 
                        $<TARGET_OBJECTS:grace_singleton>
                        $<TARGET_OBJECTS:grace_connectivity> 
                        $<TARGET_OBJECTS:grace_forest> 
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos
                        spdlog::spdlog )

add_library( grace_finalize OBJECT grace_finalize.cpp )
target_include_directories( grace_finalize PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_finalize PRIVATE 
                    $<TARGET_OBJECTS:grace_parallel> 
                    $<TARGET_OBJECTS:grace_singleton>
                    $<TARGET_OBJECTS:grace_connectivity> 
                    $<TARGET_OBJECTS:grace_forest> 
                    yaml_cpp::yaml
                    MPI::MPI_CXX
                    p4est::sc
                    p4est::p4est 
                    Kokkos::kokkos
                    spdlog::spdlog )