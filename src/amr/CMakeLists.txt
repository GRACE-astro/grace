add_library( thunder_connectivity OBJECT connectivity.cpp )
target_include_directories( thunder_connectivity PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_connectivity PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est )

add_library( thunder_forest OBJECT forest.cpp )
target_include_directories( thunder_forest PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_forest PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       $<TARGET_OBJECTS:thunder_connectivity> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est )

add_library( thunder_amr_functions OBJECT amr_functions.cpp )
target_include_directories( thunder_amr_functions PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_amr_functions PRIVATE 
                        $<TARGET_OBJECTS:thunder_parallel> 
                        $<TARGET_OBJECTS:thunder_singleton> 
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est  )


add_library( thunder_coordinates OBJECT coordinates.cpp )
target_include_directories( thunder_coordinates PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_coordinates PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       $<TARGET_OBJECTS:thunder_connectivity> 
                       $<TARGET_OBJECTS:thunder_forest>
                       $<TARGET_OBJECTS:thunder_amr_functions>
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       Kokkos::kokkos )

#add_library( thunder_regrid OBJECT regrid.cpp )
#target_include_directories( thunder_regrid PRIVATE "${HEADER_DIR}" )
#target_link_libraries( thunder_regrid PRIVATE 
#                        $<TARGET_OBJECTS:thunder_parallel> 
#                        $<TARGET_OBJECTS:thunder_singleton> 
#                        $<TARGET_OBJECTS:thunder_connectivity> 
#                        $<TARGET_OBJECTS:thunder_forest>
 #                       $<TARGET_OBJECTS:thunder_amr_functions>
#                        yaml_cpp::yaml
#                        MPI::MPI_CXX
#                        p4est::sc
#                        p4est::p4est 
#ß                        Kokkos::kokkos )