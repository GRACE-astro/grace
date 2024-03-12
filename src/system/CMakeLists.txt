add_library( thunder_runtime_utilities OBJECT print.cpp )
target_include_directories( thunder_runtime_utilities PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_runtime_utilities PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc)