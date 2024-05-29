add_library( grace_parallel OBJECT mpi_wrappers.cc )
target_include_directories(grace_parallel PRIVATE "${HEADER_DIR}")
target_link_libraries(grace_parallel PRIVATE 
                      MPI::MPI_CXX 
                      p4est::sc )