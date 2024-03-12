add_library( thunder_parallel OBJECT mpi_wrappers.cc )
target_include_directories(thunder_parallel PRIVATE "${HEADER_DIR}")
target_link_libraries(thunder_parallel PRIVATE 
                      MPI::MPI_CXX 
                      p4est::sc )