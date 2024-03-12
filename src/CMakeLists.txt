set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")


add_library( thunder_error_handlers OBJECT
             errors/abort.cc 
             errors/error.cc )

target_include_directories( thunder_error_handlers PRIVATE "${HEADER_DIR}")

add_library( thunder_parallel OBJECT 
             parallel/mpi_wrappers.cc )

target_include_directories(thunder_parallel PRIVATE "${HEADER_DIR}")

target_link_libraries(thunder_parallel PRIVATE 
                      MPI::MPI_CXX 
                      p4est::sc )

add_library(thunder_singleton OBJECT utils/lifetime_tracker.cc) 

target_include_directories(thunder_singleton PRIVATE  "${HEADER_DIR}")



