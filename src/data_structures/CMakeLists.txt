add_library(thunder_variable_registration OBJECT variable_indices.cpp)

target_include_directories( thunder_variable_registration PRIVATE "${HEADER_DIR}" )
target_link_libraries(  thunder_variable_registration PRIVATE
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos )