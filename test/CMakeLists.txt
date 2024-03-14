set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/")

add_executable( errors_test test_errors.cc ) 

target_include_directories(errors_test PRIVATE "${HEADER_LIST}")
target_compile_options(errors_test PRIVATE -g)
target_link_libraries( errors_test PRIVATE 
                        $<TARGET_OBJECTS:thunder_error_handlers> )

add_library(mpi_tests_main mains/mpi_tests_main.cc)
target_include_directories(mpi_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  mpi_tests_main PRIVATE 
Catch2::Catch2
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_error_handlers>
MPI::MPI_CXX
p4est::sc )

add_executable( parallel_test test_parallel.cc )
target_include_directories(parallel_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  parallel_test PRIVATE 
mpi_tests_main
Catch2::Catch2
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_error_handlers>
MPI::MPI_CXX
p4est::sc )




add_library(parser_tests_main mains/parser_tests_main.cc)
target_include_directories(parser_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  parser_tests_main PRIVATE 
Catch2::Catch2
yaml_cpp::yaml
$<TARGET_OBJECTS:thunder_error_handlers>
$<TARGET_OBJECTS:thunder_singleton> )

add_executable( parser_test test_parser.cc )
target_include_directories(parser_test PRIVATE "${HEADER_LIST}")

target_link_libraries(  parser_test PRIVATE 
parser_tests_main
yaml_cpp::yaml
Catch2::Catch2
$<TARGET_OBJECTS:thunder_error_handlers>
$<TARGET_OBJECTS:thunder_singleton> )

add_library(p4est_tests_main mains/p4est_tests_main.cc)
target_include_directories(p4est_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  p4est_tests_main PRIVATE 
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
p4est::sc
p4est::p4est 
Kokkos::kokkos )

add_executable( connectivity_test test_connectivities.cc )
target_include_directories(connectivity_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  connectivity_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
p4est::sc
p4est::p4est )


add_library(kokkos_tests_main mains/kokkos_tests_main.cc)
target_include_directories(kokkos_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  kokkos_tests_main PRIVATE 
Catch2::Catch2
$<TARGET_OBJECTS:thunder_error_handlers>
$<TARGET_OBJECTS:thunder_singleton>
yaml_cpp::yaml
Kokkos::kokkos )


add_executable( sc_wrappers_test test_sc_wrappers.cpp )
target_include_directories(sc_wrappers_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  sc_wrappers_test PRIVATE 
p4est_tests_main
Catch2::Catch2
MPI::MPI_CXX
p4est::sc
p4est::p4est )

add_executable( forest_test test_forest.cpp )
target_include_directories(forest_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  forest_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
p4est::sc
p4est::p4est )

add_executable( varlist_test test_variable_list.cpp )
target_include_directories(varlist_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  varlist_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
p4est::sc
p4est::p4est 
Kokkos::kokkos)

add_executable( coordinate_test test_coordinates.cpp )
target_include_directories(coordinate_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  coordinate_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
p4est::sc
p4est::p4est )



add_test(NAME parallel_test COMMAND mpirun -n 2 ./parallel_test)
add_test(NAME parser_test COMMAND parser_test)
add_test(NAME sc_wrappers_test COMMAND 
        sc_wrappers_test --thunder-parfile ./configs/basic_config.yaml)
add_test(NAME connectivity_test COMMAND 
        connectivity_test --thunder-parfile ./configs/basic_config.yaml)
add_test(NAME forest_test COMMAND 
        forest_test --thunder-parfile ./configs/basic_config.yaml)
add_test(NAME varlist_test COMMAND 
        varlist_test --thunder-parfile ./configs/basic_config.yaml )
        add_test(NAME coordinate_test COMMAND 
        coordinate_test --thunder-parfile ./configs/basic_config.yaml )


