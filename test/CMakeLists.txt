set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/")

add_executable( errors_test test_errors.cc ) 

target_include_directories(errors_test PRIVATE "${HEADER_LIST}")
target_compile_options(errors_test PRIVATE -g)
target_link_libraries( errors_test PRIVATE 
                        $<TARGET_OBJECTS:thunder_error_handlers>
                        $<TARGET_OBJECTS:thunder_parallel>
                        MPI::MPI_CXX 
                        p4est::sc 
                        spdlog::spdlog )

add_library(mpi_tests_main mains/mpi_tests_main.cc)
target_include_directories(mpi_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  mpi_tests_main PRIVATE 
Catch2::Catch2
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_error_handlers>
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog )

add_executable( parallel_test test_parallel.cc )
target_include_directories(parallel_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  parallel_test PRIVATE 
mpi_tests_main
Catch2::Catch2
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_error_handlers>
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog )




add_library(parser_tests_main mains/parser_tests_main.cc)
target_include_directories(parser_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  parser_tests_main PRIVATE 
Catch2::Catch2
yaml_cpp::yaml
$<TARGET_OBJECTS:thunder_error_handlers>
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_singleton> 
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog  )

add_executable( parser_test test_parser.cc )
target_include_directories(parser_test PRIVATE "${HEADER_LIST}")

target_link_libraries(  parser_test PRIVATE 
parser_tests_main
yaml_cpp::yaml
Catch2::Catch2
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_error_handlers>
$<TARGET_OBJECTS:thunder_singleton> 
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog )

add_library(p4est_tests_main mains/p4est_tests_main.cc)
target_include_directories(p4est_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  p4est_tests_main PRIVATE 
Catch2::Catch2
yaml_cpp::yaml
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est 
Kokkos::kokkos
spdlog::spdlog )

add_executable( connectivity_test test_connectivities.cc )
target_include_directories(connectivity_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  connectivity_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est
spdlog::spdlog )


add_library(kokkos_tests_main mains/kokkos_tests_main.cc)
target_include_directories(kokkos_tests_main PRIVATE "${HEADER_LIST}")
target_link_libraries(  kokkos_tests_main PRIVATE 
Catch2::Catch2
$<TARGET_OBJECTS:thunder_error_handlers>
$<TARGET_OBJECTS:thunder_parallel>
$<TARGET_OBJECTS:thunder_singleton>
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog )


add_executable( sc_wrappers_test test_sc_wrappers.cpp )
target_include_directories(sc_wrappers_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  sc_wrappers_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est
spdlog::spdlog )

add_executable( forest_test test_forest.cpp )
target_include_directories(forest_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  forest_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est
spdlog::spdlog )

add_executable( varlist_test test_variable_list.cpp )
target_include_directories(varlist_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  varlist_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est 
Kokkos::kokkos
spdlog::spdlog)

add_executable( coordinate_test test_coordinates.cpp )
target_include_directories(coordinate_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  coordinate_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
yaml_cpp::yaml
Kokkos::kokkos
$<TARGET_OBJECTS:thunder_vtk_output>
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est 
VTK::VTK 
spdlog::spdlog)
vtk_module_autoinit(
    TARGETS coordinate_test
    MODULES ${VTK_LIBRARIES}
    )

add_executable( volume_output_test test_output.cpp )
target_include_directories(volume_output_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  volume_output_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
$<TARGET_OBJECTS:thunder_vtk_output>
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est 
VTK::VTK 
spdlog::spdlog)
vtk_module_autoinit(
    TARGETS volume_output_test
    MODULES ${VTK_LIBRARIES}
    )


add_executable( interpolator_test test_interpolators.cpp )
target_include_directories( interpolator_test PRIVATE "${HEADER_LIST}")
target_compile_options(interpolator_test PRIVATE -Wno-dangling-field)
target_link_libraries( interpolator_test PRIVATE
        kokkos_tests_main
        Catch2::Catch2 
        Kokkos::kokkos ) 

add_executable( integrator_test test_integration.cpp )
target_include_directories( integrator_test PRIVATE "${HEADER_LIST}")
target_compile_options(integrator_test PRIVATE -O3)
target_link_libraries( integrator_test PRIVATE
        Catch2::Catch2WithMain ) 


add_executable( regrid_test test_regridding.cpp )
target_include_directories(regrid_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  regrid_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
$<TARGET_OBJECTS:thunder_vtk_output>
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est 
VTK::VTK 
spdlog::spdlog)
vtk_module_autoinit(
    TARGETS regrid_test
    MODULES ${VTK_LIBRARIES}
    )

add_executable( bc_test test_bc.cpp )
target_include_directories(bc_test PRIVATE "${HEADER_LIST}")
target_link_libraries(  bc_test PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
$<TARGET_OBJECTS:thunder_vtk_output>
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
p4est::sc
p4est::p4est 
VTK::VTK 
spdlog::spdlog )
vtk_module_autoinit(
TARGETS bc_test
MODULES ${VTK_LIBRARIES}
)

add_executable( bc_test_unigrid test_bc_unigrid.cpp )
target_include_directories(bc_test_unigrid PRIVATE "${HEADER_LIST}")
target_link_libraries(  bc_test_unigrid PRIVATE 
p4est_tests_main
Catch2::Catch2
thunder_objects
$<TARGET_OBJECTS:thunder_vtk_output>
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
p4est::sc
p4est::p4est 
VTK::VTK 
spdlog::spdlog )
vtk_module_autoinit(
TARGETS bc_test
MODULES ${VTK_LIBRARIES}
)

add_test(NAME parallel_test COMMAND mpirun -n 2 ./parallel_test)
add_test(NAME parser_test COMMAND ./parser_test)
add_test(NAME sc_wrappers_test COMMAND 
        mpirun -n 1 ./sc_wrappers_test --thunder-parfile ./configs/basic_config.yaml)
add_test(NAME connectivity_test COMMAND 
        mpirun -n 1 ./connectivity_test --thunder-parfile ./configs/basic_config.yaml)
add_test(NAME forest_test COMMAND 
        mpirun -n 1 ./forest_test --thunder-parfile ./configs/basic_config.yaml)
add_test(NAME varlist_test COMMAND 
        mpirun -n 1 ./varlist_test --thunder-parfile ./configs/basic_config.yaml )
add_test(NAME coordinate_test_nolog COMMAND 
        mpirun -n 1 ./coordinate_test --thunder-parfile ./configs/basic_config.yaml )
add_test(NAME coordinate_test_log COMMAND 
        mpirun -n 1 ./coordinate_test --thunder-parfile ./configs/regrid_test_spherical_log.yaml )
add_test(NAME interpolator_test COMMAND 
        ./interpolator_test )
add_test(NAME regrid_test_cartesian COMMAND 
        mpirun -n 8 ./regrid_test --thunder-parfile ./configs/regrid_test_cartesian.yaml)
add_test(NAME regrid_test_nolog COMMAND 
        mpirun -n 8 ./regrid_test --thunder-parfile ./configs/regrid_test_spherical.yaml )
add_test(NAME regrid_test_log COMMAND 
        mpirun -n 8 ./regrid_test --thunder-parfile ./configs/regrid_test_spherical_log.yaml )
add_test(NAME bc_test_unigrid_cartesian COMMAND 
        mpirun -n 8 ./bc_test_unigrid --thunder-parfile ./configs/regrid_test_cartesian.yaml )
add_test(NAME bc_test_unigrid_spherical COMMAND 
        mpirun -n 2 ./bc_test_unigrid --thunder-parfile ./configs/regrid_test_spherical.yaml )
add_test(NAME bc_test_cartesian COMMAND 
        mpirun -n 8 ./bc_test --thunder-parfile ./configs/regrid_test_cartesian.yaml )
add_test(NAME bc_test_spherical COMMAND 
        mpirun -n 2 ./bc_test --thunder-parfile ./configs/regrid_test_spherical.yaml )
##add_custom_command(
#        TARGET
#        COMMENT "Run tests"
#        POST_BUILD 
#        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test" 
#        COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIGURATION> -R "Unit tests" --output-on-failure
#)