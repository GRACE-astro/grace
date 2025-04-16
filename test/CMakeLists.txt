# Utility functions to register tests 
# in GRACE's unit-test framework
include(register_test)


# Here we define some object files that act as 
# mains for the Catch2 tests in different setups

# Tests involving MPI
add_library(mpi_tests_main mains/mpi_tests_main.cc)
target_include_directories(mpi_tests_main PRIVATE "${HEADER_DIR}")
target_link_libraries(  mpi_tests_main PRIVATE 
Catch2::Catch2
$<TARGET_OBJECTS:grace_parallel>
$<TARGET_OBJECTS:grace_error_handlers>
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog 
ZLIB::ZLIB)

# Tests involving the yaml parser
add_library(parser_tests_main mains/parser_tests_main.cc)
target_include_directories(parser_tests_main PRIVATE "${HEADER_DIR}")
target_link_libraries(  parser_tests_main PRIVATE 
Catch2::Catch2
yaml_cpp::yaml
$<TARGET_OBJECTS:grace_error_handlers>
$<TARGET_OBJECTS:grace_parallel>
$<TARGET_OBJECTS:grace_singleton> 
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog  
ZLIB::ZLIB)

# Tests involving the whole grace infrastructure
add_library(p4est_tests_main mains/p4est_tests_main.cc)
target_include_directories(p4est_tests_main PRIVATE "${HEADER_DIR}")
target_link_libraries(  p4est_tests_main PRIVATE 
Catch2::Catch2
yaml_cpp::yaml
MPI::MPI_CXX
OpenMP::OpenMP_CXX
p4est::sc
p4est::p4est 
Kokkos::kokkos
spdlog::spdlog 
ZLIB::ZLIB
$<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling> )

# Tests involving Kokkos
add_library(kokkos_tests_main mains/kokkos_tests_main.cc)
target_include_directories(kokkos_tests_main PRIVATE "${HEADER_DIR}")
target_link_libraries(  kokkos_tests_main PRIVATE 
Catch2::Catch2
$<TARGET_OBJECTS:grace_error_handlers>
$<TARGET_OBJECTS:grace_parallel>
$<TARGET_OBJECTS:grace_singleton>
yaml_cpp::yaml
Kokkos::kokkos
MPI::MPI_CXX
p4est::sc 
spdlog::spdlog )

# Now add all the executables for the tests
# the functions ensure that the correct libraries
# are linked and keep this file relatively tidy 
add_mpi_test(parallel_test test_parallel.cc )
add_parser_test(parser_test test_parser.cc )
add_grace_test(connectivity_test test_connectivities.cc)
add_grace_test(sc_wrappers_test test_sc_wrappers.cpp)
add_grace_test(forest_test test_forest.cpp)
add_grace_test(varlist_test test_variable_list.cpp)
add_grace_test(coordinate_test test_coordinates.cpp)
add_grace_test(volume_output_test test_output.cpp)
add_grace_test(regrid_test test_regridding.cpp)
add_grace_test(bc_test test_bc.cpp)
add_grace_test(scalar_reductions_test test_scalar_reductions.cpp)
add_grace_test(hdf5_output_test test_hdf5_IO.cpp)
add_grace_test(hdf5_output_sliced_test test_hdf5_IO_sliced.cpp)
add_grace_test(eos_test test_eos.cpp)
add_grace_test(metric_test test_metric.cpp)
add_grace_test(c2p_test test_c2p.cpp)
add_grace_test(gridloop_test test_grid_loops.cpp)
add_grace_test(cart_coords_test test_cartesian_coordinates.cpp)
add_grace_test(bssn_test test_bssn.cpp)
add_grace_test(TOV_test test_TOV.cpp)
add_grace_test(test_p4est_search test_p4est_search.cpp)

add_kokkos_test(interpolator_test test_interpolators.cpp)
add_kokkos_test(device_vector_test test_device_vector.cc)
add_kokkos_test(lagrange_prolongation_test test_lagrange_interpolators.cpp)

add_standalone_test(integrator_test test_integration.cpp)

add_executable( errors_test test_errors.cc ) 
target_include_directories(errors_test PRIVATE "${HEADER_DIR}")
target_compile_options(errors_test PRIVATE -g)
target_link_libraries( errors_test PRIVATE 
                        $<TARGET_OBJECTS:grace_error_handlers>
                        $<TARGET_OBJECTS:grace_parallel>
                        MPI::MPI_CXX 
                        p4est::sc 
                        spdlog::spdlog 
                        ZLIB::ZLIB)

# Register tests with ctest 
add_test(NAME parallel_test COMMAND mpirun -n 2 ./parallel_test)
add_test(NAME parser_test COMMAND ./parser_test)
add_test(NAME sc_wrappers_test COMMAND 
        mpirun -n 1 ./sc_wrappers_test --grace-parfile ./configs/basic_config.yaml)
add_test(NAME connectivity_test COMMAND 
        mpirun -n 1 ./connectivity_test --grace-parfile ./configs/basic_config.yaml)
add_test(NAME forest_test COMMAND 
        mpirun -n 1 ./forest_test --grace-parfile ./configs/basic_config.yaml)
add_test(NAME varlist_test COMMAND 
        mpirun -n 1 ./varlist_test --grace-parfile ./configs/basic_config.yaml )
add_test(NAME coordinate_test_nolog COMMAND 
        mpirun -n 1 ./coordinate_test --grace-parfile ./configs/basic_config.yaml )
add_test(NAME coordinate_test_log COMMAND 
        mpirun -n 1 ./coordinate_test --grace-parfile ./configs/regrid_test_spherical_log.yaml )
add_test(NAME interpolator_test COMMAND 
        ./interpolator_test )
add_test(NAME regrid_test_cartesian COMMAND 
        mpirun -n 2 ./regrid_test --grace-parfile ./configs/regrid_test_cartesian.yaml)
add_test(NAME <regrid_test_nolog> COMMAND 
        mpirun -n 2 ./regrid_test --grace-parfile ./configs/regrid_test_spherical.yaml )
add_test(NAME regrid_test_log COMMAND 
        mpirun -n 2 ./regrid_test --grace-parfile ./configs/regrid_test_spherical_log.yaml )
add_test(NAME bc_test_unigrid COMMAND 
        mpirun -n 2 ./bc_test_unigrid --grace-parfile ./configs/bc_test_unigrid.yaml )
add_test(NAME bc_test COMMAND 
        mpirun -n 2 ./bc_test --grace-parfile ./configs/bc_test.yaml )
add_test(NAME scalar_reductions_test COMMAND 
        mpirun -n 1 ./scalar_reductions_test --grace-parfile ./configs/scalar_reductions_test.yaml )
add_test(NAME pwpoly_test COMMAND 
        mpirun -n 1 ./eos_test --grace-parfile ./configs/pwpoly_test_config.yaml )
add_test(NAME metric_test COMMAND 
        mpirun -n 1 ./metric_test --grace-parfile ./configs/basic_config.yaml )
add_test(NAME gamma2_c2p_test COMMAND 
        mpirun -n 1 ./c2p_test --grace-parfile ./configs/c2p_test_gamma2.yaml )
add_test(NAME sly4_pw_c2p_test COMMAND 
        mpirun -n 1 ./c2p_test --grace-parfile ./configs/c2p_test_sly4_pw.yaml )
add_test(NAME device_vector_test COMMAND ./device_vector_test )
add_test(NAME test_p4est_search COMMAND 
        mpirun -n 1 ./test_p4est_search --grace-parfile ./configs/h5_io_test.yaml )
add_test(NAME hdf5_output_test COMMAND 
        mpirun -n 1 ./hdf5_output_test --grace-parfile ./configs/h5_io_test.yaml )
add_test(NAME hdf5_output_sliced_test COMMAND 
        mpirun -n 1 ./test_hdf5_IO_sliced --grace-parfile ./configs/h5_io_test.yaml )