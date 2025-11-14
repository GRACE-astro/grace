/**
* See:
* https://stackoverflow.com/questions/58289895/is-it-possible-to-use-catch2-for-testing-an-mpi-code
* https://github.com/catchorg/Catch2/issues/566
*/
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <grace_config.h> 

#include <grace/system/grace_initialize.hh>
#include <iostream>
#include <grace/parallel/mpi_wrappers.hh>
#include <sstream>
#include <grace/evolution/initial_data.hh>
#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] ) {
    Kokkos::printf("MAIN: Entered main()\n");
    
    grace::initialize(argc,argv);
    Kokkos::printf("MAIN: grace::initialize() complete\n");

    
    grace::set_initial_data();
    Kokkos::printf("Setting initial data.\n");
    
    std::stringstream ss;
    Kokkos::printf("MAIN: Created stringstream\n");
    
    /* save old buffer and redirect output to string stream */
    auto cout_buf = std::cout.rdbuf( ss.rdbuf() );
    Kokkos::printf("MAIN: Redirected cout to stringstream\n");
    
    Kokkos::printf("MAIN: About to call Catch::Session().run()\n");
    int result = Catch::Session().run( argc, argv );
    Kokkos::printf("MAIN: Catch::Session().run() returned with result=%d\n", result);
    
    /* reset buffer */
    std::cout.rdbuf( cout_buf );
    Kokkos::printf("MAIN: Reset cout buffer\n");
    
    std::stringstream print_rank;
    print_rank << "Rank ";
    print_rank.width(2);
    print_rank << std::right << parallel::mpi_comm_rank() << ":\n";
    Kokkos::printf("MAIN: Created print_rank stream\n");
    
    for ( int i{1}; i<parallel::mpi_comm_size(); ++i ){
        Kokkos::printf("MAIN: Barrier loop iteration i=%d\n", i);
        parallel::mpi_barrier(sc_MPI_COMM_WORLD);
        if ( i == parallel::mpi_comm_rank() ){
            /* if all tests are passed, it's enough if we hear that from 
             * the master. Otherwise, print results */
            if ( ss.str().rfind("All tests passed") == std::string::npos )
                std::cout << print_rank.str() + ss.str();
        }
    }
    
    Kokkos::printf("MAIN: After barrier loop\n");
    
    /* have master print last, because it's the one with the most assertions */
    parallel::mpi_barrier(sc_MPI_COMM_WORLD);
    Kokkos::printf("MAIN: Final barrier complete\n");
    
    if ( parallel::mpi_comm_rank() == 0 )
        std::cout << print_rank.str() + ss.str();
    
    Kokkos::printf("MAIN: About to return result=%d\n", result);
    return result;
}