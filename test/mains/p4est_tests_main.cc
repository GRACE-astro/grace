/**
* See:
* https://stackoverflow.com/questions/58289895/is-it-possible-to-use-catch2-for-testing-an-mpi-code
* https://github.com/catchorg/Catch2/issues/566
*/
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <thunder_config.h> 

#include <thunder/system/mpi_runtime.hh>
#include <thunder/system/kokkos_runtime.hh>
#include <thunder/system/p4est_runtime.hh>
#include <thunder/system/thunder_runtime.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/amr/forest.hh>
#include <thunder/data_structures/variables.hh>

#include <sstream>

int main( int argc, char* argv[] ) {
    thunder::mpi_runtime::initialize(argc,argv) ; 
    thunder::kokkos_runtime::initialize(argc,argv) ; 
    thunder::p4est_runtime::initialize() ; 
    thunder::runtime::initialize() ; 
    thunder::config_parser::initialize("configs/basic_config.yaml");
    thunder::amr::connectivity::initialize() ; 
    thunder::amr::forest::initialize()       ; 
    thunder::variable_list::initialize()     ; 
    std::stringstream ss;
    /* save old buffer and redirect output to string stream */
    auto cout_buf = std::cout.rdbuf( ss.rdbuf() ); 
    int result = Catch::Session().run( argc, argv );
    /* reset buffer */
    std::cout.rdbuf( cout_buf );
    std::stringstream print_rank;
    print_rank << "Rank ";
    print_rank.width(2);
    print_rank << std::right << parallel::mpi_comm_rank() << ":\n";

    for ( int i{1}; i<parallel::mpi_comm_size(); ++i ){
        parallel::mpi_barrier(sc_MPI_COMM_WORLD);
        if ( i == parallel::mpi_comm_rank() ){
            /* if all tests are passed, it's enough if we hear that from 
             * the master. Otherwise, print results */
            if ( ss.str().rfind("All tests passed") == std::string::npos )
                std::cout << print_rank.str() + ss.str();
        }
    }
    /* have master print last, because it's the one with the most assertions */
    parallel::mpi_barrier(sc_MPI_COMM_WORLD);
    if ( parallel::mpi_comm_rank() == 0 )
        std::cout << print_rank.str() + ss.str();
    return result;
}