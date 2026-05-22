#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <Kokkos_Core.hpp>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[])
{
    // GRACE's GRACE_PRINT_* macros gate output on `parallel::mpi_comm_rank()`,
    // which aborts the process if MPI hasn't been initialised.  Init MPI
    // around the Catch session even though the kokkos tests don't otherwise
    // need it — defensive against future warning paths firing during these
    // tests.
    parallel::mpi_init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    grace::config_parser::initialize("configs/basic_config.yaml");
    int result = Catch::Session().run(argc, argv);
    Kokkos::finalize();
    parallel::mpi_finalize();
    return result;
}
