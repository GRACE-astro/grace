#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[])
{
    // GRACE's GRACE_PRINT_* macros (used by the error / unknown-parameter
    // warning paths inside the config parser and `to_file`) call
    // `parallel::mpi_comm_rank()`, which aborts the process if MPI hasn't
    // been initialised.  Treat the parser main the same as the MPI main:
    // init / finalise MPI around the Catch session.  Single-rank launch
    // (`./parser_test`) is still fine since OpenMPI supports that.
    parallel::mpi_init(&argc, &argv);
    grace::config_parser::initialize("configs/basic_config.yaml");
    int result = Catch::Session().run(argc, argv);
    parallel::mpi_finalize();
    return result;
}
