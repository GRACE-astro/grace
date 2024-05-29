#include <grace/config/config_parser.hh>
#include <Kokkos_Core.hpp>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    grace::config_parser::initialize("configs/basic_config.yaml");
    int result = Catch::Session().run( argc, argv );
    Kokkos::finalize() ; 
    return result ;
}