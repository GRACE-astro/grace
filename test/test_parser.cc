#include <thunder/config/config_parser.hh>
#include <thunder/utils/singleton_holder.hh>
#include <thunder/utils/creation_policies.hh>

#include <catch2/catch_test_macros.hpp>

#include <fstream>

TEST_CASE("basic_parser", "[basic_parser]")
{
    using namespace thunder ; 
    SECTION("construction")
    {
        //!< IMPORTANT: use the ampersand, copy is not allowed
        //!< and compiler messages are not always very helpful.
        auto& config = config_parser::get() ; 

        REQUIRE( config["name"].as<std::string>() == "thunder" ) ; 
    } 
    SECTION("modification")
    {
        //!< IMPORTANT: use the ampersand, copy is not allowed
        //!< and compiler messages are not always very helpful.
        auto& config = config_parser::get() ; 
        auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        config["last_modified"] = std::ctime(&t) ; 
        config.to_file("configs/basic_config_modified.yaml") ;
    }
}