#include <catch2/catch_test_macros.hpp>
#include <thunder_config.h>
#include <Kokkos_Core.hpp>
#include <thunder/amr/forest.hh>
#include <iostream>
#include <fstream>
#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/macros.hh>

TEST_CASE("amr_forest'\t'[amr_forest]")
{
    using namespace thunder ;
    auto& vars = thunder::variable_list::get() ; 

    auto coords = vars.getcoords() ; 
    auto h_coords = Kokkos::create_mirror_view(coords) ;
    Kokkos::deep_copy(h_coords, coords) ; 

    int nq = coords.extent( THUNDER_NSPACEDIM+1 ) ;
    int nx = coords.extent(0) ; 
    int ny = coords.extent(1) ; 
    #ifdef THUNDER_3D 
    int nz = coords.extent(2) ;
    #else 
    int nz = 1 ;
    #endif 
    std::cout << nq << '\t' << nx << '\t' << ny << '\t' << nz << std::endl ; 
    std::ofstream fout("coords.dat") ;
    for( int iq=0; iq<nq; ++iq ) {
        for( int ii=0; ii<nx; ii++){
            for( int ij=0; ij<ny; ++ij){
                for( int ik=0; ik<nz; ++ik){
                    fout << iq << '\t' << ii << '\t'<< ij << '\t'<< ik << '\t'<< h_coords(VEC(ii,ij,ik),iq,0) <<  '\t'
                    << h_coords(VEC(ii,ij,ik),iq,1) 
                    #ifdef THUNDER_3D
                    <<  '\t' << h_coords(VEC(ii,ij,ik),iq,2)  << std::endl ;
                    #else 
                    << std::endl ; 
                    #endif 
                }
            }
        }
    }
    fout.close() ; 
}