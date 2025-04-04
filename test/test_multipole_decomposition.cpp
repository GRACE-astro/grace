/**
 * @file test_multipole_decomposition.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @version 
 * @date 2024-04-12
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/cell_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/numerics/gridloop.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/utils/numerics/spherical_harmonics.hh>
#include <grace/IO/sphere_output.hh>
#include <grace/healpix/detectors.hh>


#define DBG_REGRID_TEST

TEST_CASE("Multipole decomposition", "[multipole-decomposition]")
{
    using namespace grace::variables ; 
    using namespace grace ; 
    #if defined(GRACE_ENABLE_BURGERS) or defined(GRACE_ENABLE_SCALAR_ADV)
    int const DENS = U ; 
    int const DENS_ = U ; 
    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "U" ; 
    #else
    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "dens" ; 
    #endif

    auto const interp_order = grace::get_param<uint32_t>("amr","prolongation_order") ; 

    DECLARE_GRID_EXTENTS ; 

    /*************************************************/
    /*                Fetch arrays                   */
    /*************************************************/
    auto& state  = grace::variable_list::get().getstate()  ;
    auto& sstate  = grace::variable_list::get().getstaggeredstate()  ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 



    /*************************************************/
    /*             Get healpix specific info         */
    /*************************************************/
    auto& runtime = grace::runtime::get( ) ;   
    const int n_detectors = runtime.n_surface_output_spheres();
    const int nside = runtime.nside_surface_output_spheres();
    const int max_ell = runtime.max_degree_multipoles_surface_output_spheres();
    constexpr int spin_weight = -2; //runtime.spin_weight();
    
    /*************************************************/
    /*            Define filling func                */
    /*************************************************/


     auto const func = [&] (const int spin_weight, const int ell, const int m,
                                      VEC(const double& x,const double& y,const double &z))
    {
        auto const r = std::sqrt(x*x + y*y + z*z);
        auto const th=std::acos(z/r);
        auto const ph=std::atan2(y,x);
        
        std::array<double,2> sY_lm;
        grace::utils::multipole_spherical_harmonic(spin_weight,ell,m,
                                                   th, ph,sY_lm[0],sY_lm[1]) ;
        return sY_lm; 
    } ; 

    // assert(0==1); // testing sth

    /*************************************************/
    /*                   fill data                   */
    /*     here we fill the ghost zones as well.     */
    /*************************************************/

    /*************************************************/
    /*  We fill out a cell-centred variable and      */
    /*            corner-centred one                 */
    /*************************************************/

    // for simplicity, we will assume that our cell-centred 
    // and corner-centred fields are of the form:
    // F = Sum_l=2,3,4 Sum_m=-l,l  a_lm R(r) s_Y_lm(th , phi)
    // with R(r) = 1 
    // a_lm = multipole_index(l,m) 
    // for example:
    // a_(2,-2) = 4 
    // etc 
    // therefore, the projection of F (whether cell-centred or corner-staggered)
    // should equal the index chosen a_lm
    // <F,s_Y_lm> = Int_S2 F* s_Y_lm dA = a_lm 
    


    std::vector<double> multipole_weights{};
    multipole_weights.resize((max_ell+1)*(max_ell+1));
    // constexpr std::array<double, (max_ell+1)*(max_ell+1)> multipole_weights{};

    for( int ell = math::abs(spin_weight) ; ell <= max_ell ; ell++){
        for( int m = -ell ; m <= ell ; m++){
            const int idx_multipole = grace::utils::multipole_index(ell,m); 
            multipole_weights[idx_multipole]=idx_multipole;
            std::cout << "multipole weight: " << idx_multipole << std::endl;
            GRACE_VERBOSE("multipole weight : {}",idx_multipole );
            // printf("multipole weight : %d",idx_multipole);
            // assert(idx_multipole!=0);
        }
    }

    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ;
            h_state_mirror(VEC(i,j,k),DENS,q) = 0.;
            for( int ell = math::abs(spin_weight) ; ell <= max_ell ; ell++){
                for( int m = -ell ; m <= ell ; m++){
                    const int idx_multipole = grace::utils::multipole_index(ell,m); 
                    auto const fVal = func(spin_weight, ell, m, VEC(pcoords[0],pcoords[1],pcoords[2]));
                    // if F is real, then its coefficients in the multipole expansion must satisfy:
                    // a_(l,-m) = (-1)^m a_(l,m)*
                    // if()
                    // we need to pick a very specific form of the coefficient 
                    // to get an answer we can easily verify
                    h_state_mirror(VEC(i,j,k),DENS,q) += multipole_weights[idx_multipole] * fVal[0];
                    }
                }
        },
        {VEC(false,false,false)},
        true
    ) ; 

    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)}, 
                true
            ) ;
            h_corner_mirror(VEC(i,j,k),PSI4RE,q) = 0.0;
            h_corner_mirror(VEC(i,j,k),PSI4IM,q) = 0.0;
            for( int ell = math::abs(spin_weight) ; ell <= max_ell ; ell++){
                for( int m = -ell ; m <= ell ; m++){
                    const int idx_multipole = grace::utils::multipole_index(ell,m); 
                    auto const fVal = func(spin_weight, ell, m, VEC(pcoords[0],pcoords[1],pcoords[2]));
                    h_corner_mirror(VEC(i,j,k),PSI4RE,q) += multipole_weights[idx_multipole] * fVal[0];
                    h_corner_mirror(VEC(i,j,k),PSI4IM,q) += multipole_weights[idx_multipole] * fVal[1];
                    }
                }
            // printf("func value: %f", h_corner_mirror(VEC(i,j,k),PSI4RE,q));

        },
        {VEC(true,true,true)},
        true
    ) ; 
    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 
    /*************************************************/
    /*                   move one                       */
    /*************************************************/
    #ifdef DBG_REGRID_TEST
    /*write output and regrid*/
    grace::IO::write_cell_output(true,true,true) ; 
    //grace::amr::regrid() ;  
    grace::runtime::get().increment_iteration() ; 
    grace::runtime::get().set_timestep(1) ; 
    grace::runtime::get().increment_time(); 
    grace::IO::write_cell_output(true,true,true) ; 
    #else
    grace::amr::regrid() ;
    #endif 
    /*************************************************/
    /*                 Copy D2H                      */
    /*************************************************/
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    auto h_corner_mirror_new = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    Kokkos::deep_copy(h_corner_mirror_new,sstate.corner_staggered_fields); 
    /*************************************************/
    /*                   Check                       */
    /*************************************************/


    // host_grid_loop<false>(
    //     [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
    //         auto pcoords = coord_system.get_physical_coordinates(
    //             {VEC(i,j,k)},
    //             q,
    //             true
    //         ) ;
    //         CHECK_THAT( h_state_mirror_new(VEC(i,j,k),DENS,q)
    //                   , Catch::Matchers::WithinAbs(
    //                               h_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
    //                             , 1e-12 )) ;
    //     },
    //     {VEC(false,false,false)},
    //     false
    // ) ; 
    // host_grid_loop<false>(
    //     [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
    //         auto pcoords = coord_system.get_physical_coordinates(
    //             {VEC(i,j,k)},
    //             q,
    //             {VEC(0,0,0)},
    //             true
    //         ) ;
    //         CHECK_THAT( h_corner_mirror_new(VEC(i,j,k),DENS,q)
    //                   , Catch::Matchers::WithinAbs(
    //                               h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
    //                             , 1e-12 )) ;
    //     },
    //     {VEC(true,true,true)},
    //     false
    // ) ; 
}