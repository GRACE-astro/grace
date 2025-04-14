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
    // using namespace grace::variables ; 
    using namespace grace ; 
    using namespace Kokkos ;

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
    auto& aux   = grace::variable_list::get().getaux()   ;
    auto& saux   = grace::variable_list::get().getstaggeredaux()   ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    auto h_aux_mirror = Kokkos::create_mirror_view(aux) ; 
    auto h_corner_aux_mirror = Kokkos::create_mirror_view(saux.corner_staggered_fields) ; 



    /*************************************************/
    /*             Get healpix specific info         */
    /*************************************************/
    auto& runtime = grace::runtime::get( ) ;   
    
    const int n_detectors = 1;
    //const int nside = 128; // not used in this example 
    const int nside = 256; // not used in this example 
    const int ntheta = 400;
    const int nphi = 800;

    // const int ntheta = 200;
    // const int nphi = 360;
    
    const int max_ell = 4;
    std::array<double,3> centre{0,0,0}; 
    const double radius = 0.2; 
    const std::string sphere_name = "test_detector"; 
    //const std::string sphere_type = "uniform"; 
    const std::string sphere_type = "healpix"; 
    
    std::vector<std::array<double,3>> output_spheres_centres ;
    std::vector<double>               output_spheres_radii   ;
    std::vector<std::string> output_spheres_names            ;
    std::vector<std::string> output_spheres_types            ;

    output_spheres_centres.push_back(centre);
    output_spheres_radii.push_back(radius);
    output_spheres_names.push_back(sphere_name);
    output_spheres_types.push_back(sphere_type);

    constexpr int spin_weight = -2; //runtime.spin_weight();
    // constexpr int spin_weight = 0; 
    
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
        }
    }

    // host_grid_loop<true>(
    //     [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
    //         auto pcoords = coord_system.get_physical_coordinates(
    //             {VEC(i,j,k)},
    //             q,
    //             true
    //         ) ;
    //         h_state_mirror(VEC(i,j,k),DENS,q) = 0.;
    //         for( int ell = math::abs(spin_weight) ; ell <= max_ell ; ell++){
    //             for( int m = -ell ; m <= ell ; m++){
    //                 const int idx_multipole = grace::utils::multipole_index(ell,m); 
    //                 auto const fVal = func(spin_weight, ell, m, VEC(pcoords[0],pcoords[1],pcoords[2]));
    //                 // if F is real, then its coefficients in the multipole expansion must satisfy:
    //                 // a_(l,-m) = (-1)^m a_(l,m)*
    //                 // if()
    //                 // we need to pick a very specific form of the coefficient 
    //                 // to get an answer we can easily verify
    //                 h_state_mirror(VEC(i,j,k),DENS,q) += multipole_weights[idx_multipole] * fVal[0];
    //                 }
    //             }
    //     },
    //     {VEC(false,false,false)},
    //     true
    // ) ; 

    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)}, 
                true
            ) ;
            
            h_corner_aux_mirror(VEC(i,j,k),PSI4RE,q) = 0.0;
            h_corner_aux_mirror(VEC(i,j,k),PSI4IM,q) = 0.0;

            
            // h_corner_aux_mirror
            // make a combination..:
            for( int ell = math::abs(spin_weight) ; ell <= max_ell ; ell++){
                for( int m = -ell ; m <= ell ; m++){
                    const int idx_multipole = grace::utils::multipole_index(ell,m); 
                    auto const fVal = func(spin_weight, ell, m, VEC(pcoords[0],pcoords[1],pcoords[2]));
                    //GRACE_VERBOSE("fVals are {}, {}", fVal[0], fVal[1]);
                    h_corner_aux_mirror(VEC(i,j,k),PSI4RE,q) += multipole_weights[idx_multipole] * fVal[0];
                    h_corner_aux_mirror(VEC(i,j,k),PSI4IM,q) += multipole_weights[idx_multipole] * fVal[1];
                    }
                }
            // just use a single mode: 
            // this has idx_multipole = 8 
            // auto const fVal = func(spin_weight, 2, 2, VEC(pcoords[0],pcoords[1],pcoords[2]));
            // auto const fVal = func(spin_weight, 0, 0, VEC(pcoords[0],pcoords[1],pcoords[2]));
            // printf("fvals: %f, %f", fVal[0], fVal[1]);
            // h_corner_aux_mirror(VEC(i,j,k),PSI4RE,q) = fVal[0];
            // h_corner_aux_mirror(VEC(i,j,k),PSI4IM,q) = fVal[1];

            // printf("Harmonic values at: %")

        },
        {VEC(true,true,true)},
        true
    ) ; 


    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 
    Kokkos::deep_copy(aux,h_aux_mirror); 
    Kokkos::deep_copy(saux.corner_staggered_fields,h_corner_aux_mirror); 
    Kokkos::fence();

    /*************************************************/
    /*                  Multipoles                   */
    /*************************************************/

    // prepare the spin-weighted spherical harmonics for computation: 
    using Complex = Kokkos::complex<double>;
    using HostM = Kokkos::HostSpace;

    Kokkos::View<Complex**, HostM> sw_sph_harmonics = Kokkos::View<Complex**, HostM>();
{
    if(sw_sph_harmonics.extent(0)==0 and sw_sph_harmonics.extent(1)==0){  // if data=0
        grace::IO::update_spin_weighted_spherical_harmonics(sw_sph_harmonics, spin_weight, max_ell, nside, ntheta, nphi, grace::IO::SPHERICAL_GRID_TYPE::HEALPIX);
        GRACE_VERBOSE("SWSH updated. Current size: {} x {}.", sw_sph_harmonics.extent(0),sw_sph_harmonics.extent(1) ) ; 
    }

    grace::IO::initialize_spherical_detectors(n_detectors, nside,
                                                ntheta, nphi,
                                                output_spheres_centres,
                                                output_spheres_radii,
                                                output_spheres_names,
                                                output_spheres_types );

    grace::IO::update_spherical_detectors();

    GRACE_VERBOSE("Updated spherical surfaces info - multipole computation.") ; 

    grace::IO::compute_spherical_surface_variable_data({"Psi4Re", "Psi4Im"}, {},{},
                                             {},{},{});

    GRACE_VERBOSE("Interpolated variables on spherical surfaces for multipole decomposition.") ; 

    std::map< std::string, Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> > complex_det_surface_data; 

    for (auto& [name, detector] : grace::IO::detectors) {
            GRACE_VERBOSE("Detector name: {} ", name);
            std::vector<int> det_indices = detector.get_local_rank_sphere_indices();
            std::map<std::string,std::vector<double>> det_surface_data = detector.get_local_rank_detector_surface_data();   

            complex_det_surface_data = grace::IO::complexify_detector_data(det_surface_data)  ; 

            // osc TOV (hydro w/o hydro) - strong field test 
            // l=m=2 
            // id_kernel ["quadrupole solution"]? 
            // plane wave  - weak field test
            // to do: port this routine from CPU to GPU 
            auto const det_all_multipoles  = grace::IO::get_all_multipoles(spin_weight, max_ell, nside, ntheta, nphi,
                                                    detector.grid_type, det_indices, sw_sph_harmonics, complex_det_surface_data);


            for (auto const& [var_name, var_data] : det_all_multipoles){
                    GRACE_VERBOSE(" var_name {}", var_name);

                    for (int idx_multipole = 0 ; idx_multipole < var_data.extent(0); idx_multipole++ ){
                            GRACE_INFO(" var_name, multipole_index, Re, Im : {}, {}, {}, {}",
                                         var_name, idx_multipole,
                                        var_data(idx_multipole).real(),
                                        var_data(idx_multipole).imag() );
                    }
                }
    }}


    
    // finalize the SWSH 
    sw_sph_harmonics = Kokkos::View<Complex**, HostM>();;

 
    //         CHECK_THAT( h_state_mirror_new(VEC(i,j,k),DENS,q)
    //                   , Catch::Matchers::WithinAbs(
    //                               h_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
    //                             , 1e-12 )) ;

}