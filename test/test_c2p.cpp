/**
 * @file test_c2p.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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

#include <grace_config.h>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/scalar_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/system/grace_system.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>


#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

#define N 100
#define DUMP_RESIDUAL_TO_FILE

static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
compute_residual(grace::grmhd_prims_array_t const& new_prims, grace::grmhd_prims_array_t& old_prims)
{
    double err{0.} ; 
    std::array<unsigned int, 5> indices {RHOL, EPSL,ZXL,ZYL,ZZL} ; 
    for( auto const i: indices ) {
        err += math::abs((new_prims[i]-old_prims[i])/(old_prims[i]+1e-50)) ; 
    }
    return 1./indices.size() * (err) ; 
}

static void GRACE_ALWAYS_INLINE
fill_primitive_views(Kokkos::View<double *> lrho, Kokkos::View<double *> ltemp) {
    double const start_logrho{-12};
    double const end_logrho{-2.8}   ; 
    double const start_logT{-10}   ; 
    double const end_logT{2.3}    ;
    double const dlrho{(end_logrho-start_logrho)/N}, dlT{(end_logT-start_logT)/N} ; 
    Kokkos::parallel_for("fill_views", N, 
    KOKKOS_LAMBDA(int const& i) {
        lrho(i) = start_logrho + static_cast<double>(i)*dlrho ; 
        ltemp(i) = start_logT + static_cast<double>(i)*dlT ; 
    }) ; 
}

// Function to generate a random unit vector of 3 components
static void generateRandomUnitVector(double &x, double &y, double &z) {
    // Seed the random number generator
    std::srand(std::time(0));

    // Generate random angles for spherical coordinates
    double theta = ((double) std::rand() / RAND_MAX) * 2.0 * M_PI; // Random angle theta in [0, 2*pi)
    double phi = ((double) std::rand() / RAND_MAX) * M_PI;         // Random angle phi in [0, pi)

    // Convert spherical coordinates to Cartesian coordinates
    x = std::sin(phi) * std::cos(theta);
    y = std::sin(phi) * std::sin(theta);
    z = std::cos(phi);
}

static void get_velocity_from_W(double const& W, grace::metric_array_t const& metric, Kokkos::View<double ***> vel) {
    double const z = sqrt(W*W-1.) ;
    auto h_vel = Kokkos::create_mirror_view(vel) ; 
    for( int i=0; i<N;++i) { 
        for( int j=0; j<N; ++j){
            generateRandomUnitVector(h_vel(i,j,0),h_vel(i,j,1),h_vel(i,j,2)) ; 
            double n2 = metric.square_vec({h_vel(i,j,0),h_vel(i,j,1),h_vel(i,j,2)}) ; 
            //h_vel(i,j,0) = 0.; h_vel(i,j,1)=1.; h_vel(i,j,2) = 0.;
            for(int iv=0; iv<3; ++iv)
                h_vel(i,j,iv) *= z / sqrt(n2) ; 
        }
    }
    Kokkos::deep_copy(vel,h_vel) ; 
}

template<typename eos_t>
static void check_c2p(eos_t eos){
    // let's initialize the metric to something non-trivial 
    double chi = 0.25 ;
    double alp = 0.6 ;
    using namespace grace ;
    metric_array_t minkowski_metric ({1./chi,0.,0.,1./chi,0.,1./chi},{0.,0.,0.},alp) ; 
        
    double const W = 2. ; 

    Kokkos::View<double *> d_logrho("logrho", N) ; 
    Kokkos::View<double *> d_logT("logT", N) ;
    Kokkos::View<double ***> d_vel("vel", N,N,3) ; 
    Kokkos::View<double **> d_res("residual", N,N) ;
    Kokkos::View<double **> d_eps("eps", N,N) ;
    Kokkos::View<double **> d_press("press", N,N) ;
    fill_primitive_views(d_logrho,d_logT) ; 
    get_velocity_from_W(W, minkowski_metric, d_vel) ; 
    double const ye = 0.1 ; 
    
    auto atmo_params = get_atmo_params() ; 
    auto excision_params = get_excision_params() ; 
    Kokkos::parallel_for("check_c2p_residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N}),
    KOKKOS_LAMBDA( int const& i, int const& j ) {
        grmhd_prims_array_t prims ; 
        prims[RHOL] = Kokkos::pow(10.,d_logrho(i)) ; 
        prims[TEMPL] = Kokkos::pow(10.,d_logT(j))  ; 
        prims[YEL] = ye; 
        prims[ZXL] = d_vel(i,j,0) ; 
        prims[ZYL] = d_vel(i,j,1) ; 
        prims[ZZL] = d_vel(i,j,2) ; 
        prims[BXL] = prims[BYL] = prims[BZL] = 0.0 ; 
        
        double csnd2 ;
        unsigned int err ;  
        double temp{0} ; 
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_impl(prims[EPSL],csnd2,prims[ENTL],prims[TEMPL],prims[RHOL],prims[YEL],err) ; 
        grmhd_cons_array_t cons ; 
        prims_to_conservs(prims,cons,minkowski_metric) ; 
        cons[BSXL] = cons[BSYL] = cons[BSZL] = 0.0 ; 

        double rtp[3] = {1,1,1} ; //unused 
        grmhd_prims_array_t new_prims = prims ; 
        c2p_err_t c2perr ; 
        conservs_to_prims<eos_t>(cons,new_prims,minkowski_metric,eos,atmo_params,excision_params,rtp,c2perr) ; 

        d_res(i,j) = compute_residual(new_prims,prims) ;
        d_press(i,j) = new_prims[PRESSL] ;  
        //d_press(i,j) = d_vel(i,j,0)*d_vel(i,j,0) 
        //             + d_vel(i,j,1)*d_vel(i,j,1)
        //             + d_vel(i,j,2)*d_vel(i,j,2) ; 

    }) ; 
    auto h_res = Kokkos::create_mirror_view(d_res) ;
    auto h_eps = Kokkos::create_mirror_view(d_eps) ;
    auto h_press = Kokkos::create_mirror_view(d_press) ; 
    Kokkos::deep_copy(h_res,d_res) ; 
    Kokkos::deep_copy(h_eps,d_eps) ; 
    Kokkos::deep_copy(h_press,d_press) ; 
    #ifdef DUMP_RESIDUAL_TO_FILE
    std::ofstream outfile{"c2p_residual.txt"} ;
    outfile << std::setprecision(15) ; 
    auto h_rho = Kokkos::create_mirror_view(d_logrho) ; 
    Kokkos::deep_copy(h_rho,d_logrho) ;
    auto h_temp = Kokkos::create_mirror_view(d_logT) ; 
    Kokkos::deep_copy(h_temp,d_logT) ;
    int const width=20;
    #endif 
    
    for( int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            #ifdef DUMP_RESIDUAL_TO_FILE
            outfile << std::fixed << std::setprecision(16) ;
            outfile << h_rho(i)
                    << '\t' << h_temp(j)
                    << '\t' << h_eps(i,j)
                    << '\t' << h_press(i,j)
                    << '\t' << h_res(i,j) << std::endl ; 
            #endif 
            #if 1
            CHECK_THAT(
                h_res(i,j),
                Catch::Matchers::WithinAbs(0., 1e-6)
            ) ; 
            #endif
        }
    }
}

TEST_CASE("c2p", "[c2p-hydro]") {
    auto eos = grace::eos::get().get_hybrid_pwpoly() ; 
    check_c2p(eos) ; 
}    

