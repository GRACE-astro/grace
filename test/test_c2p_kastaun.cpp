/**
 * @file test_c2p_kastaun.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de), Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-12-20
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
#include <grace/physics/eos/grmhd_c2p_kastaun.hh>

// for the Cartesian Kerr-Schild metric
#include <grace/physics/id/FMtorus/KerrSchild.hh>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include <Kokkos_Random.hpp> 

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
    std::array<unsigned int, 5> indices {RHOL, EPSL,VXL,VYL,VZL} ; 
    for( auto const i: indices ) {
        err += math::abs((new_prims[i]-old_prims[i])/(old_prims[i]+1e-50)) ; 
    }
    return 1./indices.size() * (err) ; 
}

static void GRACE_ALWAYS_INLINE
fill_primitive_views(Kokkos::View<double *> lrho, Kokkos::View<double *> ltemp) {
    double const start_logrho{-12};
    double const end_logrho{-2.8}   ; 
    double const start_logT{-1}   ; 
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

// Function to generate a random unit vector of 3 components
static void generateRandom3DUnitVector(double &x, double &y, double &z) {

    // Generate random angles for spherical coordinates
    double theta = ((double) std::rand() / RAND_MAX) * 2.0 * M_PI; // Random angle theta in [0, 2*pi)
    double phi = ((double) std::rand() / RAND_MAX) * M_PI;         // Random angle phi in [0, pi)

    // Convert spherical coordinates to Cartesian coordinates
    x = std::sin(phi) * std::cos(theta);
    y = std::sin(phi) * std::sin(theta);
    z = std::cos(phi);
}



static void findEulerian3Velocity(grace::metric_array_t const& metric,
                                        double const& W,
                                        double& v1, double& v2, double& v3
                                        ){
    generateRandom3DUnitVector(v1,v2,v3);
    // now, we compute the norm of this vector in the true metric:
    auto Cfactor = metric.contract_vec_covec({v1,v2,v3},
                                            metric.lower({v1,v2,v3}));
    auto factor = Kokkos::sqrt( (1. - 1./(W*W)) / Cfactor);

    v1*=factor;
    v2*=factor;
    v3*=factor;
}

// note: only valid for Minkowski
static void get_velocity_from_W(double const& W, Kokkos::View<double ***> vel) {
    double const v = Kokkos::sqrt( math::int_pow<2>(W) -  1) / W; 
    auto h_vel = Kokkos::create_mirror_view(vel) ; 
    for( int i=0; i<N;++i) { 
        for( int j=0; j<N; ++j){
            generateRandomUnitVector(h_vel(i,j,0),h_vel(i,j,1),h_vel(i,j,2)) ; 
            //h_vel(i,j,0) = 0.; h_vel(i,j,1)=1.; h_vel(i,j,2) = 0.;
            for(int iv=0; iv<3; ++iv)
                h_vel(i,j,iv) *= v ; 
        }
    }
    Kokkos::deep_copy(vel,h_vel) ; 
}

static void GRACE_HOST_DEVICE normalize_smallb_magnetization(grace::metric_array_t const& metric,
                                        double const& b2,
                                        double& bUt, double& bU1, double& bU2, double& bU3
                                        ){
    // now, we compute the norm of this vector in the true metric:
    auto Cfactor = metric.contract_4dvec_4dcovec({bUt,bU1,bU2,bU3},
                                            metric.lower_4vec({bUt,bU1,bU2,bU3}));
    auto factor = Kokkos::sqrt(b2 / Cfactor);

    bUt*=factor;
    bU1*=factor;
    bU2*=factor;
    bU3*=factor;
}


static void GRACE_DEVICE find_smallb(grace::metric_array_t const& metric ,
                            const std::array<double,3>& eulVel, const double& b2,
                             Kokkos::View<double*> d_bvec) {
    auto g4=metric.gmunu();  // g4[0]  = g_tt, g4[1,2,3] = g_ti, etc
    double const v2_ = metric.square_vec({eulVel[0],eulVel[1],eulVel[2]}) ; 
    double const W_  = 1./Kokkos::sqrt(1-v2_) ; 
    const std::array<double, 4> u4 {W_*1/metric.alp(),
                                    W_*(-metric.beta(0)/metric.alp() + eulVel[0]),
                                    W_*(-metric.beta(1)/metric.alp() + eulVel[1]),
                                    W_*(-metric.beta(2)/metric.alp() + eulVel[2])
                                     };
  
    // NOTE: 
    // Setting up a random number generator on device is problematic
    // Therefore, we rresort to a fixed orientation of the spatial
    // part of the b^mu vector;
    // the relative angle between the b^i and U^i is anyway steered by 
    // the randomness in U^i 

    d_bvec(1)= 0.25;
    d_bvec(2)= -0.565;
    d_bvec(3)= 0.365;
    
    // g_munu u^mu b^nu = 0 , which uniquely determines the time-like component:
    auto g_ti_bUi = g4[1]*d_bvec(1) + g4[2]*d_bvec(2) + g4[3]*d_bvec(3);

    auto g_ti_u4Ui = g4[1]*u4[1] + g4[2]*u4[2] + g4[3]*u4[3];

    auto g_ij_bUi_u4Uj = metric.contract_vec_covec({d_bvec(1), d_bvec(2), d_bvec(3)},
                                    metric.lower({u4[1],u4[2],u4[3]}) 
                                                            );

    d_bvec(0) = (-g_ij_bUi_u4Uj -g_ti_bUi*u4[0]) / (g4[0]*u4[0] + g_ti_u4Ui ); 

    normalize_smallb_magnetization(metric, b2, d_bvec(0),d_bvec(1),d_bvec(2),d_bvec(3));
    
}



template<typename eos_t>
static void check_c2p(eos_t eos, const grace::metric_array_t metric ){
    using namespace grace ;
        
    double const W = 2. ; 

    Kokkos::View<double *> d_logrho("logrho", N) ; 
    Kokkos::View<double *> d_logT("logT", N) ;
    Kokkos::View<double ***> d_vel("vel", N,N,3) ; 
    Kokkos::View<double **> d_res("residual", N,N) ;
    Kokkos::View<double **> d_eps("eps", N,N) ;
    Kokkos::View<double **> d_press("press", N,N) ;

    fill_primitive_views(d_logrho,d_logT) ; 
    
    // only valid for minkowski
    // get_velocity_from_W(W, d_vel) ; 
    auto h_vel = Kokkos::create_mirror_view(d_vel) ; 
    // this is valid for general metrics:
     for( int i=0; i<N;++i) { 
        for( int j=0; j<N; ++j){
            findEulerian3Velocity(metric, W, h_vel(i,j,0), h_vel(i,j,1), h_vel(i,j,2));
        }
    }
    Kokkos::deep_copy(d_vel,h_vel) ; 

    double const ye = 0.1 ; 

    Kokkos::parallel_for("check_c2p_residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N}),
    KOKKOS_LAMBDA( int const& i, int const& j ) {
        grmhd_prims_array_t prims ; 
        prims[RHOL] = Kokkos::pow(10.,d_logrho(i)) ; 
        prims[TEMPL] = Kokkos::pow(10.,d_logT(j))  ; 
        prims[YEL] = ye; 
        prims[VXL] = d_vel(i,j,0) ; 
        prims[VYL] = d_vel(i,j,1) ; 
        prims[VZL] = d_vel(i,j,2) ;         
        prims[BXL] =0.;
        prims[BYL] =0.;
        prims[BZL] =0.;
        
        // primitive view holds coordinate 3-velocities
        prims[VXL] = metric.alp()*prims[VXL] - metric.beta(0) ;
        prims[VYL] = metric.alp()*prims[VYL] - metric.beta(1) ;
        prims[VZL] = metric.alp()*prims[VZL] - metric.beta(2) ;
        double csnd2 ;
        unsigned int err ;  
        //double temp{0} ; 
        prims[PRESSL] = eos.press_eps_csnd2__temp_rho_ye(prims[EPSL],csnd2,prims[TEMPL],prims[RHOL],prims[YEL],err) ;

        grmhd_cons_array_t cons ; 
        // conservs_from_prims(cons,prims,metric) ; 
        prims_to_conservs(prims,cons,metric) ;
        d_eps(i,j) = cons[STYL] ;
        grmhd_prims_array_t new_prims = prims ; 
        conservs_to_prims<eos_t, grmhd_c2p_kastaun_t>(cons,new_prims,metric,eos,0.) ; 

        d_res(i,j) = compute_residual(new_prims,prims) ;
        d_press(i,j) = new_prims[PRESSL] ; 
 
        // d_press(i,j) = d_vel(i,j,0)*d_vel(i,j,0) 
        //              + d_vel(i,j,1)*d_vel(i,j,1)
        //              + d_vel(i,j,2)*d_vel(i,j,2) ; 

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
            outfile << std::fixed << std::setprecision(15) ;
            outfile << std::left << std::setw(width) << h_rho(i)
                    << std::left << std::setw(width) << h_temp(j)
                    << std::left << std::setw(width) << h_eps(i,j)
                    << std::left << std::setw(width) << h_press(i,j)
                    << std::left << std::setw(width) << h_res(i,j) << std::endl ; 
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


template<typename eos_t>
static void check_c2p_mhd(eos_t eos, const grace::metric_array_t metric){
    using namespace grace ;
        
    double const W = 50 ;  // challenging lorentz factor
    double const b2_over_rho = 100000;  // and very high magnetization 
    GRACE_INFO("Testing Kastaun C2P in an MHD setting for a high Lorentz factor (W={}) and magnetization (sigma={})", W, b2_over_rho) ;

    Kokkos::View<double *> d_logrho("logrho", N) ; 
    Kokkos::View<double *> d_logT("logT", N) ;
    Kokkos::View<double ***> d_vel("vel", N,N,3) ; 
    Kokkos::View<double **> d_res("residual", N,N) ;
    Kokkos::View<double **> d_eps("eps", N,N) ;
    Kokkos::View<double **> d_press("press", N,N) ;
    // Allocate d_bvec for find_smallb
    Kokkos::View<double*> d_bvec("bvec", 4);

    fill_primitive_views(d_logrho,d_logT) ; 
    // only valid for minkowski
    // get_velocity_from_W(W, d_vel) ; 
    auto h_vel = Kokkos::create_mirror_view(d_vel) ; 

    // this is valid for general metrics:
     for( int i=0; i<N;++i) { 
        for( int j=0; j<N; ++j){
            findEulerian3Velocity(metric, W, h_vel(i,j,0), h_vel(i,j,1), h_vel(i,j,2));
        }
    }
    Kokkos::deep_copy(d_vel,h_vel) ; 

    double const ye = 0.1 ; 

    Kokkos::parallel_for("check_c2p_residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N}),
    KOKKOS_LAMBDA( int const& i, int const& j ) {
        grmhd_prims_array_t prims ; 
        prims[RHOL] = Kokkos::pow(10.,d_logrho(i)) ; 
        prims[TEMPL] = Kokkos::pow(10.,d_logT(j))  ; 
        prims[YEL] = ye; 
        prims[VXL] = d_vel(i,j,0) ; 
        prims[VYL] = d_vel(i,j,1) ; 
        prims[VZL] = d_vel(i,j,2) ; 
        // for the above Eulerian velocity, generate an adequate smallb (b^\mu vector)
        //   i) orthogonal to u^\mu  : g_\mu\nu u^\mu b^\nu = 0
        //   ii) its norm equal to:                     b^2 = sigma * rho

        find_smallb(metric, 
                                 {prims[VXL],prims[VYL],prims[VZL]},
                                 b2_over_rho* prims[RHOL],
                                 d_bvec);

        std::array<double,3> eulB;
        // now obtain the magnetic field in the Eulerian frame 
        get_eulerianB_from_smallb(metric,   
                                    {d_bvec(0),d_bvec(1),d_bvec(2),d_bvec(3)},
                                     {prims[VXL],prims[VYL],prims[VZL]},
                                    eulB);
        prims[BXL] = eulB[0];
        prims[BYL] = eulB[1];
        prims[BZL] = eulB[2];

        // primitive view holds coordinate 3-velocities, so we convert from Eulerian 
        prims[VXL] = metric.alp()*prims[VXL] - metric.beta(0) ;
        prims[VYL] = metric.alp()*prims[VYL] - metric.beta(1) ;
        prims[VZL] = metric.alp()*prims[VZL] - metric.beta(2) ;
        
        double csnd2 ;
        unsigned int err ;  
        //double temp{0} ; 
        prims[PRESSL] = eos.press_eps_csnd2__temp_rho_ye(prims[EPSL],csnd2,prims[TEMPL],prims[RHOL],prims[YEL],err) ;

        grmhd_cons_array_t cons ; 
        // conservs_from_prims(cons,prims,metric) ; 
        prims_to_conservs(prims,cons,metric) ;
        d_eps(i,j) = cons[STYL] ;
        grmhd_prims_array_t new_prims = prims ; 
        conservs_to_prims<eos_t, grmhd_c2p_kastaun_t>(cons,new_prims,metric,eos,0.) ; 

        d_res(i,j) = compute_residual(new_prims,prims) ;
        d_press(i,j) = new_prims[PRESSL] ; 
        
        // d_press(i,j) = d_vel(i,j,0)*d_vel(i,j,0) 
        //              + d_vel(i,j,1)*d_vel(i,j,1)
        //              + d_vel(i,j,2)*d_vel(i,j,2) ; 

    }) ; 
    auto h_res = Kokkos::create_mirror_view(d_res) ;
    auto h_eps = Kokkos::create_mirror_view(d_eps) ;
    auto h_press = Kokkos::create_mirror_view(d_press) ; 
    Kokkos::deep_copy(h_res,d_res) ; 
    Kokkos::deep_copy(h_eps,d_eps) ; 
    Kokkos::deep_copy(h_press,d_press) ; 

    #ifdef DUMP_RESIDUAL_TO_FILE
    std::ofstream outfile{"c2p_mhd_residual.txt"} ;
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
            outfile << std::fixed << std::setprecision(15) ;
            outfile << h_rho(i) << " " 
                    << h_temp(j) << " "
                    << h_eps(i,j) << " "
                    << h_press(i,j) << " "
                    << h_res(i,j) << std::endl ; 
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


TEST_CASE("c2p_mhd", "[c2p-mhd]") {
    auto eos = grace::eos::get().get_hybrid_pwpoly() ; 

    grace::metric_array_t minkowski_metric ({1.,0.,0.,1.,0.,1.},{0.,0.,0.},1.) ; 

    double x=12.0;
    double y=0;
    double z=6.0;
    double M=1.0;
    double a=0.9375;
    auto gKS = get_KS_metric(x,y,z, M,a);
    grace::metric_array_t kerr_schild_metric ({gKS[KS_GXX],gKS[KS_GXY],gKS[KS_GXZ],gKS[KS_GYY],gKS[KS_GYZ],gKS[KS_GZZ]},
                                       {gKS[KS_BETAX],gKS[KS_BETAY],gKS[KS_BETAZ]},
                                        gKS[KS_ALPHA]) ; 

    check_c2p(eos, minkowski_metric) ; 
    check_c2p_mhd(eos,minkowski_metric) ; 

    check_c2p(eos, kerr_schild_metric) ; 
    check_c2p_mhd(eos,kerr_schild_metric) ; 

    // for (int i=0; i<6; i++){
    //     for(int j=0; j<6; j++){
    //                 for(int k=0; k<6; k++){
    //                     x=-12.0+i*4.0;
    //                     y=-12.0+j*4.0;
    //                     z=-12.0+k*4.0;
    //                     grace::metric_array_t KSm ({gKS[KS_GXX],gKS[KS_GXY],gKS[KS_GXZ],gKS[KS_GYY],gKS[KS_GYZ],gKS[KS_GZZ]},
    //                                    {gKS[KS_BETAX],gKS[KS_BETAY],gKS[KS_BETAZ]},
    //                                     gKS[KS_ALPHA]);
    //                     check_c2p_mhd(eos,KSm) ; 
    //                     check_c2p(eos,KSm) ; 
    //                 }
    //     }
    // }

}    

