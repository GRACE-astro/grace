/**
 * @file test_magnetic_field_transformations.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-12-23
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

template < typename ArType >
static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
compute_difference(ArType original_field, ArType recovered_field)
{
    double err{0.} ; 
    for(size_t i = 0 ; i< original_field.size(); i++ ) {
        err += math::abs((original_field[i]-recovered_field[i])/(original_field[i]+1e-50)) ; 
    }
    return 1./original_field.size() * (err) ; 
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

// Function to generate a random unit vector in 4D
static void generateRandom4DUnitVector(double &x, double &y, double &z, double &w) {
    // Seed the random number generator
    std::srand(std::time(0));

    // Generate random angles for 4D hyperspherical coordinates
    double theta1 = ((double) std::rand() / RAND_MAX) * 2.0 * M_PI; // Random angle in [0, 2*pi)
    double theta2 = ((double) std::rand() / RAND_MAX) * M_PI;       // Random angle in [0, pi)
    double theta3 = ((double) std::rand() / RAND_MAX) * M_PI;       // Random angle in [0, pi)

    // Convert hyperspherical coordinates to Cartesian coordinates
    x = std::sin(theta3) * std::sin(theta2) * std::cos(theta1);
    y = std::sin(theta3) * std::sin(theta2) * std::sin(theta1);
    z = std::sin(theta3) * std::cos(theta2);
    w = std::cos(theta3);
}


/// @brief fill out the small b array
/// @param metric 
/// @param smallb 
/// @param eulVel  - the 3-velocity U^i of the fluid u^mu = W * (n^mu + U^mu)

static void fill_out_smallb(grace::metric_array_t const& metric , Kokkos::View<double **> smallb,
                            const std::array<double,3>& eulVel) {
    auto h_smallb = Kokkos::create_mirror_view(smallb) ; 
    auto g4=metric.gmunu();  // g4[0]  = g_tt, g4[1,2,3] = g_ti, etc
    double const v2_ = metric.square_vec({eulVel[0],eulVel[1],eulVel[2]}) ; 
    double const W_  = 1./Kokkos::sqrt(1-v2_) ; 
    const std::array<double, 4> u4 {W_*1/metric.alp(),
                                    W_*(-metric.beta(0)/metric.alp() + eulVel[0]),
                                    W_*(-metric.beta(1)/metric.alp() + eulVel[1]),
                                    W_*(-metric.beta(2)/metric.alp() + eulVel[2])
                                     };

    for( int i=0; i<N;++i) { 
        //generateRandom4DUnitVector(h_smallb(i,0),h_smallb(i,1),h_smallb(i,2),h_smallb(i,3)) ; 
        generateRandom3DUnitVector(h_smallb(i,1),h_smallb(i,2),h_smallb(i,3)) ; 
        // g_munu u^mu b^nu = 0 , which uniquely determines the time-like component:

        auto g_ti_bUi = g4[1]*h_smallb(i,1) + g4[2]*h_smallb(i,2) + g4[3]*h_smallb(i,3);

        auto g_ti_u4Ui = g4[1]*u4[1] + g4[2]*u4[2] + g4[3]*u4[3];

        auto g_ij_bUi_u4Uj = metric.contract_vec_covec({h_smallb(i,1), h_smallb(i,2), h_smallb(i,3)},
                                         metric.lower({u4[1],u4[2],u4[3]}) 
                                                                );

        h_smallb(i,0) = (-g_ij_bUi_u4Uj -g_ti_bUi*u4[0]) / (g4[0]*u4[0] + g_ti_u4Ui ); 
    }
   
    Kokkos::deep_copy(smallb,h_smallb) ; 
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

static void fill_out_eulB(Kokkos::View<double **> eulB) {
    auto h_eulB = Kokkos::create_mirror_view(eulB) ; 
    for( int i=0; i<N;++i) { 
            generateRandom3DUnitVector(h_eulB(i,0),h_eulB(i,1),h_eulB(i,2)) ; 
    }
    Kokkos::deep_copy(eulB,h_eulB) ; 
}


// we probe the Lorentz factor 
// and the co-moving magnetic field strength 
static void check_magnetic_field_transformations(grace::metric_array_t const& metric ){
    using namespace grace ; 
    double const W = 5.0;
    // first get the  3-velocity
    // device_vel
    Kokkos::View<double *> d_vel("eulU", 3) ;  // randomly generated direction
    // host_velEulerian
    auto h_vel = Kokkos::create_mirror_view(d_vel) ; 
    double const vfactor = Kokkos::sqrt( math::int_pow<2>(W) -  1) / W; 
    // Seed the random number generator
    std::srand(std::time(0));
    
    findEulerian3Velocity(metric, W, h_vel(0), h_vel(1), h_vel(2));

    Kokkos::deep_copy(d_vel,h_vel) ; 
    double const v2 = metric.square_vec({h_vel(0),h_vel(1),h_vel(2)}) ; 
    double const W_check  = 1./Kokkos::sqrt(1-v2) ; 
    printf("Test performed for W = %.3f, \n", W_check);

    // now get the original magnetic fields - b^\mu, B^i
    // note: smallb must be orthogonal to u^i that we are using;
    // otherwise, the frame transformations employed in grmhd_helpers.hh will not work,
    // as they are derived under the assumption of specific relations holding between frames (more concretely, g_munu b^mu u^mu = 0)
    Kokkos::View<double **> d_smallb("smallb", N,4) ;  // N vectors to randomize, each has 4 components, 
    // the normal magnetic field has no such constraint because we recover the 4-d b^\mu from B^i and U^i,
    // there is a-priori no restriction regarding B^i
    Kokkos::View<double **> d_eulB  ("eulB"  , N,3) ;  // N vectors to randomize, each has 3 components, 

    // fill them out:
    fill_out_eulB(d_eulB);
    // smallb cannot be arbitrary, like B^i, but instead has to be orthogonal to the chosen u^i!
    fill_out_smallb(metric, d_smallb, {h_vel(0),h_vel(1),h_vel(2)});

    Kokkos::View<double *> d_res_tocomov("residual-comov", N) ;
    Kokkos::View<double *> d_res_toeuler("residual-euler", N) ;

    Kokkos::parallel_for("check_transformations",
        Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA( int const& i ) {

        // forward transformations 
        std::array<double, 4> smallb_from_eulB;
        std::array<double, 3> eulB_from_smallb;

        get_smallb_from_eulerianB(metric, 
                                {d_eulB(i,0),d_eulB(i,1),d_eulB(i,2)},
                                {d_vel(0),d_vel(1),d_vel(2)},
                                    smallb_from_eulB);

        get_eulerianB_from_smallb(metric,  
                                    std::array<double,4>{d_smallb(i,0),d_smallb(i,1),d_smallb(i,2),d_smallb(i,3)},
                                    std::array<double,3>{d_vel(0),d_vel(1),d_vel(2)},
                                    eulB_from_smallb);

        // backward transformations 
        std::array<double, 3> eulB_recovered;
        std::array<double, 4> smallb_recovered;

        get_smallb_from_eulerianB(metric,                        
                                    eulB_from_smallb,
                                    std::array<double,3>{d_vel(0),d_vel(1),d_vel(2)},
                                    smallb_recovered);

        get_eulerianB_from_smallb(metric, 
                                    smallb_from_eulB,
                                    std::array<double,3>{d_vel(0),d_vel(1),d_vel(2)},
                                    eulB_recovered);
      
        // compute differences between final and original fields
        d_res_tocomov(i) = compute_difference(std::array<double,4>{d_smallb(i,0),d_smallb(i,1),d_smallb(i,2),d_smallb(i,3)},
                                              smallb_recovered
                                            ) ;

        d_res_toeuler(i) = compute_difference(std::array<double,3>{d_eulB(i,0),d_eulB(i,1),d_eulB(i,2)},
                                            eulB_recovered
                                            ) ;
                        
    }) ; 
    auto h_res_toeuler = Kokkos::create_mirror_view(d_res_toeuler) ;
    auto h_res_tocomov = Kokkos::create_mirror_view(d_res_tocomov) ;
    Kokkos::deep_copy(h_res_toeuler,d_res_toeuler) ; 
    Kokkos::deep_copy(h_res_tocomov,d_res_tocomov) ; 

    
    for( int i=0; i<N; ++i){
        #if 1
        CHECK_THAT(
            h_res_toeuler(i),
            Catch::Matchers::WithinAbs(0., 1e-10)
        ) ; 
        CHECK_THAT(
            h_res_tocomov(i),
            Catch::Matchers::WithinAbs(0., 1e-10)
        ) ; 
        #endif
    
    }
}


TEST_CASE("magneticfield", "[magnetic-field-transformations]") {

    grace::metric_array_t minkowski_metric ({1.,0.,0.,1.,0.,1.},{0.,0.,0.},1.) ; 
    check_magnetic_field_transformations(minkowski_metric) ; 

    // gxx, gxy, gxz, gyy, gyz, gzz, betax,betay,betaz, alpha
    auto M=1.0;
    auto Psi = [&](const double r){return math::int_pow<4>(1. + M/(2*r));};
    auto alp = [&](const double r){return (  ( 1. - M/(2*r)) / ( 1. + M/(2*r))   );};
    auto rad=M*8.0;

    grace::metric_array_t isotropic_schwarzschild ({Psi(rad),0.,0.,Psi(rad),0.,Psi(rad)},{0.,0.,0.},alp(rad)) ; 
    printf("Psi %f, alp %f \n", Psi(rad), alp(rad));
    check_magnetic_field_transformations(isotropic_schwarzschild) ; 

}    

