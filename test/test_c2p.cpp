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
#include <grace/physics/c2p.hh>
#include <grace/system/grace_system.hh>
#include <iostream>
#include <fstream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#define N 10000

static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
conservs_from_prims(grmhd_cons_array_t& cons, grmhd_prims_array_t& prims, metric_array_t const& metric)
{
    double const v2 = metric.square_vec({prims[VXL],prims[VYL],prims[VZL]}) ; 
    double const W  = 1./Kokkos::sqrt(1-v2) ; 
    cons[DENSL] = W * metric.sqrtgamma() * prims[RHOL] ; 

}

static void fill_data_vectors( std::vector<double>& rho
                             , std::vector<double>& press
                             , std::vector<double>& eps ) 
{
    std::ifstream file("sly_eos_test.txt");
    if (!file.is_open()) {
        return ;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double val1, val2, val3;
        if (!(iss >> val1 >> val2 >> val3)) {
            std::cerr << "Error reading line: " << line << std::endl;
            return ;
        }
        rho.push_back(val1);
        press.push_back(val2);
        eps.push_back(val3);
    }
    file.close();
}

TEST_CASE("c2p", "[c2p-hydro]") {
    auto eos = grace::eos::get().get_hybrid_pwpoly() ; 

    metric_array_t minkowski_metric ({1.,0.,0.,1.,0.,1.},{0.,0.,0.},1.) ; 
        
    Kokkos::View<double *> d_w_lorentz("W", N) ; 
    Kokkos::View<double *> d_logrho("logrho", N) ; 
    Kokkos::View<double *> d_logT("logT", N) ; 


    auto h_press = Kokkos::create_mirror_view(d_press);
    auto h_eps   = Kokkos::create_mirror_view(d_eps)  ;

    std::vector<double> rho,press,eps ; 

    fill_data_vectors(rho,press,eps) ; 


    #define DEEP_COPY_VEC_TO_VIEW(vec,view) \
            do { \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                for( int i=0; i < vec.size(); ++i){                 \
                    host_view(i) = vec[i] ;                         \
                }                                                   \
                Kokkos::deep_copy(view,host_view) ;                 \
            } while(0)
    
    DEEP_COPY_VEC_TO_VIEW(rho,d_rho) ; 





    Kokkos::parallel_for("pwp_test_fill",N,
    KOKKOS_LAMBDA (int i){
        double eps,csnd2; 
        double temp{0.}, rho{2e-11}, ye{0} ; 
        unsigned int err ; 
        d_press(i) = eos.press_eps_csnd2__temp_rho_ye(eps,csnd2,temp,d_rho(i),ye,err) ;
        d_eps(i)   = eps;  
    }) ; 
    Kokkos::deep_copy(h_press,d_press);
    Kokkos::deep_copy(h_eps, d_eps)   ;
    
    for( int i=0; i<N; ++i){
        CHECK_THAT(
            h_press(i),
            Catch::Matchers::WithinAbs(press[i], 1e-4)
        ) ; 
        CHECK_THAT(
            h_eps(i),
            Catch::Matchers::WithinAbs(eps[i], 1e-4)
        ) ; 
    }
}