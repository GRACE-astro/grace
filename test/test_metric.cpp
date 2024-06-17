/**
 * @file test_metric.cpp
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
#include <grace/utils/metric_utils.hh>
#include <grace/system/grace_system.hh>
#include <iostream>
#include <fstream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#define DEEP_COPY_VEC_TO_VIEW(vec,view) \
            do { \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                for( int i=0; i < vec.size(); ++i){                 \
                    host_view(i) = vec[i] ;                         \
                }                                                   \
                Kokkos::deep_copy(view,host_view) ;                 \
            } while(0)
#define DEEP_COPY_VIEW_TO_VEC(view,vec)                             \
            do {                                                    \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                Kokkos::deep_copy(host_view,view) ;                 \
                vec.resize(host_view.extent(0));                    \
                for( int i=0; i < vec.size(); ++i){                 \
                    vec[i] = host_view(i) ;                         \
                }                                                   \
            } while(0)
#define DEEP_COPY_VIEW_TO_SYMBOL(view, var)                         \
            do {                                                    \
                auto host_view = Kokkos::create_mirror_view(view) ; \
                Kokkos::deep_copy(host_view,view) ;                 \
                var = host_view(0) ;                                \
            } while(0)

TEST_CASE("metric", "[metric-tests]") {
    /* First: test copying metric into a kernel */
    Kokkos::View<double*> v("v",3), v2("v2",1) ;
    std::vector<double> vh{1,1,1} ; 
    DEEP_COPY_VEC_TO_VIEW(vh,v) ; 
    grace::metric_array_t minkowski(
          {1.,0.,0.,1.,0.,1.}
        , {0.,0.,0.}
        , 1.
    ) ; 
    Kokkos::parallel_for("Test-contraction",1,
    KOKKOS_LAMBDA (int const& i) {
        v2(0) = minkowski.square_vec({v(0),v(1),v(2)}) ; 
    }) ; 
    double v2h ;
    DEEP_COPY_VIEW_TO_SYMBOL(v2,v2h) ; 
    CHECK_THAT(
            v2h,
            Catch::Matchers::WithinAbs(3., 1e-4)
        ) ;
    Kokkos::parallel_for("Test-contraction",1,
    KOKKOS_LAMBDA (int const& i) {
        v2(0) = minkowski.square_covec({v(0),v(1),v(2)}) ; 
    }) ; 
    DEEP_COPY_VIEW_TO_SYMBOL(v2,v2h) ; 
    CHECK_THAT(
            v2h,
            Catch::Matchers::WithinAbs(3., 1e-4)
        ) ;
    /* Second: Try constructing on device */
    Kokkos::parallel_for("Test-contraction",1,
    KOKKOS_LAMBDA (int const& i) {
        grace::metric_array_t minkowski2(
          {1.,0.,0.,1.,0.,1.}
        , {0.,0.,0.}
        , 1.
        ) ; 
        v2(0) = minkowski2.square_vec({v(0),v(1),v(2)}) ; 
    }) ;
    DEEP_COPY_VIEW_TO_SYMBOL(v2,v2h) ; 
    CHECK_THAT(
            v2h,
            Catch::Matchers::WithinAbs(3., 1e-4)
        ) ;
    Kokkos::View<double*> beta("beta",3), gamma("gamma",6) 
                        , gammainv("gammainv", 6)
                        , alp("alp", 1) 
                        , sqrtg("sqrtg", 1);
    
    Kokkos::parallel_for("Test-contraction",1,
    KOKKOS_LAMBDA (int const& i) {
        for( int iv=0; iv<3; ++iv)
            beta(iv) = minkowski.beta(iv) ; 
        for( int iv=0; iv<6; ++iv){
            gamma(iv) = minkowski.gamma(iv);
            gammainv(iv) = minkowski.invgamma(iv);
        } 

        alp(0) = minkowski.alp()     ; 
        sqrtg(0) = minkowski.sqrtg() ; 
    }) ;
    /* Third: check metric function retrieval methods */
    std::vector<double> hbeta, hgamma, hgammainv ;
    double halp, hsqrtg ; 
    DEEP_COPY_VIEW_TO_VEC(beta,hbeta) ; 
    DEEP_COPY_VIEW_TO_VEC(gamma,hgamma) ; 
    DEEP_COPY_VIEW_TO_VEC(gammainv,hgammainv) ; 
    DEEP_COPY_VIEW_TO_SYMBOL(alp,halp) ; 
    DEEP_COPY_VIEW_TO_SYMBOL(sqrtg,hsqrtg) ; 

    std::vector<double> gamma_ex{ 1., 0.,0., 1.,0., 1.} ;
    for( int iv=0; iv<3; ++iv) {
        CHECK_THAT(
            hbeta[iv],
            Catch::Matchers::WithinAbs(0., 1e-12)
        ) ;
    }
    for( int iv=0; iv<6; ++iv) {
        CHECK_THAT(
            hgamma[iv],
            Catch::Matchers::WithinAbs(gamma_ex[iv], 1e-12)
        ) ;
        CHECK_THAT(
            hgammainv[iv],
            Catch::Matchers::WithinAbs(gamma_ex[iv], 1e-12)
        ) ;
    }
    CHECK_THAT(
            halp,
            Catch::Matchers::WithinAbs(1., 1e-12)
        ) ;
    CHECK_THAT(
            hsqrtg,
            Catch::Matchers::WithinAbs(1., 1e-12)
        ) ;
    /* Test other vector related methods */
    Kokkos::View<double*> vlow("v_low",3), vup("v_up",3) ;
    Kokkos::parallel_for("Test-contraction",1,
    KOKKOS_LAMBDA (int const& i) {
        auto const vl = minkowski.lower({v(0),v(1),v(2)}) ; 
        auto const vu = minkowski.raise({v(0),v(1),v(2)}) ; 
        for( int iv=0; iv<3; ++iv) {
            vlow(iv)= vl[i] ; 
            vup(iv) = vu[i] ;
        }
    }) ; 
    std::vector<double> vuh, vlh ; 
    DEEP_COPY_VIEW_TO_VEC(vlow,vlh) ;
    DEEP_COPY_VIEW_TO_VEC(vup,vuh) ; 
    for( int iv=0; iv<3; ++iv ) {
        CHECK_THAT(
            vlh[iv],
            Catch::Matchers::WithinAbs(vh[iv], 1e-12)
        ) ;
        CHECK_THAT(
            vuh[iv],
            Catch::Matchers::WithinAbs(vh[iv], 1e-12)
        ) ;
    }
}