/**
 * @file test_tabulated_eos.cpp
 * @author Khalil Pierre (khalil3.14erre@gmail.com)
 * @brief 
 * @date 2025-02-05
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
#include <grace/system/print.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_setup.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <iostream>
#include <fstream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <hdf5.h>

static void check_find_index(grace::tabulated_eos_t table) {
    auto h_logrho = Kokkos::create_mirror_view(table.get_logrho()) ;
    auto h_logtemp = Kokkos::create_mirror_view(table.get_logtemp()) ;
    auto h_yes = Kokkos::create_mirror_view(table.get_yes()) ;

    Kokkos::deep_copy(h_logrho, table.get_logrho()) ;
    Kokkos::deep_copy(h_logtemp, table.get_logtemp()) ;
    Kokkos::deep_copy(h_yes, table.get_yes()) ;

    std::cout << "---------------------RHO COORDINATE CHECK---------------------" << std::endl ; 

    Kokkos::View<int *, grace::default_space> rho_results("rhoresults", table.get_logrho().size()) ;
    auto h_rho_results = Kokkos::create_mirror_view(rho_results) ;

    Kokkos::parallel_for("Check_logrho", table.get_logrho().size(), KOKKOS_LAMBDA (int i){
        double coord_spacing = table.get_logrho(1) - table.get_logrho(0) ;
        double inverse_coord_spacing = 1. / coord_spacing;

        rho_results(i) = table._find_index_uniform(table.get_logrho(), inverse_coord_spacing, table.get_logrho(i) + (coord_spacing / 2.)) ;

    }) ;

    Kokkos::deep_copy(h_rho_results, rho_results) ;

    Kokkos::fence() ;

    for (int i = 0; i < h_logrho.size(); ++i) {

        if (i < h_logrho.size() - 2) {
            REQUIRE_THAT(h_rho_results(i), Catch::Matchers::WithinAbs(i, 1e-10)) ;
        } else {
            REQUIRE_THAT(h_rho_results(i), Catch::Matchers::WithinAbs(h_logrho.size() - 2, 1e-10)) ;
        }
        
    }

    std::cout << "---------------------TEMP COORDINATE CHECK---------------------" << std::endl ; 

    Kokkos::View<int *, grace::default_space> temp_results("tempresults", table.get_logtemp().size()) ;
    auto h_temp_results = Kokkos::create_mirror_view(temp_results) ;

    Kokkos::parallel_for("Check_logtemp", table.get_logtemp().size(), KOKKOS_LAMBDA (int i){
        double coord_spacing = table.get_logtemp(1) - table.get_logtemp(0) ;
        double inverse_coord_spacing = 1. / coord_spacing;

        temp_results(i) = table._find_index_uniform(table.get_logtemp(), inverse_coord_spacing, table.get_logtemp(i) + (coord_spacing / 2.)) ;

    }) ;

    Kokkos::deep_copy(h_temp_results, temp_results) ;

    Kokkos::fence() ;

    for (int i = 0; i < h_logtemp.size(); ++i) {

        if (i < h_logtemp.size() - 2) {
            REQUIRE_THAT(h_temp_results(i), Catch::Matchers::WithinAbs(i, 1e-10)) ;
        } else {
            REQUIRE_THAT(h_temp_results(i), Catch::Matchers::WithinAbs(h_logtemp.size() - 2, 1e-10)) ;
        }
        
    }


    std::cout << "---------------------YES COORDINATE CHECK---------------------" << std::endl ; 

    Kokkos::View<int *, grace::default_space> yes_results("yesresults", table.get_yes().size()) ;
    auto h_yes_results = Kokkos::create_mirror_view(yes_results) ;

    Kokkos::parallel_for("Check_yes", table.get_yes().size(), KOKKOS_LAMBDA (int i){
        double coord_spacing = table.get_yes(1) - table.get_yes(0) ;
        double inverse_coord_spacing = 1. / coord_spacing;

        yes_results(i) = table._find_index_uniform(table.get_yes(), inverse_coord_spacing, table.get_yes(i) + (coord_spacing / 2.)) ;

    }) ;

    Kokkos::deep_copy(h_yes_results, yes_results) ;

    Kokkos::fence() ;

    for (int i = 0; i < h_yes.size(); ++i) {

        if (i < h_yes.size() - 2) {
            REQUIRE_THAT(h_yes_results(i), Catch::Matchers::WithinAbs(i, 1e-10)) ;
        } else {
            REQUIRE_THAT(h_yes_results(i), Catch::Matchers::WithinAbs(h_yes.size() - 2, 1e-10)) ;
        }
        
    }

}

static void check_interpolation(grace::tabulated_eos_t table) {

    using namespace grace ;


    //Creating mirror views for the lrho, ltemp and yes coordinates
    auto h_logrho = Kokkos::create_mirror_view(table.get_logrho()) ;
    auto h_logtemp = Kokkos::create_mirror_view(table.get_logtemp()) ;
    auto h_yes = Kokkos::create_mirror_view(table.get_yes()) ;

    //Coordinate spacing is needed on host
    auto h_coord_spacing = Kokkos::create_mirror_view(table._coord_spacing) ;
    auto h_inverse_coord_spacing = Kokkos::create_mirror_view(table._inverse_coord_spacing) ;

    //copy coordinate date from device too host
    Kokkos::deep_copy(h_logrho, table.get_logrho()) ;
    Kokkos::deep_copy(h_logtemp, table.get_logtemp()) ;
    Kokkos::deep_copy(h_yes, table.get_yes()) ;

    //copy coordinate spacing 
    Kokkos::deep_copy(h_coord_spacing, table._coord_spacing) ;
    Kokkos::deep_copy(h_inverse_coord_spacing, table._inverse_coord_spacing) ;
    
    //Want to interpolate at cell center and edges so +1 points are needed
    const int nx = h_logrho.size() + 9 ; 
    const int ny = h_logtemp.size() + 9 ; 
    const int nz = h_yes.size() + 9 ;

    //The spacing for the lrho, ltemp and yes coordinates is calculated
    const double dx = h_coord_spacing(tabulated_eos_t::dim::rho) ;
    const double dy = h_coord_spacing(tabulated_eos_t::dim::temp) ;
    const double dz = h_coord_spacing(tabulated_eos_t::dim::yes) ;

    std::cout << "The value for dx = " << dx << std::endl;

    //The zeroth midpoint is calculated here this will be the start position for the interpelation
    const double rho_midpoint0 = h_logrho(0) - 9 * dx / 2 ;
    const double temp_midpoint0 = h_logtemp(0) - 9 * dy / 2 ;
    const double yes_midpoint0 = h_yes(0) - 9 * dz / 2;

    Kokkos::View<double***, grace::default_space> interptable("InterpTable", nx, ny, nz) ;
    auto h_interptable = Kokkos::create_mirror_view(interptable) ;

    Kokkos::parallel_for("fill_interpolation_table"
    , Kokkos::MDRangePolicy<Kokkos::Rank<3>, default_execution_space>({0, 0, 0}, {nx, ny, nz})
    , KOKKOS_LAMBDA(int i, int j, int k) {

        //Position is calculated for corresponding i, j, k
        double rho_midpoint = rho_midpoint0 + i * dx ;
        double temp_midpoint = temp_midpoint0 + j * dy ;
        double yes_midpoint = yes_midpoint0 + k * dz ;

        //The midpoint positions are used to and the fake pressure table is used
        //to calculate 
        interptable(i, j, k) = table._interpolate_table( tabulated_eos_t::EV::PRESS
                                                        , rho_midpoint
                                                        , temp_midpoint
                                                        , yes_midpoint) ;

    });

    Kokkos::deep_copy(h_interptable, interptable) ;

    auto const z_host = [&] ( double x, double y, double z ) {
        return 2.5*x + 4.2*y - 5.1*z + 3.7  ; 
    } ; 
    
    
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            
            double rho_midpoint = rho_midpoint0 + i * dx ;
            double temp_midpoint = temp_midpoint0 + j * dy ;
            double yes_midpoint = yes_midpoint0 + k * dz ; 

            double Z = z_host(rho_midpoint, temp_midpoint, yes_midpoint) ;
            
            //std::cout << "Current i = " << i << ", j = " << j << ", k = " << k << std::endl ;
            //std::cout << "Current rho = " << rho_midpoint << ", temp = " << temp_midpoint << ", yes = " << yes_midpoint << std::endl ;
            REQUIRE_THAT(h_interptable(i, j, k), Catch::Matchers::WithinAbs(Z, 1e-10)) ;
    }     
}


TEST_CASE("Tabulated EOS", "[pwpolytrope]")
{
    std::string table_path = "/home/it4i-kpierre/data/DD2+VQCD_soft_quark_fraction.h5";  

    grace::tabulated_eos_t _tabulated = grace::setup_tabulated_eos_compose(table_path.c_str(), true);

    check_find_index(_tabulated) ;
    check_interpolation(_tabulated) ;

    auto host_table = Kokkos::create_mirror_view( _tabulated.get_table(grace::tabulated_eos_t::EV::PRESS) ) ;

    //Want to print out 1 dimensional slice of table





    // auto h_logrho = Kokkos::create_mirror_view(_tabulated.get_logrho()) ;
    // auto h_logtemp = Kokkos::create_mirror_view(_tabulated.get_logtemp()) ;
    // auto h_yes = Kokkos::create_mirror_view(_tabulated.get_yes()) ;

    // Kokkos::deep_copy(h_logrho, _tabulated.get_logrho()) ;
    // Kokkos::deep_copy(h_logtemp, _tabulated.get_logtemp()) ;
    // Kokkos::deep_copy(h_yes, _tabulated.get_yes()) ;

    // Kokkos::View<int[1], grace::default_space> result("result") ;
    // auto h_result = Kokkos::create_mirror_view(result) ;


    // Kokkos::parallel_for("EVOL", 10, KOKKOS_LAMBDA (int i) {
        
    //     double coordinate_spacing = _tabulated.get_yes(1) - _tabulated.get_yes(0) ;
    //     double inverse_coordinate_spacing = 1 / coordinate_spacing ;

    //     Kokkos::printf("yes 0 is given by = %f\n", _tabulated.get_yes(0)) ;
        
    //     result(0) = _tabulated._find_index_uniform(_tabulated.get_yes(), inverse_coordinate_spacing, _tabulated.get_yes(10) + (coordinate_spacing / 2));

    // }) ;


    // deep_copy(h_result, result);

    // std::cout << "The returned index is = " << h_result(0) << std::endl ;


    // std::cout << "(";
    // for (int i = 0; i < h_yes.size(); ++i){
    //     if (i < h_yes.size() - 1){
    //         std::cout << h_yes(i) << ", ";
    //     } else {
    //         std::cout << h_yes(i) << ")" << std::endl;
    //     }
    // }

    // int coordinate_size = 10;
    // double coordinate_spacing = 0.25;
    // double inverse_coordinate_spacing = 1. / coordinate_spacing;

    // Kokkos::View<double *, grace::default_space> x("CoordinateSystem", coordinate_size);
    // auto h_x = Kokkos::create_mirror_view(x);

    // std::cout << "(";
    // for (int i = 0; i < coordinate_size; ++i){
    //     h_x[i] = i * coordinate_spacing;

    //     if (i < 9){
    //         std::cout << i * coordinate_spacing <<", ";
    //     } else {
    //         std::cout << i * coordinate_spacing <<")";

    //     }
        
    // }

    // Kokkos::deep_copy(x, h_x);

    // Kokkos::View<int[1], grace::default_space> result("result");
    // auto h_result = Kokkos::create_mirror_view(result);


    // double xpos = 1.5;

    // Kokkos::parallel_for("Evol", 1, KOKKOS_LAMBDA (int i) {

    //     result(0) = _tabulated._find_index_uniform(x, inverse_coordinate_spacing, xpos);
    // });

    // Kokkos::deep_copy(h_result, result);

    // GRACE_INFO("The result is {}", h_result(0));
 

    // Kokkos::View<double*, grace::default_space> ltemp("ltemp");
    // auto h_ltemp = Kokkos::create_mirror_view(ltemp);

    // Kokkos::deep_copy(ltemp, _tabulated.get_logtemp()); 


    // auto h_yes = Kokkos::create_mirror_view(_tabulated.get_yes());
    // Kokkos::deep_copy(h_yes, _tabulated.get_yes());

    // std::cout << "(";
    // for (int i = 0; i < h_yes.size(); ++i){
    //     if (i < h_yes.size() - 1){
    //         std::cout << h_yes(i) << ", ";
    //     } else {
    //         std::cout << h_yes(i) << ")" << std::endl;
    //     }
        
    // }

    // std::cout << "The min value is h_l: " << h_yes(0) << std::endl;
    // std::cout << "The max value is h_l: " << h_yes(h_yes.size() - 1) << std::endl;



    // auto h_logrho = Kokkos::create_mirror_view(_tabulated.get_logrho());
    // auto h_logtemp = Kokkos::create_mirror_view(_tabulated.get_logtemp());
    //auto h_yes = Kokkos::create_mirror_view(_tabulated.get_yes());

    // Kokkos::deep_copy(h_logrho, _tabulated.get_logrho());
    // Kokkos::deep_copy(h_logtemp, _tabulated.get_logtemp());
    //Kokkos::deep_copy(h_logyes, _tabulated.get_yes());
    

    // const int N = h_logrho.size();

    // std::cout << "The size of h_logrho is: " << N << std::endl;
    
    // const double dx_lrho = h_logrho(1) - h_logrho(0);
    
    // const double dx_ltemp = h_logtemp(1) - h_logtemp(0);
    
    //const double dx_yes = h_yes(1) - h_yes(0);

    // Kokkos::View<double *, grace::default_space> lrho_midpoints("lrhomidpoint", N - 1);
    // Kokkos::View<double *, grace::default_space> ltemp_midpoints("lrhomidpoint", N - 1);
    // Kokkos::View<double *, grace::default_space> yes_midpoints("lrhomidpoint", N - 1);

    // auto h_lrho_midpoints = Kokkos::create_mirror_view(lrho_midpoints);
    // auto h_temp_midpoints = Kokkos::create_mirror_view(ltemp_midpoints);
    // auto h_yes_midpoints = Kokkos::create_mirror_view(yes_midpoints);


    // //TODO!! Is this the fastest way of working out midpoints would it be quicker
    // //todo x1/2(i) = x(i) + 0.5* (x(i+1) - x(i)). Data has to be transfered from 
    // //host to device namely dx_... .
    // Kokkos::parallel_for("Evol", N - 1, KOKKOS_LAMBDA (int i) {
    //     lrho_midpoints(i) = 0.5 * dx_lrho + _tabulated.get_logrho(i);
    //     ltemp_midpoints(i) = 0.5 * dx_ltemp + _tabulated.get_logtemp(i);
    //     yes_midpoints(i) = 0.5 * dx_yes + _tabulated.get_yes(i);     
        
    // });

    

    
}