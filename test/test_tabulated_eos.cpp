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


TEST_CASE("Tabulated EOS", "[pwpolytrope]")
{
    std::string table_path = "/users/pierrekh/data/eos_tables/compose_tables/DD2+VQCD_soft_quark_fraction.h5";  

    grace::tabulated_eos_t _tabulated = grace::setup_tabulated_eos_compose(table_path.c_str());

    int coordinate_size = 10;
    double coordinate_spacing = 0.25;
    double inverse_coordinate_spacing = 1. / coordinate_spacing;

    Kokkos::View<double *, grace::default_space> x("CoordinateSystem", coordinate_size);
    auto h_x = Kokkos::create_mirror_view(x);

    std::cout << "(";
    for (int i = 0; i < coordinate_size; ++i){
        h_x[i] = i * coordinate_spacing;

        if (i < 9){
            std::cout << i * coordinate_spacing <<", ";
        } else {
            std::cout << i * coordinate_spacing <<")";

        }
        
    }

    Kokkos::deep_copy(x, h_x);

    Kokkos::View<int[1], grace::default_space> result("result");
    auto h_result = Kokkos::create_mirror_view(result);


    double xpos = 1.5;

    Kokkos::parallel_for("Evol", 1, KOKKOS_LAMBDA (int i) {

        result(0) = _tabulated._find_index_uniform(x, inverse_coordinate_spacing, xpos);
    });

    Kokkos::deep_copy(h_result, result);

    GRACE_INFO("The result is {}", h_result(0));
 

    // Kokkos::View<double*, grace::default_space> ltemp("ltemp");
    // auto h_ltemp = Kokkos::create_mirror_view(ltemp);

    // Kokkos::deep_copy(ltemp, _tabulated.get_logtemp()); 


    auto h_yes = Kokkos::create_mirror_view(_tabulated.get_yes());
    Kokkos::deep_copy(h_yes, _tabulated.get_yes());

    std::cout << "(";
    for (int i = 0; i < h_yes.size(); ++i){
        if (i < h_yes.size() - 1){
            std::cout << h_yes(i) << ", ";
        } else {
            std::cout << h_yes(i) << ")" << std::endl;
        }
        
    }

    std::cout << "The min value is h_l: " << h_yes(0) << std::endl;
    std::cout << "The max value is h_l: " << h_yes(h_yes.size() - 1) << std::endl;



    auto h_logrho = Kokkos::create_mirror_view(_tabulated.get_logrho());
    auto h_logtemp = Kokkos::create_mirror_view(_tabulated.get_logtemp());
    //auto h_yes = Kokkos::create_mirror_view(_tabulated.get_yes());

    Kokkos::deep_copy(h_logrho, _tabulated.get_logrho());
    Kokkos::deep_copy(h_logtemp, _tabulated.get_logtemp());
    //Kokkos::deep_copy(h_logyes, _tabulated.get_yes());
    

    const int N = h_logrho.size();

    std::cout << "The size of h_logrho is: " << N << std::endl;
    
    const double dx_lrho = h_logrho(1) - h_logrho(0);
    
    const double dx_ltemp = h_logtemp(1) - h_logtemp(0);
    
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