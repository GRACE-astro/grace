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


    GRACE_INFO("BREAKPOINT 1");

    double xpos = 1.5;

    Kokkos::parallel_for("Evol", 1, KOKKOS_LAMBDA (int i) {

        result(0) = _tabulated._find_index_uniform(x, inverse_coordinate_spacing, xpos);
    });

    Kokkos::deep_copy(h_result, result);

    GRACE_INFO("The result is {}", h_result(0));

    
}