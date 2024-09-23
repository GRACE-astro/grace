/**
 * @file evaluate_regrid_criterion.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-23
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

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/regrid_helpers.tpp>
#include <grace/amr/regrid_helpers.hh>
#include <grace/data_structures/grace_data_structures.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

void evaluate_regrid_criterion(
      std::string const& criterion
    , std::string const& criterion_var
    , Kokkos::View<int *, grace::default_space>& regrid_flags
)
{
    using namespace grace  ; 
    using namespace Kokkos ;

    if ( criterion == "FLASH_second_deriv" ) {
        double const eps = get_param<double>("amr", "FLASH_criterion_eps") ; 
        auto u = get_variable_subview(criterion_var) ; 
        amr::flash_second_deriv_criterion<decltype(u)> kernel{ u } ; 
        evaluate_regrid_criterion_kernel(
            d_regrid_flags,
            kernel,
            eps
        ) ; 
    } else if ( criterion == "simple_threshold" ) {
        auto u = get_variable_subview(criterion_var) ; 
        amr::simple_threshold_criterion<decltype(u)> kernel{ u } ;
        evaluate_regrid_criterion_kernel(
            d_regrid_flags,
            kernel,
        ) ;  
    } else if ( criterion == "gradient" ) {
        auto u = get_variable_subview(criterion_var) ; 
        amr::gradient_criterion<decltype(u)> kernel{ u } ;
        evaluate_regrid_criterion_kernel(
            d_regrid_flags,
            kernel,
        ) ;
    } else if ( criterion == "shear" ) {
        auto vx =  get_variable_subview("vel[0]") ; 
        auto vy =  get_variable_subview("vel[1]") ; 
        #ifdef GRACE_3D 
        auto vz =  get_variable_subview("vel[2]") ; 
        #endif 
        amr::shear_criterion<decltype(vx)> kernel{ VEC(vx,vy,vz) } ; 
        evaluate_regrid_criterion_kernel( d_regrid_flags
                                 , kernel) ;
    } else {
        ERROR("Unsupported refinement criterion.") ; 
    }
}

}} /* namespace grace::amr */