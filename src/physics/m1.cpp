/**
 * @file m1.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2024-11-24
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

// configuration
#include <grace_config.h>

// general headers
#include <grace/utils/device.h>
#include <grace/utils/inline.h>

// config 
#include <grace/config/config_parser.hh>

// grid
#include <grace/amr/amr_functions.hh>

// utilities
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

// m1 includes
#include <grace/physics/m1.hh>
#include <grace/physics/m1_helpers.hh>
#include <grace/physics/eas_policies.hh>
#include <grace/physics/id/m1_initial_data.hh>

// grmhd + eos includes 
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>

// Kokkos 
#include <Kokkos_Core.hpp>

// STL
#include <string> 

namespace grace {

template < typename eos_t >
void set_m1_eas() {
    auto& state = grace::variable_list::get().getstate() ; 
    auto& sstate = grace::variable_list::get().getstagstate() ; 
    auto& aux = grace::variable_list::get().getaux() ;
    set_m1_eas(state,sstate,aux) ; 
}


template < typename eos_t >
void set_m1_eas(
      grace::var_array_t& state
    , grace::staggered_variable_arrays_t& sstate
    , grace::var_array_t& aux
) 
{
    using namespace grace  ;
    using namespace Kokkos ;
    
    DECLARE_GRID_EXTENTS ;

    auto eos = eos::get().get_eos<eos_t>() ;  

    auto eas_kind = grace::get_param<std::string>("m1","eas","kind") ; 

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
    
    if ( kind == "test" ) {
        coord_array_t<GRACE_NSPACEDIM> cart_pcoords ; 
        grace::fill_physical_coordinates(cart_pcoords,grace::STAG_CENTER,/*cartesian coords*/ false) ;
        test_eas_op op(aux) ; 
        parallel_for(GRACE_EXECUTION_TAG("EVOL","compute_eas"), policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
            {
                double xyz[3] = {
                    cart_pcoords(VEC(i,j,k),0,q),
                    cart_pcoords(VEC(i,j,k),1,q),
                    cart_pcoords(VEC(i,j,k),2,q)
                } ; 
                op(VEC(i,j,k),q,xyz) ; 
            }
        ); 
    } else {
        ERROR("EAS computation method not supported.") ; 
    }
}


template < typename id_kernel_t > 
static void set_m1_initial_data_impl(
    id_kernel_t id_kernel
)
{
    using namespace grace  ;
    using namespace Kokkos ;
    
    DECLARE_GRID_EXTENTS ;

    auto& state = grace::variable_list::get().getstate() ; 
    auto& sstate = grace::variable_list::get().getstagstate() ; 
    auto& aux = grace::variable_list::get().getaux() ; 

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
    parallel_for(
        GRACE_EXECUTION_TAG("ID","set_m1_id"),
        policy,
        KOKKOS_LAMBDA (VEC(int const i, int const j, int const k), int const q) {


        }
    ) ; 
}

template < typename eos_t >
void set_m1_initial_data() {
    using namespace grace  ;
    using namespace Kokkos ;
    
    DECLARE_GRID_EXTENTS ;


    auto eos = eos::get().get_eos<eos_t>() ;
    

    auto id_type = grace::get_param<std::string>("m1","id_type") ; 

    if ( id_type == "straight_beam" ) {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ; 

    } else if (id_type == "curved_beam" ) {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ; 

    } else if ( id_type == "scattering") {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ; 

    } else if ( id_type == "shadow" ) {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ; 
        
    } else if ( id_type == "emitting_sphere") {
        auto hydro_id_type = grace::get_param<std::string>("grmhd","id_type") ;
        ASSERT(hydro_id_type=="minkowski_vacuum", "For M1 tests the hydro must be set to minkowski_vacuum") ; 

    } else if ( id_type == "zero") {

    }

    // now set eas 
    set_m1_eas<eos_t>() ; 

}

/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)        \
template                                 \
void set_m1_initial_data<EOS>( );        \
template                                 \
void set_m1_eas<EOS>(                    \
      grace::var_array_t&                \
    , grace::staggered_variable_arrays_t&\
    , grace::var_array_t&                \
)                                        \
template                                 \
void set_m1_eas<EOS>()


INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/

} // namespace grace 
