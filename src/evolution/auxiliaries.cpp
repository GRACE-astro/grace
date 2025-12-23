/**
 * @file auxiliaries.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-13
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#include <grace/evolution/auxiliaries.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#ifdef GRACE_ENABLE_BURGERS 
#include <grace/physics/burgers.hh>
#endif 
#ifdef GRACE_ENABLE_SCALAR_ADV
#include <grace/physics/scalar_advection.hh>
#endif
#ifdef GRACE_ENABLE_GRMHD
//#include <grace/physics/admbase.hh>
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#endif
#ifdef GRACE_ENABLE_BSSN_METRIC
#include <grace/physics/bssn.hh>
#include <grace/physics/bssn_helpers.hh>
#endif 
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/physics/eos/eos_types.hh>

#include <Kokkos_Core.hpp>  
#include <cmath>
namespace grace {


void compute_auxiliary_quantities() {
    auto& state = grace::variable_list::get().getstate() ; 
    auto& sstate = grace::variable_list::get().getstaggeredstate() ; 
    auto& aux   = grace::variable_list::get().getaux()   ;
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            grace::get_param<std::string>("eos", "cold_eos_type") ;
        if( cold_eos_type == "piecewise_polytrope" ) {
            compute_auxiliary_quantities<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>(state,sstate,aux) ; 
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
    
}

template< typename eos_t >
void compute_auxiliary_quantities(
      grace::var_array_t& state
    , grace::staggered_variable_arrays_t& sstate
    , grace::var_array_t& aux  ) 
{
    Kokkos::Profiling::pushRegion("Compute auxiliaries") ;
    GRACE_VERBOSE("Computing auxiliary quantities at iteration {}", grace::get_iteration()) ; 
     
    using namespace grace ; 
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& idx     = grace::variable_list::get().getinvspacings() ;  

    #ifdef GRACE_ENABLE_GRMHD

    auto eos = eos::get().get_eos<eos_t>() ;  
    atmo_params_t atmo_params ; 
    
    atmo_params.rho_fl = grace::get_param<double>("grmhd","atmosphere","rho_fl") ; 
    atmo_params.temp_fl = grace::get_param<double>("grmhd","atmosphere","temp_fl") ; 
    atmo_params.ye_fl = grace::get_param<double>("grmhd","atmosphere","ye_fl") ; 

    atmo_params.rho_fl_scaling = grace::get_param<double>("grmhd","atmosphere","rho_scaling") ; 
    atmo_params.temp_fl_scaling = grace::get_param<double>("grmhd","atmosphere","temp_scaling") ;


    auto excision_pars = grace::get_param<YAML::Node>("grmhd","excision") ;
    excision_params_t excision_params ; 
    auto excision_kind = grace::get_param<std::string>("grmhd","excision","excision_criterion"); 
    //excision_pars["excision_criterion"].as<std::string>() ;
    if ( excision_kind == "radius" ) {
        excision_params.excise_by_radius = true ;
    } else if ( excision_kind == "lapse") {
        excision_params.excise_by_radius = false ;
    } else {
        ERROR("Unrecognized excision criterion") ; 
    }
    excision_params.r_ex = grace::get_param<double>("grmhd","excision","excision_radius"); 
    excision_params.alp_ex = grace::get_param<double>("grmhd","excision","excision_lapse"); 
    
    excision_params.rho_ex =  grace::get_param<double>("grmhd","excision","rho_excision"); 
    excision_params.temp_ex =  grace::get_param<double>("grmhd","excision","temp_excision"); 

    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,state,sstate,aux,atmo_params,excision_params) ; 
    #define GET_AUX \
    grmhd_eq_system(auxiliaries_computation_kernel_t{}, VEC(i,j,k), q, pcoords)
    #else 
    #define GET_AUX
    #endif 

    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    grace::fill_physical_coordinates(pcoords) ;

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ; 
    parallel_for(GRACE_EXECUTION_TAG("EVOL","get_auxiliaries"), policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        GET_AUX ; 
        auto Bx = Kokkos::subview(sstate.face_staggered_fields_x,
                                 VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), BSX_, q) ; 
        auto By = Kokkos::subview(sstate.face_staggered_fields_y,
                                 VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), BSY_, q) ; 
        auto Bz = Kokkos::subview(sstate.face_staggered_fields_z,
                                 VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), BSZ_, q) ;   
        aux(VEC(i,j,k),BDIV_,q) = (Bx(VEC(i+1,j,k)) - Bx(VEC(i,j,k))) * idx(0,q) 
                                + (By(VEC(i,j+1,k)) - By(VEC(i,j,k))) * idx(1,q)
                                + (Bz(VEC(i,j,k+1)) - Bz(VEC(i,j,k))) * idx(2,q) ; 
        #if 0
        if ( Kokkos::fabs(aux(VEC(i,j,k),BDIV_,q)) > 1e-15) {
            printf("DIVB! i %d j %d k %d q %d DBX (%.10f-%.10f) DBY (%.10f-%.10f) DBZ (%.10f-%.10f) DIVB %.3g\n",
            i,j,k,q,
            Bx(VEC(i+1,j,k)), Bx(VEC(i,j,k)),
            By(VEC(i,j+1,k)),By(VEC(i,j,k)),
            Bz(VEC(i,j,k+1)),Bz(VEC(i,j,k)),
            aux(VEC(i,j,k),BDIV_,q)
        ) ; 
        
        }
        #endif
    }) ; 


    #ifdef GRACE_ENABLE_BSSN_METRIC
    auto k1 = grace::get_param<double>("bssn","k1") ;
    auto eta = grace::get_param<double>("bssn","eta") ;
    auto epsdiss = grace::get_param<double>("bssn","epsdiss") ; 
    bssn_system_t bssn_eq_system(state,aux,sstate,k1,eta,epsdiss) ; 

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
            interior_policy({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
        
    parallel_for(GRACE_EXECUTION_TAG("EVOL","get_bssn_auxiliaries"), interior_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        bssn_eq_system.template compute_constraint_violations<4>(
            aux, 
            {idx(0,q),idx(1,q),idx(2,q)}, 
            VEC(i,j,k), 
            q) ; 
 
    }) ; 
    #endif 


    #undef GET_AUX
    Kokkos::Profiling::popRegion() ; 
}
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                       \
template                                                                \
void compute_auxiliary_quantities<EOS>(                                 \
                           grace::var_array_t&         \
                         , grace::staggered_variable_arrays_t& \
                         , grace::var_array_t& aux )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}