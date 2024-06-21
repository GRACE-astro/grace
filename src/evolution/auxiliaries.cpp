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
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
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
    auto& aux   = grace::variable_list::get().getaux()   ;
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            grace::get_param<std::string>("eos", "cold_eos_type") ;
        if( cold_eos_type == "piecewise_polytrope" ) {
            compute_auxiliary_quantities<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>(state,aux) ; 
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
    
}

template< typename eos_t >
void compute_auxiliary_quantities(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& aux  ) 
{
    Kokkos::Profiling::pushRegion("Compute auxiliaries") ;
    GRACE_VERBOSE("Computing auxiliary quantities at iteration {}", grace::get_iteration()) ; 
     
    using namespace grace ; 
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;

    #ifdef GRACE_ENABLE_GRMHD
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t,weno_reconstructor_t<3>,hll_riemann_solver_t>
        grmhd_eq_system(eos,state,aux) ; 
    #define GET_AUX \
    grmhd_eq_system(auxiliaries_computation_kernel_t{}, VEC(i,j,k), q)
    #else 
    #define GET_AUX
    #endif 

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ; 
    parallel_for(GRACE_EXECUTION_TAG("EVOL","get_auxiliaries"), policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        GET_AUX ; 
    }) ; 

    #undef GET_AUX
    #if 0
    #ifdef GRACE_ENABLE_SCALAR_ADV
    auto& params = grace::config_parser::get() ; 
    double VEC(ax,ay,az) ; 
    EXPR(
    ax = params["scalar_advection"]["ax"].as<double>() ;,
    ay = params["scalar_advection"]["ay"].as<double>() ;,
    az = params["scalar_advection"]["az"].as<double>() ; )

    double sigma = params["scalar_advection"]["gaussian_sigma"].as<double>() ;
    EXPR(
    double xc = params["scalar_advection"]["gaussian_x_c"].as<double>() ;,
    double yc = params["scalar_advection"]["gaussian_y_c"].as<double>() ;,
    double zc = params["scalar_advection"]["gaussian_z_c"].as<double>() ; 
    )
    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_aux_mirror = Kokkos::create_mirror_view(aux) ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    Kokkos::deep_copy(h_state_mirror, state) ; 
    Kokkos::deep_copy(h_aux_mirror, aux)     ; 

    double t = grace::get_simulation_time() ; 

    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif 
        /* Physical coordinates of cell center */
        auto pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            true
        ) ; 
        EXPR(
        double const x0 = 
            xc + t * ax - pcoords[0] ;, 
        double const y0 = 
            yc + t * ay - pcoords[1] ;, 
        double const z0 = 
            zc + t * az - pcoords[2] ; 
        ) 
        double const r = sqrt( EXPR(
            math::int_pow<2>(x0), + math::int_pow<2>(y0), + math::int_pow<2>(z0)
        )) ; 
        double const exact_solution = exp(- 0.5 * math::int_pow<2>(r)/math::int_pow<2>(sigma)) / sigma / sqrt(2*M_PI) ; 

        h_aux_mirror(VEC(i,j,k),ERR,q) = fabs( exact_solution - h_state_mirror(VEC(i,j,k),U,q))  ; 
    }
    Kokkos::deep_copy(aux,h_aux_mirror) ; 
    #endif 
    #endif 
    Kokkos::Profiling::popRegion() ; 
}
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                       \
template                                                                \
void compute_auxiliary_quantities<EOS>(                                 \
                           grace::var_array_t<GRACE_NSPACEDIM>&         \
                         , grace::var_array_t<GRACE_NSPACEDIM>& aux )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}